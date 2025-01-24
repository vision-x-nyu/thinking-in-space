import io
import json
import os
import re
import time
from typing import List, Tuple

from accelerate import Accelerator, DistributedType
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm

from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

import hashlib

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmBlockThreshold, HarmCategory
    from google.api_core.exceptions import ResourceExhausted

    NUM_SECONDS_TO_SLEEP = os.getenv("GEMINI_INTV_AFTER_FAILED", 10)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
    genai.configure(api_key=GOOGLE_API_KEY)

except Exception as e:
    eval_logger.error(f"Error importing generativeai: {str(e)}, try to install genai by ```pip install google-generativeai```")
    genai = None

OBJ_REL_DISTANCE_TEMPLATE = """
Measuring from the closest point of each object, which of these objects ({choice_a}, {choice_b}, {choice_c}, {choice_d}) is the closest to the {category}?
""".strip()

OBJ_REL_DIRECTION_HARD_TEMPLATE = """
If I am standing by the {positioning_object} and facing the {orienting_object}, is the {querying_object} to my front-left, front-right, back-left, or back-right?
The directions refer to the quadrants of a Cartesian plane (if I am standing at the origin and facing along the positive y-axis).
""".strip()

OBJ_REL_DIRECTION_MEDIUM_TEMPLATE = """
If I am standing by the {positioning_object} and facing the {orienting_object}, is the {querying_object} to my left, right, or back?
An object is to my back if I would have to turn at least 135 degrees in order to face it.
""".strip()

OBJ_REL_DIRECTION_EASY_TEMPLATE = """
If I am standing by the {positioning_object} and facing the {orienting_object}, is the {querying_object} to the left or the right of the {orienting_object}?
""".strip()


def extract_categories_of_interest(doc):
    if doc['question_type'].startswith('object_rel_direction'):
        if doc['question_type'] == "object_rel_direction_hard":
            template = OBJ_REL_DIRECTION_HARD_TEMPLATE
        elif doc['question_type'] == "object_rel_direction_medium":
            template = OBJ_REL_DIRECTION_MEDIUM_TEMPLATE
        elif doc['question_type'] == "object_rel_direction_easy":
            template = OBJ_REL_DIRECTION_EASY_TEMPLATE
        pattern = re.escape(template)
        pattern = pattern.replace(r'\{positioning_object\}', r'(?P<positioning_object>.+?)')
        pattern = pattern.replace(r'\{orienting_object\}', r'(?P<orienting_object>.+?)')
        pattern = pattern.replace(r'\{querying_object\}', r'(?P<querying_object>.+?)')
        
        if doc['question_type'] == "object_rel_direction_easy":
            pattern = re.compile(
                r"^If I am standing by the "
                r"(?P<positioning_object>.*?)"
                r" and facing the "
                r"(?P<orienting_object>.*?)"
                r", is the "
                r"(?P<querying_object>.*?)"
                r" to the left or the right of the "
                r"(?P=orienting_object)\?$"
            )
    elif doc['question_type'].startswith('object_rel_distance'):
        pattern = re.escape(OBJ_REL_DISTANCE_TEMPLATE)
        pattern = pattern.replace(r'\{choice_a\}', r'(?P<choice_a>.+?)')
        pattern = pattern.replace(r'\{choice_b\}', r'(?P<choice_b>.+?)')
        pattern = pattern.replace(r'\{choice_c\}', r'(?P<choice_c>.+?)')
        pattern = pattern.replace(r'\{choice_d\}', r'(?P<choice_d>.+?)')
        pattern = pattern.replace(r'\{category\}', r'(?P<category>.+?)')
    
    match = re.match(pattern, doc['question'])
    if match:
        return match.groupdict()
    else:
        return None

COGMAP_PROMPT_TEMPLATE = """[Task]
This video captures an indoor scene. Your objective is to identify specific objects within the video, understand the spatial arrangement of the scene, and estimate the center point of each object, assuming the entire scene is represented by a 10x10 grid.

[Rule]
1. We provide the categories to care about in this scene: {categories_of_interest}. Focus ONLY on these categories.
2. Estimate the center location of each instance within the provided categories, assuming the entire scene is represented by a 10x10 grid.
3. If a category contains multiple instances, include all of them.
4. Each object's estimated location should accurately reflect its real position in the scene, preserving the relative spatial relationships among all objects.

[Output]
Present the estimated center locations for each object as a list within a dictionary.
STRICTLY follow this JSON format:
{{"category name": [(x_1, y_1), ...], ...}}"""

@register_model("gemini_api")
class GeminiAPI(lmms):
    def __init__(
        self,
        model_version: str = "gemini-1.5-pro",
        modality: str = "image",
        timeout: int = 120,
        continual_mode: bool = False,
        response_persistent_folder: str = None,  # We will cache the Gemini API response in this path and use it for future requests
        **kwargs,
    ) -> None:
        super().__init__()
        self.model_version = model_version
        self.timeout = timeout
        self.generation_config = {'temperature': 0, 'top_p': 1, 'top_k': 1, 'max_output_tokens': 8192}
        self.model = genai.GenerativeModel(model_version, generation_config=self.generation_config)
        self.continual_mode = continual_mode
        if self.continual_mode and response_persistent_folder is None:
            raise ValueError("Continual mode requires a persistent path for the response. We will cache the Gemini API response in this path and use it for future requests. Please provide a valid path.")
        self.response_persistent_folder = response_persistent_folder
        if self.continual_mode:
            if not os.path.exists(self.response_persistent_folder):
                os.makedirs(self.response_persistent_folder)
            self.response_persistent_file = os.path.join(self.response_persistent_folder, f"{self.model_version}_response.json")
            if os.path.exists(self.response_persistent_file):
                with open(self.response_persistent_file, "r") as f:
                    self.response_cache = json.load(f)
                self.cache_mode = "resume"
            else:
                self.response_cache = {}
                self.cache_mode = "start"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            assert self.continual_mode is False, "Continual mode is not supported with distributed inference."
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

        self.modality = modality
        
    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def get_image_size(self, image):
        # Create a BytesIO object to store the image bytes
        img_byte_array = io.BytesIO()

        # Save the image to the BytesIO object
        image.save(img_byte_array, format="PNG")

        # Get the size of the BytesIO object
        img_size = img_byte_array.tell()

        return img_size

    def encode_video(self, video_path):
        uploaded_obj = genai.upload_file(path=video_path)
        time.sleep(5)
        return uploaded_obj

    def encode_image(self, image_path):
        uploaded_obj = genai.upload_file(path=image_path)
        time.sleep(5)
        return uploaded_obj

    def convert_video(self, images):
        for idx, img in enumerate(images):
            if self.modality == "video" and isinstance(img, str):
                try:
                    images[idx] = self.encode_video(img)
                except Exception as e:
                    eval_logger.error(f"Error converting video: {str(e)}")
            elif self.modality == 'image' and isinstance(img, str):
                try:
                    images[idx] = self.encode_image(img)
                except Exception as e:
                    eval_logger.error(f"Error converting image: {str(e)}")
        return images

    def generate_until(self, requests) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        def get_uuid(task, split, doc_id):
            return f"{task}___{split}___{doc_id}"

        for contexts, gen_kwargs, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            if self.continual_mode is True and self.cache_mode == "resume":
                doc_uuid = get_uuid(task, split, doc_id)
                if doc_uuid in self.response_cache:
                    content = self.response_cache[doc_uuid]
                    if content:
                        res.append(content)
                        pbar.update(1)
                        continue

            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            if visuals != [None]:
                visuals = self.flatten(visuals)
                visuals = self.convert_video(visuals)

            categories_of_interest = extract_categories_of_interest(self.task_dict[task][split][doc_id])
            assert categories_of_interest is not None
            categories_of_interest = list(categories_of_interest.values())
            
            messages = [COGMAP_PROMPT_TEMPLATE.format(categories_of_interest=categories_of_interest), contexts]

            contents = []
            for index, message in enumerate(messages):
                if index == 0:
                    if visuals != [None]: # always follow gemini's suggestion that take video first
                        chat_session = self.model.start_chat(
                            history=[
                                {
                                    "role": "user",
                                    "parts": [visuals[0]]
                                },
                            ])
                    else:
                        chat_session = self.model.start_chat()

                for attempt in range(5):
                    try:
                        content = chat_session.send_message(message)
                        content = content.text
                        if isinstance(message, str):
                            contents.append(message)
                        contents.append(content)
                        print(message)
                        print(content)
                        break
                    except Exception as e:
                        eval_logger.info(f"Attempt {attempt + 1} failed with error: {str(e)}")
                        if isinstance(e, ValueError):
                            try:
                                eval_logger.info(f"Prompt feed_back: {content.prompt_feedback}")
                                content = ""
                                break
                            except Exception:
                                pass
                        if attempt < 5 - 1:  # If we have retries left, sleep and then continue to next attempt
                            time.sleep(NUM_SECONDS_TO_SLEEP)
                        else:  # If this was the last attempt, log and return empty
                            eval_logger.error(f"All 5 attempts failed. Last error message: {str(e)}")
                            content = ""
                            if isinstance(e, ResourceExhausted):
                                eval_logger.error("Quota exceed!!!")
            res.append(json.dumps(contents))
            pbar.update(1)

            if self.continual_mode is True:  # Cache the response
                doc_uuid = get_uuid(task, split, doc_id)
                self.response_cache[doc_uuid] = content
                with open(self.response_persistent_file, "w") as f:
                    json.dump(self.response_cache, f)

        pbar.close()
        return res

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        assert False, "Gemini API not support"

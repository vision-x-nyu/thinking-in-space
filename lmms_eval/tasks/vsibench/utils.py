
import os
import json
from pathlib import Path
import yaml
from loguru import logger as eval_logger
from functools import partial
import numpy as np
import pandas as pd

import datasets

MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]
NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]

METRICS_FOR_MCA = {
    "accuracy": "exact_match",
}

METRICS_FOR_NA = {
    "MRA:.5:.95:.05": "partial(mean_relative_accuracy, start=.5, end=.95, interval=.05)",
}


hf_home = os.getenv("HF_HOME", "~/.cache/huggingface/")
base_cache_dir = os.path.expanduser(hf_home)
with open(Path(__file__).parent / "vsibench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def vsibench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    video_path = doc["dataset"] + "/" + doc["scene_name"] + ".mp4"
    video_path = os.path.join(cache_dir, video_path)
    if os.path.exists(video_path):
        video_path = video_path
    else:
        raise FileExistsError(f"video path:{video_path} does not exist.")
    return [video_path]


def vsibench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
        
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "") or "These are frames of a video."
    
    if doc['question_type'] in NA_QUESTION_TYPES:
        post_prompt = lmms_eval_specific_kwargs.get("na_post_prompt", "") or "Please answer the question using a single word or phrase."
        return pre_prompt + "\n" + question + "\n" + post_prompt
    elif doc['question_type'] in MCA_QUESTION_TYPES:
        options = "Options:\n" + "\n".join(doc["options"])
        post_prompt = lmms_eval_specific_kwargs.get("mca_post_prompt", "") or "Answer with the option's letter from the given choices directly."
        return "\n".join([pre_prompt, question, options, post_prompt])
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    if os.getenv('LMMS_EVAL_SHUFFLE_DOCS', None):
        eval_logger.info(f"Environment variable LMMS_EVAL_SHUFFLE_DOCS detected, dataset will be shuffled.")
        return dataset.shuffle(seed=42)
    subset_indexes = [3, 9, 22, 27, 37, 41, 42, 80, 81, 91, 106, 125, 141, 145, 151, 153, 154, 166, 182, 188, 202, 206, 219, 285, 289, 379, 380, 423, 427, 435, 443, 446, 462, 463, 485, 512, 515, 521, 530, 534, 536, 552, 556, 562, 565, 586, 594, 602, 608, 609, 610, 613, 616, 617, 619, 626, 654, 661, 662, 666, 672, 673, 675, 679, 686, 696, 699, 717, 825, 839, 842, 869, 874, 879, 913, 927, 935, 968, 987, 1023, 1036, 1049, 1059, 1125, 1149, 1164, 1199, 1213, 1215, 1235, 1271, 1292, 1321, 1338, 1358, 1359, 1373, 1376, 1378, 1381, 1395, 1405, 1417, 1447, 1462, 1463, 1471, 1472, 1479, 1485, 1500, 1519, 1528, 1531, 1554, 1560, 1570, 1571, 1575, 1582, 1584, 1601, 1645, 1670, 1682, 1689, 1690, 1713, 1741, 1762, 1795, 1838, 1855, 1933, 2019, 2023, 2051, 2060, 2073, 2093, 2101, 2107, 2136, 2140, 2145, 2151, 2154, 2224, 2231, 2233, 2242, 2295, 2309, 2333, 2336, 2357, 2370, 2372, 2373, 2392, 2394, 2413, 2424, 2440, 2478, 2488, 2491, 2496, 2497, 2557, 2559, 2568, 2574, 2586, 2615, 2635, 2652, 2653, 2663, 2672, 2678, 2681, 2682, 2687, 2689, 2693, 2698, 2709, 2715, 2747, 2752, 2757, 2771, 2786, 2794, 2795, 2797, 2819, 2851, 2869, 2871, 2876, 2891, 2894, 2897, 2910, 2919, 2969, 2975, 2978, 2985, 3016, 3027, 3046, 3070, 3072, 3075, 3089, 3102, 3133, 3143, 3150, 3179, 3180, 3182, 3306, 3307, 3312, 3313, 3329, 3348, 3367, 3390, 3401, 3404, 3432, 3464, 3532, 3561, 3585, 3586, 3657, 3666, 3697, 3711, 3712, 3716, 3728, 3730, 3767, 3778, 3780, 3781, 3783, 3787, 3806, 3814, 3819, 3832, 3833, 3836, 3839, 3842, 3859, 3874, 3880, 3882, 3886, 3938, 3950, 3956, 3966, 3967, 3975, 4011, 4052, 4074, 4077, 4082, 4084, 4102, 4139, 4144, 4145, 4147, 4149, 4171, 4180, 4183, 4184, 4188, 4198, 4214, 4237, 4240, 4243, 4298, 4316, 4330, 4359, 4362, 4364, 4376, 4381, 4387, 4403, 4405, 4424, 4430, 4438, 4453, 4457, 4458, 4476, 4507, 4508, 4512, 4547, 4551, 4558, 4565, 4583, 4585, 4630, 4645, 4661, 4663, 4689, 4697, 4698, 4707, 4750, 4764, 4768, 4773, 4798, 4818, 4838, 4840, 4844, 4870, 4874, 4890, 4900, 4922, 4923, 4925, 4932, 4939, 4945, 4963, 4964, 4965, 4966, 4970, 4972, 4974, 4975, 4978, 4983, 4989, 4990, 4991, 4993, 5002, 5005, 5007, 5008, 5024, 5026, 5038, 5040, 5044, 5045, 5048, 5049, 5050, 5051, 5055, 5058, 5063, 5064, 5071, 5075, 5090, 5103, 5112, 5113, 5114, 5117, 5120, 5121, 5124, 5131, 5134, 5135, 5136, 5139, 5146, 5147]
    datasets = dataset.filter(lambda x: x['id'] in subset_indexes).filter(lambda x: x['question_type'].startswith("object_rel_distance") or x['question_type'].startswith('object_rel_direction'))
    return datasets

def fuzzy_matching(pred):
    return pred.split(' ')[0].rstrip('.').strip()

def exact_match(pred, target):
    return 1. if pred.lower() == target.lower() else 0.

def abs_dist_norm(pred, target):
    return abs(pred - target) / target

def mean_relative_accuracy(pred, target, start, end, interval):
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()

WORST_CASE_FOR_METRICS = {
    "accuracy": 0.,
    "MRA:.5:.95:.05": 0.,
}

def to_float(pred):
    try:
        pred = float(pred)
    except BaseException as e:
        pred = None
    return pred

def vsibench_process_results(doc, results):
    
    doc['prediction'] = json.loads(results[0])[-1]
    if doc['question_type'] in MCA_QUESTION_TYPES:
        for key, value in METRICS_FOR_MCA.items():
            doc[key] = eval(value)(fuzzy_matching(doc['prediction']), doc['ground_truth'])
    elif doc['question_type'] in NA_QUESTION_TYPES:
        for key, value in METRICS_FOR_NA.items():
            try:
                doc[key] = eval(value)(to_float(fuzzy_matching(doc['prediction'])), to_float(doc['ground_truth']))
            except TypeError:
                doc[key] = WORST_CASE_FOR_METRICS[key]
    else:
        raise ValueError(f"Unknown question type: {doc['question_type']}")

    return {"vsibench_score": doc}

def vsibench_aggregate_results(results):
    results = pd.DataFrame(results)
    
    output = {}

    for question_type, question_type_indexes in results.groupby('question_type').groups.items():
        per_question_type = results.iloc[question_type_indexes]
        
        if question_type in MCA_QUESTION_TYPES:
            for metric in METRICS_FOR_MCA.keys():
                output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
        elif question_type in NA_QUESTION_TYPES:
            for metric in METRICS_FOR_NA.keys():
                if metric == 'success_rate':
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()
                else:
                    output[f"{question_type}_{metric}"] = per_question_type[metric].mean()

        else:
            raise ValueError(f"Unknown question type: {question_type}")
    
    output['object_rel_direction_accuracy'] = sum([
        output.pop('object_rel_direction_easy_accuracy', 0.),
        output.pop('object_rel_direction_medium_accuracy', 0.),
        output.pop('object_rel_direction_hard_accuracy', 0.),
    ]) / 3.
    
    output['overall'] = sum([_ for _ in output.values()]) / len(output)
    eval_logger.info(f"Evaluation results: {output}")
    return output

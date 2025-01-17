#!/bin/bash

set -e

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
else
    IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
    gpu_count=${#devices[@]}
fi

export OPENAI_API_KEY="" # API KEY FOR OPENAI CHATGPT
export GOOGLE_API_KEY="" # API KEY FOR GOGOLE GEMINI

benchmark=vsibench
output_path=logs/$(TZ="America/New_York" date "+%Y%m%d")
num_processes=4
num_frames=32
launcher=accelerate

available_models="gemini_1p5_flash"

while [[ $# -gt 0 ]]; do
    case "$1" in
    --benchmark)
        benchmark="$2"
        shift 2
        ;;
    --num_processes)
        num_processes="$2"
        shift 2
        ;;
    --model)
        IFS=',' read -r -a models <<<"$2"
        shift 2
        ;;
    --output_path)
        output_path="$2"
        shift 2
        ;;
    --limit)
        limit="$2"
        shift 2
        ;;
    *)
        echo "Unknown argument: $1"
        exit 1
        ;;
    esac
done

if [ "$models" = "all" ]; then
    IFS=',' read -r -a models <<<"$available_models"
fi

for model in "${models[@]}"; do
    echo "Start evaluating $model..."

    case "$model" in
    "gemini_1p5_flash")
        model_family="gemini_api"
        model_args="model_version=gemini-1.5-flash,modality=video"
        ;;
    "gemini_1p5_pro_002")
        model_family="gemini_api"
        model_args="model_version=gemini-1.5-pro,modality=video"
        ;;
    "gemini_2p0_flash_exp")
        model_family="gemini_api"
        model_args="model_version=gemini-2.0-flash-exp,modality=video"
        ;;
    *)
        echo "Unknown model: $model"
        exit -1
        ;;
    esac

    if [ "$launcher" = "python" ]; then
        export LMMS_EVAL_LAUNCHER="python"
        evaluate_script="python \
            "
    elif [ "$launcher" = "accelerate" ]; then
        export LMMS_EVAL_LAUNCHER="accelerate"
        evaluate_script="accelerate launch \
            --num_processes=$num_processes \
            "
    fi

    evaluate_script="$evaluate_script -m lmms_eval \
        --model $model_family \
        --model_args $model_args \
        --tasks $benchmark \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix $model \
        --output_path $output_path/$benchmark \
        "

    if [ -n "$limit" ]; then
        evaluate_script="$evaluate_script \
            --limit $limit \
        "
    fi
    echo $evaluate_script
    eval $evaluate_script
done

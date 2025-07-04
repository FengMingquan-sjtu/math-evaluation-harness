set -ex

PROMPT_TYPE="direct" # direct / cot / pal / tool-integrated
MODEL_NAME_OR_PATH=$1
OUTPUT_DIR=$2
# ======= nanogpt ======
#CUDA_VISIBLE_DEVICES=1 bash scripts/run_eval.sh ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.5rho/2025-06-27_16-52-44/ckpt-30000.pt ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.5rho/2025-06-27_16-52-44/ckpt-30000/math_eval
#CUDA_VISIBLE_DEVICES=1 bash scripts/run_eval.sh ../nanoGPT/out/cont-gpt2-owm-37B/ckpt.pt ../nanoGPT/out/cont-gpt2-owm-37B/math_eval
#CUDA_VISIBLE_DEVICES=1 bash scripts/run_eval.sh ../nanoGPT/out/cont-gpt2-124M-owm-15B/ckpt-30000.pt ../nanoGPT/out/cont-gpt2-124M-owm-15B/ckpt-30000/math_eval
#CUDA_VISIBLE_DEVICES=1 bash scripts/run_eval.sh ../nanoGPT/out/cont-gpt2-124M-owm-15B/ckpt-14000.pt ../nanoGPT/out/cont-gpt2-124M-owm-15B/ckpt-14000/math_eval


#CUDA_VISIBLE_DEVICES=0 bash scripts/run_eval.sh ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.1rho/2025-06-29_16-26-24/ckpt-100000.pt ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.1rho/2025-06-29_16-26-24/ckpt-100000/math_eval
#CUDA_VISIBLE_DEVICES=4 bash scripts/run_eval.sh ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.2rho/2025-06-29_15-38-38/ckpt-60000.pt ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.2rho/2025-06-29_15-38-38/ckpt-60000/math_eval
#CUDA_VISIBLE_DEVICES=5 bash scripts/run_eval.sh ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.3rho/2025-06-29_15-39-51/ckpt-40000.pt ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.3rho/2025-06-29_15-39-51/ckpt-40000/math_eval
#CUDA_VISIBLE_DEVICES=5 bash scripts/run_eval.sh ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.4rho/2025-06-28_21-08-15/ckpt-30000.pt ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.4rho/2025-06-28_21-08-15/ckpt-30000/math_eval
#CUDA_VISIBLE_DEVICES=6 bash scripts/run_eval.sh ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.5rho/2025-06-28_20-58-46/ckpt-24000.pt ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.5rho/2025-06-28_20-58-46/ckpt-24000/math_eval
#CUDA_VISIBLE_DEVICES=0 bash scripts/run_eval.sh ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.6rho/2025-06-28_19-40-12/ckpt-20000.pt ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.6rho/2025-06-28_19-40-12/ckpt-20000/math_eval
#CUDA_VISIBLE_DEVICES=3 bash scripts/run_eval.sh ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.7rho/2025-06-28_19-48-55/ckpt-18000.pt ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.7rho/2025-06-28_19-48-55/ckpt-18000/math_eval

#CUDA_VISIBLE_DEVICES=4 bash scripts/run_eval.sh ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.8rho/2025-06-28_19-52-54/ckpt-16000.pt ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-0.8rho/2025-06-28_19-52-54/ckpt-16000/math_eval

#CUDA_VISIBLE_DEVICES=7 bash scripts/run_eval.sh ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-1.0rho/2025-06-29_11-51-42/ckpt-12000.pt ../nanoGPT/out/cont-gpt2-124M-owm-7.5B-1.0rho/2025-06-29_11-51-42/ckpt-12000/math_eval

#CUDA_VISIBLE_DEVICES=7 bash scripts/run_eval.sh ../nanoGPT/out/cont-gpt2-1.5B-owm-15B/2025-07-02_21-50-52/ckpt-26000.pt ../nanoGPT/out/cont-gpt2-1.5B-owm-15B/2025-07-02_21-50-52/ckpt-26000/math_eval



#bash scripts/run_eval.sh gpt2 ../nanoGPT/out/gpt2
#bash scripts/run_eval.sh TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T  ../nanoGPT/out/tinyallama

# ======= Base Models =======
# PROMPT_TYPE="cot" # direct / cot / pal / tool-integrated
# MODEL_NAME_OR_PATH=${HF_MODEL_DIR}/mistral/Mistral-7B-v0.1
# MODEL_NAME_OR_PATH=${HF_MODEL_DIR}/llemma/llemma_7b
# MODEL_NAME_OR_PATH=${HF_MODEL_DIR}/internlm/internlm2-math-base-7b
# MODEL_NAME_OR_PATH=${HF_MODEL_DIR}/deepseek/deepseek-math-7b-base


# ======= SFT Models =======
# PROMPT_TYPE="deepseek-math" # self-instruct / tora / wizard_zs / deepseek-math / kpmath
# MODEL_NAME_OR_PATH=${HF_MODEL_DIR}/deepseek/deepseek-math-7b-rl
# MODEL_NAME_OR_PATH=${HF_MODEL_DIR}/deepseek/deepseek-math-7b-instruct



# DATA_NAMES="gsm8k" --max_tokens_per_call 200
#DATA_NAMES="gsm8k,minerva_math,svamp,asdiv,mawps,tabmwp,mathqa,mmlu_stem,sat_math"
DATA_NAMES="mathqa"
SPLIT="test"
NUM_TEST_SAMPLE=-1


# single-gpu
TOKENIZERS_PARALLELISM=false \
/cpfs/user/fengmingquan/miniconda3/envs/nanogpt/bin/python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --data_names ${DATA_NAMES} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --save_outputs \
    --max_tokens_per_call 5 \
    --batch_size 32 \
    --overwrite
    # --use_hf \
    # --use_vllm \


# multi-gpu
#python3 scripts/run_eval_multi_gpus.py \
#     --model_name_or_path $MODEL_NAME_OR_PATH \
#     --output_dir $OUTPUT_DIR \
#     --data_names ${DATA_NAMES} \
#     --prompt_type "cot" \
#     --temperature 0 \
#     --save_outputs \
#     --split_data_over_gpus \
#     --available_gpus 0,1,2,3,4,5,6,7 \
#     --gpus_per_model 1 \
#     --max_tokens_per_call 128 \
#     --batch_size 256 \
#     --use_hf \
#     --overwrite
#     --use_vllm \

#!/bin/bash

MODELS=(
  "allenai/OLMo-7B-Instruct-hf"
  "allenai/OLMo-2-1124-7B-Instruct"
  "allenai/Olmo-3-7B-Instruct"
  "allenai/Olmo-3-1025-7B"
  "meta-llama/Llama-2-7b-chat-hf"
  "lzw1008/Emollama-chat-7b"
  "meta-llama/Meta-Llama-3-8B"
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "meta-llama/Meta-Llama-3.1-8B"
  "meta-llama/Meta-Llama-3.1-8B-Instruct"
  "bigscience/bloomz-7b1-mt"
  "lzw1008/Emobloom-7b"
  "mistralai/Mistral-7B-v0.3"
  "mistralai/Mistral-7B-Instruct-v0.3"
  "mistralai/Ministral-8B-Instruct-2410"
  "mistralai/Ministral-3-8B-Instruct-2512"
  "mistralai/Ministral-3-8B-Base-2512"
  "Qwen/Qwen2.5-7B"
  "Qwen/Qwen2.5-7B-Instruct-1M"
  "Qwen/Qwen3-4B"
  "Qwen/Qwen3-4B-Instruct-2507"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

# English
for MODEL in "${MODELS[@]}"; do
    echo "------------------------------------------------------"
    echo "Evaluating model $MODEL on English"
    echo "------------------------------------------------------"
    python ./code/gen_identification_preds.py -cf ./code/configs/full/en_ident_config.json --model "$MODEL"
done

# Spanish
for MODEL in "${MODELS[@]}"; do
    echo "------------------------------------------------------"
    echo "Evaluating model $MODEL on Spanish"
    echo "------------------------------------------------------"
    python ./code/gen_identification_preds.py -cf ./code/configs/full/es_ident_config.json --model "$MODEL"
done
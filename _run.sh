#!/bin/bash

cd /openvino.genai/llm_bench/python
source /root/miniforge3/bin/activate ov_genai
huggingface-cli login --token $HUGGINGFACE_TOKEN

download_model(){
  if [ ! -d "/models/$1" ]; then
    optimum-cli export openvino --trust-remote-code --model $1 /models/$1
  fi
}

run_benchmark(){
  python benchmark.py -m /models/$1 -p "What is the meaning of life?" -n 2 -d $2 --torch_compile_backend openvino
}

_DEVICE=CPU
if [ "$HAS_GPU" == "1" ]; then
  _DEVICE=GPU
fi

# meta-llama/Meta-Llama-3-8B
# meta-llama/Meta-Llama-3.1-8B
 
for model in \
  'TinyLlama/TinyLlama-1.1B-Chat-v1.0' \
  'meta-llama/Llama-2-7b-chat-hf' 
do
  download_model $model
  run_benchmark $model $_DEVICE
done


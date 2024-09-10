#!/bin/bash

cd /openvino.genai/llm_bench/python
source /root/miniforge3/bin/activate ov_genai
huggingface-cli login --token $HUGGINGFACE_TOKEN

download_model(){
  if [ ! -d "/models/$1/$2" ]; then
    optimum-cli export openvino --weight-format $2 --model $1 /models/$1/$2
  fi
}

run_benchmark(){
  echo 'Running benchmark ' $1 ' ' $2 ' ' $3
  python benchmark.py -m /models/$1/$2 -p "What is the meaning of life?" -n 2 -d $3 --torch_compile_backend openvino
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
  for quant in \
    'int8' \
    'int4'
  do
    download_model $model $quant
    run_benchmark $model $quant $_DEVICE
  done
done


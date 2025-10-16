#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")"

benchmark() {
  results_folder="./results"
  model="Qwen/Qwen2.5-1.5B-Instruct"
  dataset_name="random"
  # dataset_path="./sonnet_4x.txt"
  num_prompts=10
  qps=2
  input_len=2560
  output_len=16
  prefix_len=0
  tag="test"

  python3 ../../benchmarks/benchmark_serving.py \
          --backend vllm \
          --model $model \
          --dataset-name $dataset_name \
          --num-prompts $num_prompts \
          --random-input-len $input_len \
          --random-output-len $output_len \
          --random-prefix-len $prefix_len \
          --port 8000 \
          --result-dir $results_folder \
          --result-filename "$tag"-qps-"$qps".json \
          --request-rate "$qps"

  sleep 2
}   

benchmark
#!/bin/bash

# 设置环境变量
export shape="[2, 4, 8]"
export dtype="float32"
export seeds="[42]"
export shard="0"
export backend="gpu"

# 使用 paddle.distributed.launch 启动程序
python -m paddle.distributed.launch --gpus="0,1" pir_reshard_s_to_r.py

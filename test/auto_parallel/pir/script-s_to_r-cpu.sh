export shape="[2, 4, 8]"
export dtype="float32"
export seeds="[42]"
export shard="0"
export backend="cpu"
python -m paddle.distributed.launch --devices="cpu" pir_reshard_s_to_r.py

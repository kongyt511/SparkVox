#!/bin/bash

# Get the absolute path of the script's directory
script_dir=$(dirname "$(realpath "$0")")
# Get the root directory
root_dir=$(dirname $(dirname $(dirname "$script_dir")))

cd "$root_dir" || exit

# Set the batch size for processing
batch_size=32
data_name='m3ed'
jsonlfile="egs/data/metadata/m3ed.jsonl"
save_dir="local/${data_name}"

config_path="/aifs4su/xinshengwang/code/SparkAudio/SparkVox/egs/codec/bicodec/results/bicodec.24k/20250420_014312/config.yaml"
ckpt_path="/aifs4su/xinshengwang/code/VoxSphere/egs/recipes/librispeech/ssl2wav/results/20241202.ema.wav2vec.lmix.8192.spkFSQ.dualEncoder.fvq/ckpt/800000.pt"
data_root="$root_dir/egs/data/audios"

# Run the Python script with the specified arguments
python -m sparkvox.tools.tokenizer.audio_tokenizer.bicodec.extract_codes \
  --jsonlfile "$jsonlfile" \
  --data_root "$data_root" \
  --config_path "$config_path" \
  --ckpt_path "$ckpt_path" \
  --save_dir "$save_dir" \
  --batch_size "$batch_size"


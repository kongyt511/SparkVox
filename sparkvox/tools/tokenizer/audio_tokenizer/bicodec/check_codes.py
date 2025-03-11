import os
import torch
import soundfile as sf
from sparkvox.tools.tokenizer.audio_tokenizer.bicodec.bicodec_tokenizer import (
    BiCodecTokenizer,
)

device = torch.device("cuda:1")

tokenizer = BiCodecTokenizer(
    config_path="/aifs4su/xinshengwang/code/spark-tts/sparkvox/egs/codec/bicodec/config/bicodec.yaml",
    ckpt_path="/aifs4su/xinshengwang/code/VoxSphere/egs/recipes/librispeech/ssl2wav/results/20241202.ema.wav2vec.lmix.8192.spkFSQ.dualEncoder.fvq/ckpt/800000.pt",
    device=device,
)


wav_path = '/aifs4su/xinshengwang/code/Inference/Spark-TTS/local/zhisheng_prompt.wav'
basename = os.path.basename(wav_path)
global_tokens, semantic_tokens = tokenizer.tokenize(wav_path)

# tokens = torch.load(
#     "/aifs4su/xinshengwang/data/speech/mobvoi/bicodec_16k/P100_data_biaobei_10000_v100_mel_split_1_000001.pt"
# )

# import pdb; pdb.set_trace()
# global_tokens = tokens[:32].unsqueeze(0).to(device)
# semantic_tokens = tokens[32:].unsqueeze(0).to(device)


rec_wav = tokenizer.detokenize(global_tokens.squeeze(0), semantic_tokens)

sf.write(f"/aifs4su/xinshengwang/code/Inference/Spark-TTS/local/rec_rec_org.wav", rec_wav, 16000)
print(f'reconstructed wav saved to /aifs4su/xinshengwang/code/Inference/Spark-TTS/local/rec_rec_org.wav')

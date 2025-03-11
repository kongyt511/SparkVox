# Copyright (c) 2025 SparkAudio
#               2025 Xinsheng Wang (w.xinshawn@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generator module
"""

import hydra
import torch
import torch.nn as nn

from pathlib import Path
from typing import Dict, Any

from sparkvox.utils.file import load_config
from sparkvox.models.codec.BiCodec.modules.feat_encoder import Encoder
from sparkvox.models.codec.BiCodec.modules.feat_decoder import Decoder
from sparkvox.models.codec.BiCodec.modules.speaker_encoder import SpeakerEncoder
from sparkvox.models.codec.base.quantize.factorized_vector_quantize import (
    FactorizedVectorQuantize,
)


class Generator(nn.Module):
    """Generator module."""

    def __init__(
        self,
        mel_params: Dict[str, Any],
        encoder: nn.Module,
        decoder: nn.Module,
        quantizer: nn.Module,
        speaker_encoder: nn.Module,
        prenet: nn.Module,
        postnet: nn.Module,
        d_vector_train_start: int,
        **kwargs
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.quantizer = quantizer
        self.speaker_encoder = speaker_encoder
        self.prenet = prenet
        self.postnet = postnet
        self.d_vector_train_start = d_vector_train_start
        self.init_mel_transformer(mel_params)

    @classmethod
    def load_from_checkpoint(
        cls, config_path: Path, ckpt_path: Path, **kwargs
    ):
        """
        Load pre-trained model

        Args:
            config_path (Path): path to the model model configuration.
            ckpt_path (Path): path of model checkpoint.

        Returns:
            model (nn.Module): The loaded model instance.
        """
        cfg = load_config(config_path)
        if "config" in cfg.keys():
            cfg = cfg["config"]
        mel_params = cfg["model"]["generator"]["mel_params"]
        encoder = hydra.utils.instantiate(cfg["model"]["generator"]["encoder"])
        quantizer = hydra.utils.instantiate(cfg["model"]["generator"]["quantizer"])
        prenet = hydra.utils.instantiate(cfg["model"]["generator"]["prenet"])
        postnet = hydra.utils.instantiate(cfg["model"]["generator"]["postnet"])
        decoder = hydra.utils.instantiate(cfg["model"]["generator"]["decoder"])
        speaker_encoder = hydra.utils.instantiate(
            cfg["model"]["generator"]["speaker_encoder"]
        )

        model = cls(
            mel_params=mel_params,
            encoder=encoder,
            decoder=decoder,
            quantizer=quantizer,
            speaker_encoder=speaker_encoder,
            prenet=prenet,
            postnet=postnet,
            d_vector_train_start=cfg["model"]["generator"]["d_vector_train_start"],
        )

        state_dict = torch.load(ckpt_path, map_location="cpu")

        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict["generator"], strict=False
        )

        for key in missing_keys:
            print("missing tensor {}".format(key))
        for key in unexpected_keys:
            print("unexpected tensor {}".format(key))
        model.eval()
        model.remove_weight_norm()
        return model

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        feat = batch["feat"]
        mel = self.mel_transformer(batch["ref_wav"]).squeeze(1)

        z = self.encoder(feat.transpose(1, 2))
        vq_outputs = self.quantizer(z)

        x_vector, d_vector = self.speaker_encoder(mel.transpose(1, 2))

        if batch["step"] > self.d_vector_train_start:
            conditions = d_vector
            with_speaker_loss = False
        else:
            conditions = x_vector
            with_speaker_loss = True

        x = self.prenet(vq_outputs["z_q"], conditions)
        pred_feat = self.postnet(x)
        x = x + conditions.unsqueeze(-1)
        wav_recon = self.decoder(x)

        return {
            "vq_loss": vq_outputs["vq_loss"],
            "perplexity": vq_outputs["perplexity"],
            "cluster_size": vq_outputs["active_num"],
            "recons": wav_recon,
            "pred_feat": pred_feat,
            "x_vector": x_vector,
            "d_vector": d_vector,
            "audios": batch["wav"].unsqueeze(1),
            "with_speaker_loss": with_speaker_loss,
        }

    @torch.no_grad()
    def tokenize(self, batch: Dict[str, Any]):
        """tokenize the input audio"""
        feat = batch["feat"]
        mel = self.mel_transformer(batch["ref_wav"]).squeeze(1)
        
        z = self.encoder(feat.transpose(1, 2))
        semantic_tokens = self.quantizer.tokenize(z)
        global_tokens = self.speaker_encoder.tokenize(mel.transpose(1, 2))

        return semantic_tokens, global_tokens

    @torch.no_grad()
    def detokenize(self, semantic_tokens, global_tokens):
        """detokenize the input tokens"""
        z_q = self.quantizer.detokenize(semantic_tokens)
        d_vector = self.speaker_encoder.detokenize(global_tokens)
        x = self.prenet(z_q, d_vector)
        x = x + d_vector.unsqueeze(-1)
        wav_recon = self.decoder(x)

        return wav_recon

    def init_mel_transformer(self, config: Dict[str, Any]):
        import torchaudio.transforms as TT

        self.mel_transformer = TT.MelSpectrogram(
            config["sample_rate"],
            config["n_fft"],
            config["win_length"],
            config["hop_length"],
            config["mel_fmin"],
            config["mel_fmax"],
            n_mels=config["num_mels"],
            power=1,
            norm="slaney",
            mel_scale="slaney",
        )

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                # print(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)


# test
if __name__ == "__main__":
    config = load_config("egs/codec/bicodec/config/bicodec.yaml")
    model = hydra.utils.instantiate(config["model"]["generator"])

    duration = 0.96
    x = torch.randn(20, 1, int(duration * 16000))
    feat = torch.randn(20, int(duration * 50), 1024)
    inputs = {"feat": feat, "wav": x, "ref_wav": x, "step": 100000}
    outputs = model(inputs)
    semantic_tokens, global_tokens = model.tokenize(inputs)
    wav_recon = model.detokenize(semantic_tokens, global_tokens)
    if outputs["recons"].detach().all() == wav_recon.all():
        print("test successful")
    else:
        print("test failed")

    # for infer
    from sparkvox.utils.audio import load_audio

    wav = load_audio(
        "/aifs4su/mmdata/rawdata/speech/Game_Genshin/English/荒泷一斗/vo_SDEQ001_3_itto_09.wav",
        sampling_rate=16000,
        volume_normalize=True,
    )
    wav = torch.from_numpy(wav).float()
    feat = torch.load(
        "/aifs4su/xinshengwang/data/speech/vocoder/wav2vec/wav2vec_mix/0001423757.pt"
    )
    wav = wav.unsqueeze(0)
    feat = feat.unsqueeze(0).float()
    inputs = {"feat": feat, "wav": wav, "ref_wav": wav, "step": 100000}

    model = Generator.load_from_checkpoint(
        config_path="egs/codec/bicodec/config/bicodec.yaml",
        ckpt_path="/aifs4su/xinshengwang/code/VoxSphere/egs/recipes/librispeech/ssl2wav/results/20241202.ema.wav2vec.lmix.8192.spkFSQ.dualEncoder.fvq/ckpt/450000.pt",
        device="cpu",
    )

    model.eval()
    model.remove_weight_norm()

    semantic_tokens, global_tokens = model.tokenize(inputs)
    wav_recon = model.detokenize(semantic_tokens, global_tokens)
    import soundfile as sf

    sf.write("local/wav_recon.wav", wav_recon.squeeze().detach().numpy(), 16000)
    sf.write("local/wav.wav", inputs["wav"].squeeze().detach().numpy(), 16000)

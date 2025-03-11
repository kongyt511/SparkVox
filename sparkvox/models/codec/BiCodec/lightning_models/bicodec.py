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
BigCodec model with pytorch lightning
"""

import torch
import torch.nn as nn

from hydra.utils import instantiate
from typing import Dict, Any
from omegaconf import DictConfig

from sparkvox.models.codec.base.lightning_models.wav_codec import WavCodec
from sparkvox.models.codec.base.loss import MultiResolutionMelSpectrogramLoss


class BiCodec(WavCodec):
    """BiCodec model."""

    def __init__(self, config: DictConfig, **kwargs) -> None:
        super().__init__(config)
        pass

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass."""
        return self.model["generator"](batch)

    def init_model(self) -> None:
        """Initialize the model."""

        self.model = nn.ModuleDict()
        generator = instantiate(self.config.generator)
        discriminator = instantiate(self.config.discriminator)

        self.model["generator"] = generator
        self.model["discriminator"] = discriminator

    def init_loss_functions(self):
        """Initialize the loss functions."""
        self.mel_loss = MultiResolutionMelSpectrogramLoss(**self.config.mel_loss_params)
        self.mse_loss = nn.MSELoss()

    def compute_generator_loss(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Compute the generator loss."""
        loss_dict = {
            "vq_loss": inputs["vq_loss"],
            "perplexity": inputs["perplexity"],
            "cluster_size": inputs["cluster_size"],
        }
        
        if inputs["with_speaker_loss"]:
            loss_dict["speaker_loss"] = self.mse_loss(inputs['x_vector'].detach(), inputs['d_vector'])
        
        mel_loss = self.mel_loss(
            inputs["recons"].squeeze(1), inputs["audios"].squeeze(1)
        )

        adv_loss_dict = self.model["discriminator"].adversarial_loss(
            inputs
        )
        loss_dict["mel_loss"] = mel_loss
        loss_dict.update(adv_loss_dict)

        loss = sum(
            [
                v * loss_dict[k]
                for k, v in self.config["loss_lambdas"].items()
                if k in loss_dict
            ]
        )
        loss_dict["gen_loss"] = loss
        return loss_dict
    
    def update_batch_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        batch["step"] = self.global_step
        return batch

# test
if __name__ == "__main__":
    from sparkvox.utils.file import load_config

    config = load_config("egs/codec/bicodec/config/bicodec.yaml")
    model_config = config["model"]
    model = BiCodec(model_config)
    # model = instantiate(model_config, model_config)
    batch = {
        "step": 1,
        "wav": torch.randn(4, 1, int(16000 * 2.4)),
        "feat": torch.randn(4, 1024, int(2.4 * 50)),
    }

    output = model(batch)

    gen_loss = model.compute_generator_loss(output)
    disc_loss = model.model["discriminator"].discriminative_loss(output)
    
    print(gen_loss)
    print(disc_loss)
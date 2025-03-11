import os
import soundfile as sf
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from sparkvox.utils.file import load_config
from sparkvox.tools.tokenizer.audio_tokenizer.bicodec.bicodec_tokenizer import (
    BiCodecTokenizer as audio_tokenizer,
)
from sparkvox.tools.tokenizer.audio_tokenizer.bicodec.dataloader import (
    WavDataset as TokenizerDataset,
)


def batch_to_cuda(batch: dict, rank):
    for k in batch:
        if type(batch[k]) is torch.Tensor:
            batch[k] = batch[k].cuda(rank)
    return batch


def setup_process(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    dist.destroy_process_group()


def extract(rank, args):
    setup_process(rank, args.world_size)
    device = torch.device(f"cuda:{rank}")
    config = load_config(args.config_path)
    model = audio_tokenizer(args.config_path, args.ckpt_path, device)
    model = DDP(model, device_ids=[rank]) 
    config["datasets"]["jsonlfiles_for_extract"] = args.jsonlfile
    dataset = TokenizerDataset(config["datasets"], mode="val", extract_feat=True)
    sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=rank)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=8, collate_fn=dataset.collate_fn, sampler=sampler
    ) 

    save_dir = args.save_dir

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    if rank == 0:
        iterator = tqdm(dataloader, 
                    total=len(dataloader),
                    desc=f'Processing',
                    disable=rank != 0)
    else:
        iterator = dataloader

    for batch in iterator:
        indexs, lengths = batch["index"], batch["length"]
        batch = batch_to_cuda(batch, rank)

        with torch.no_grad():
            global_tokens, semantic_tokens = model.module.tokenize_batch(batch)
        for i in range(len(indexs)):
            indx = indexs[i]
            length = lengths[i]
            output_length = int(length / 320)
            global_token = global_tokens[i].detach().cpu().squeeze()
            semantic_token = semantic_tokens[i].detach().cpu().squeeze()[:output_length]
            assert len(global_token) == 32
            token = torch.cat([global_token, semantic_token], dim=0)
            torch.save(token, os.path.join(save_dir, f"{indx}.pt"))

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonlfile",
        type=str,
        default="/aifs4su/xinshengwang/data/speech/17_Librispeech_SLR12/LibriSpeech/test.jsonl",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="/aifs4su/xinshengwang/code/spark-tts/sparkvox/egs/codec/bicodec/config/bicodec.yaml",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/aifs4su/xinshengwang/code/VoxSphere/egs/recipes/librispeech/ssl2wav/results/20241202.ema.wav2vec.lmix.8192.spkFSQ.dualEncoder.fvq/ckpt/800000.pt",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/aifs4su/xinshengwang/data/spark-tts/bicodec/LibriSpeech",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--world_size", type=int, default=torch.cuda.device_count())

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    mp.spawn(extract, args=(args,), nprocs=args.world_size)
    # extract(0, args)


if __name__ == "__main__":
    main()

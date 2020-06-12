import sys
from typing import Tuple

import click
import numpy as np
import torch
import torchvision.transforms as T
from torch import nn
from torch.utils import data

from src import configs
from src import data as lc_data
from src import network
from src.l3c import timer

import pickle
import copy


def run_eval(
        loader: data.DataLoader, compressor: nn.Module,
) -> Tuple[network.Bits, int]:
    """ Runs entire eval epoch. """
    time_accumulator = timer.TimeAccumulator()
    compressor.eval()
    cur_agg_size = 0

    #  Get the individual images' bpsp
    individual_bpsps = [{} for _ in range(500)]  # list of dictionary

    with torch.no_grad():
        # BitsKeeper is used to aggregates bits from all eval iterations.
        bits_keeper = network.Bits()
        for i, (_, x) in enumerate(loader):
            cur_agg_size += np.prod(x.size())
            with time_accumulator.execute():
                x = x.cuda()
                bits = compressor(x)
            bits_keeper.add_bits(bits)

            bpsp = bits_keeper.get_total_bpsp(cur_agg_size)

            # Get the individual images' bpsp

            individual_bpsps[i]["rounding"] = copy.deepcopy(bits.get_bits("eval/0_rounding")) \
                                          + copy.deepcopy(bits.get_bits("eval/1_rounding")) \
                                          + copy.deepcopy(bits.get_bits("eval/2_rounding"))
            individual_bpsps[i]["image_3"] = copy.deepcopy(bits.get_bits("eval/codes_0"))
            individual_bpsps[i]["image_2"] = copy.deepcopy(bits.get_bits("eval/0_0")) + \
                                         copy.deepcopy(bits.get_bits("eval/0_1")) + \
                                         copy.deepcopy(bits.get_bits("eval/0_2"))
            individual_bpsps[i]["image_1"] = copy.deepcopy(bits.key_to_bits["eval/1_0"].item()) + \
                                         copy.deepcopy(bits.key_to_bits["eval/1_1"].item()) + \
                                         copy.deepcopy(bits.key_to_bits["eval/1_2"].item())
            individual_bpsps[i]["image_0"] = copy.deepcopy(bits.key_to_bits["eval/2_0"].item()) + \
                                         copy.deepcopy(bits.key_to_bits["eval/2_1"].item()) + \
                                         copy.deepcopy(bits.key_to_bits["eval/2_2"].item())
            print(
                f"Bpsp: {bpsp.item():.3f}; Number of Images: {i + 1}; "
                f"Batch Time: {time_accumulator.mean_time_spent()}",
                end="\r")

        print()
    return bits_keeper, cur_agg_size, individual_bpsps


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@click.command()
@click.option("--path", type=click.Path(exists=True),
              help="path to directory of eval images.")
@click.option("--file", type=click.File("r"),
              help="file for eval image names.")
@click.option("--workers", type=int, default=2,
              help="Number of worker threads to use in dataloader.")
@click.option("--resblocks", type=int, default=3, show_default=True,
              help="Number of resblocks to use.")
@click.option("--n-feats", type=int, default=64, show_default=True,
              help="Size of n_feats vector used.")
@click.option("--scale", type=int, default=3, show_default=True,
              help="Scale of downsampling")
@click.option("--load", type=click.Path(exists=True), default="/dev/null",
              help="Path to load model")
@click.option("--K", type=int, default=10,
              help="Number of clusters in logistic mixture model.")
@click.option("--crop", type=int, default=0,
              help="Size of image crops in training. 0 means no crop.")
def main(
        path: str, file, workers: int, resblocks: int, n_feats: int,
        scale: int, load: str, k: int, crop: int,
) -> None:
    configs.n_feats = n_feats
    configs.resblocks = resblocks
    configs.K = k
    configs.scale = scale

    print(sys.argv)
    print([item for item in dir(configs) if not item.startswith("__")])

    if load != "/dev/null":
        checkpoint = torch.load(load)
        print(f"Loaded model from {load}.")
        print("Epoch:", checkpoint["epoch"])
    else:
        checkpoint = {}

    compressor = network.Compressor()
    if checkpoint:
        compressor.nets.load_state_dict(checkpoint["nets"])
    compressor = compressor.cuda()

    print(f"Number of parameters: {count_params(compressor.nets)}")
    print(compressor.nets)

    transforms = []  # type: ignore
    if crop > 0:
        transforms.insert(0, T.CenterCrop(crop))

    dataset = lc_data.ImageFolder(
        path, [filename.strip() for filename in file],
        scale, T.Compose(transforms)
    )
    loader = data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=workers, drop_last=False,
    )
    print(f"Loaded dataset with {len(dataset)} images")

    bits, inp_size, individual_bpsps = run_eval(loader, compressor)
    for key in bits.get_keys():
        print(f"{key}:", bits.get_scaled_bpsp(key, inp_size))

    print("Average total bpsp is: ", bits.get_total_bpsp(inp_size))


    # Get the individual images' bpsp
    # save the individual_bpsps
    with open('test.p', 'wb') as f:
        pickle.dump(individual_bpsps,f)

if __name__ == "__main__":
    main()

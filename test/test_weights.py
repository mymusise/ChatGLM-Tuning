import torch
import os


if __name__ == '__main__':
    path1 = '../data-renmin/output/checkpoint-500/adapter_model.bin'
    path2 = '../data-renmin/output/checkpoint-2000/adapter_model.bin'

    weights1 = torch.load(path1, map_location="cpu")
    weights2 = torch.load(path2, map_location="cpu")

    assert weights1.keys() == weights2.keys()


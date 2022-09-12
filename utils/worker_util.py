import torch
import random
import numpy


def worker_init_fn(worker_id):
    pytorch_seed = torch.utils.data.get_worker_info().seed
    seed = pytorch_seed % (2 ** 32 - 1)
    random.seed(seed)
    numpy.random.seed(seed)
    print("worker:%d,pytorch_seed:%d" % (worker_id, seed))

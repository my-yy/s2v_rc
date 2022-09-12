import random
import numpy
import torch


def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # https://pytorch.org/docs/stable/notes/randomness.html

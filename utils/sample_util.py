import random
import numpy as np


def random_element(array, need_index=False):
    length = len(array)
    assert length > 0, length
    rand_index = random.randint(0, length - 1)
    if need_index:
        return array[rand_index], rand_index
    else:
        return array[rand_index]


def random_elements(array, number):
    return np.random.choice(array, number, replace=False)


def random_elements_sequential(array, number):
    if len(array) == number:
        return array

    assert len(array) > number
    length = len(array) - number + 1
    rand_index = random.randint(0, length - 1)
    sub_array = array[rand_index:rand_index + number]
    assert len(sub_array) == number
    return sub_array

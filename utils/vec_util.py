# 11月28
import numpy as np


def get_vec_length(vec):
    if type(vec) == list:
        vec = np.array(vec)
    return np.sqrt(np.sum(vec * vec))


# 会更改dict本身的内容！
def dict2unit_dict(the_dict):
    for key in the_dict:
        vec = the_dict[key]
        the_len = get_vec_length(vec)
        the_dict[key] = vec / the_len
    return the_dict


def assert_is_unit_tensor(tensor):
    npy = tensor.detach().cpu().numpy()
    length = get_vec_length(npy)
    assert np.isclose(length, 1.0)


def assert_dict_unit_vector(the_dic):
    for key in the_dic:
        v = the_dic[key]
        the_len = get_vec_length(v)
        assert np.isclose(the_len, 1.0)
        break


def get_vec_dim_in_dict(the_dic):
    for key in the_dic:
        v = the_dic[key]
        return len(v)


def to_unit_vector(vector):
    return vector / get_vec_length(vector)


def norm_batch_vector(matix):
    # matix = np.array([
    #     [3, 4],
    #     [1, 1]
    # ])
    vec_length = np.linalg.norm(matix, axis=1, keepdims=True)
    out = matix / vec_length
    #  [[0.6  , 0.8 ],
    #   [0.707, 0.707]]
    return out

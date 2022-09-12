import _pickle as pickle  # python3
import time


def read_pickle(filepath):
    start_time = time.time()
    f = open(filepath, 'rb')
    word2mfccs = pickle.load(f)
    f.close()
    # time_cost = time.time() - start_time
    # if time_cost > 10:
    #     print("read_pickle耗时:", time_cost)
    return word2mfccs


def save_pickle(save_path, save_data):
    f = open(save_path, 'wb')
    pickle.dump(save_data, f)
    f.close()


import json


def read_json(filepath):
    with open(filepath) as f:
        obj = json.load(f)
    return obj


def save_json(save_path, obj):
    with open(save_path, 'w') as f:
        json.dump(obj, f)

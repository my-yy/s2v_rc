import torch
from utils import pickle_util
from utils.path_util import look_up
import numpy as np
from torch.utils.data import DataLoader
import collections
from utils import vec_util
import time


class Dataset(torch.utils.data.Dataset):
    def __init__(self, word_list, mfcc_dict):
        word2keys = pickle_util.read_pickle(look_up("./dataset/info/500h_word2wav_keys.pkl"))
        data = []
        for word in word_list:
            array = word2keys[word]
            for key in array:
                data.append([word, key])
        self.data = data
        self.mfcc_dict = mfcc_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        word, key = self.data[index]
        mfcc, the_len = self.load_mfcc(key)
        return word, torch.FloatTensor(mfcc), the_len

    def load_mfcc(self, wav_name, aim_len=20):
        key = wav_name.replace(".wav", ".mfcc")
        data = self.mfcc_dict[key]
        cur_len = len(data)
        if cur_len < aim_len:
            data = np.concatenate([data, np.zeros([aim_len - cur_len, 13])])
        return data, cur_len


def get_word2arrays(model, word_list, mfcc_dict, worker, batch_size):
    model.eval()
    loader = DataLoader(Dataset(word_list, mfcc_dict), batch_size=batch_size, shuffle=False, num_workers=worker, pin_memory=True, drop_last=False)
    word2arrays = collections.defaultdict(list)

    start_time = time.time()
    step = 0
    with torch.no_grad():
        for data in loader:
            words, input_value, input_length = data
            _, (hidden_state, _) = model.encoder(input_value.cuda(), input_length)
            hn_2d_numpy = hidden_state[0].cpu().detach().numpy()

            # to unit vector:
            hn_2d_numpy = vec_util.norm_batch_vector(hn_2d_numpy)
            for word, vec in zip(words, hn_2d_numpy):
                word2arrays[word].append(vec)
            step += 1
            if step % 100 == 0:
                progress = step / len(loader)
                time_cost = (time.time() - start_time) / 3600
                total_time = time_cost / progress
                print("progress:%.4f,total_time:%.2fh" % (progress, total_time))

    # mean_embedding:
    final_word_dict = {}
    for word, array in word2arrays.items():
        mean_emb = np.mean(array, axis=0)
        mean_emb = vec_util.to_unit_vector(mean_emb)
        final_word_dict[word] = mean_emb
    return word2arrays, final_word_dict


if __name__ == "__main__":
    pass

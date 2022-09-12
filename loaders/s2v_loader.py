import torch
from utils import pickle_util
import numpy as np
import collections
from torch.utils.data import DataLoader

mfcc_dict = pickle_util.read_pickle("./dataset/split_mfcc_dict.pkl")


class Dataset(torch.utils.data.Dataset):
    def __init__(self, window_size, word_min_count=4, subsampling_rate=0.001):
        self.word2keys = pickle_util.read_pickle("./dataset/info/500h_word2wav_keys.pkl")

        # 1.Read alignment files and count words
        word_counter = collections.Counter()
        sentence_objs = pickle_util.read_pickle("./dataset/info/500h_word_split.pkl")
        for obj in sentence_objs:
            for w in obj["word_array"]:
                word_counter[w] += 1

        # 2.Filter out infrequent words:
        vocab_set = set([key for key, v in word_counter.items() if v >= word_min_count])
        print("Vocab size:", len(vocab_set))

        # 3.Create a long word list:
        word_arrays = []
        wav_arrays = []
        sentence_objs.sort(key=lambda x: x["key"], reverse=False)
        for obj in sentence_objs:
            key = obj["key"]
            word_array = obj["word_array"]
            for i in range(len(word_array)):
                word = word_array[i]
                if word not in vocab_set:
                    continue
                wav_name = "%s/%02d_%s.wav" % (key, i + 1, word)
                word_arrays.append(word)
                wav_arrays.append(wav_name)

        self.wav_arrays = wav_arrays
        self.word_arrays = word_arrays
        self.vocab_set = vocab_set

        # 4.Frequent words subsampling:
        word_keep_prob = calc_word_keep_prob(word_arrays, word_counter, subsampling_rate)
        self.all_pairs = create_pairs(window_size, word_arrays, subsampling_rate, word_keep_prob)

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, index):
        idx_in, idx_out = self.all_pairs[index]
        word_in, word_out = self.word_arrays[idx_in], self.word_arrays[idx_out]
        wav_in, wav_out = self.wav_arrays[idx_in], self.wav_arrays[idx_out]

        mfcc_in, len_in = self.load_mfcc(wav_in)
        mfcc_out, len_out = self.load_mfcc(wav_out)

        return torch.FloatTensor(mfcc_in), len_in, torch.FloatTensor(mfcc_out), len_out, word_in, word_out

    def load_mfcc(self, wav_name, aim_len=20):
        key = wav_name.replace(".wav", ".mfcc")
        data = mfcc_dict[key]
        cur_len = len(data)
        if cur_len < aim_len:
            data = np.concatenate([data, np.zeros([aim_len - cur_len, 13])])
        return data, cur_len


def create_pairs(window_size, text, subsampling_rate, vocab2proba_keep):
    index_pairs = []
    for i in range(window_size, len(text) - window_size, 1):
        for j in range(i - window_size, i + window_size + 1, 1):
            if i != j:
                # Subsampling mechanism
                if not subsampling_rate or (subsampling_rate and np.random.random() < vocab2proba_keep[text[i]]):
                    # if not remove_people_names or (remove_people_names and tags[i][1] != 'PERSON'):
                    index_pairs.append([i, j])
    return index_pairs


def calc_word_keep_prob(text, word_counter, subsampling_rate):
    vocab2proba_keep = {}
    for word in word_counter:
        vocab2proba_keep[word] = word_counter[word] / len(text)

    for word in vocab2proba_keep:
        vocab2proba_keep[word] = (np.sqrt(vocab2proba_keep[word] / subsampling_rate) + 1) * subsampling_rate / vocab2proba_keep[word]

    return vocab2proba_keep


def get_loader(window_size, key2mfcc, word_min_count, batch_size):
    loader = DataLoader(Dataset(window_size, key2mfcc, word_min_count), batch_size=batch_size, shuffle=False)
    return loader

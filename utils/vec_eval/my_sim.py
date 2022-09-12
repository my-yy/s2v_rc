import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
from ranking import *

file2showname = {'EN-MC-30.txt': 'MC-30',
                 'EN-MEN-TR-3k.txt': 'MEN',
                 'EN-MTurk-287.txt': 'MTurk-287',
                 'EN-MTurk-771.txt': 'MTurk-771',
                 'EN-RG-65.txt': 'RG-65',
                 'EN-RW-STANFORD.txt': 'Rare-Word',
                 'EN-SIMLEX-999.txt': 'SimLex-999',
                 'EN-SimVerb-3500.txt': 'SimVerb-3500',
                 'EN-VERB-143.txt': 'Verb-143',
                 'EN-WS-353-ALL.txt': 'WS-353',
                 'EN-WS-353-REL.txt': 'WS-353-REL',
                 'EN-WS-353-SIM.txt': 'WS-353-SIM',
                 'EN-YP-130.txt': 'YP-130'}


def get_result_v2(word2emb, use_interest_set=True):
    interest_set = set(["RG-65", "WS-353-SIM", "MC-30", "MEN"])
    word_sim_dir = os.path.join("./dataset/info/eval/files")
    result_dict = {}
    for i, filename in enumerate(os.listdir(word_sim_dir)):
        manual_dict, auto_dict = ({}, {})
        not_found, total_size = (0, 0)
        for line in open(os.path.join(word_sim_dir, filename), 'r'):
            line = line.strip().lower()
            word1, word2, val = line.split()
            if word1 in word2emb and word2 in word2emb:
                manual_dict[(word1, word2)] = float(val)
                auto_dict[(word1, word2)] = cosine_sim(word2emb[word1], word2emb[word2])
            else:
                not_found += 1
            total_size += 1
        score = spearmans_rho(assign_ranks(manual_dict), assign_ranks(auto_dict))
        key = file2showname[filename]
        if use_interest_set and key not in interest_set:
            continue
        result_dict[key] = score
    return result_dict


if __name__ == "__main__":
    pass

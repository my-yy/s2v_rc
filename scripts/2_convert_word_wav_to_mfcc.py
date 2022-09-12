from concurrent.futures import wait, ProcessPoolExecutor
import os
import pathlib
import librosa
import time
import glob
import random


def read_mfcc(flac_path, mfcc_dim):
    sample_rate = 16000
    float32_array, _ = librosa.load(flac_path, sr=sample_rate)
    mfcc = librosa.feature.mfcc(y=float32_array, sr=sample_rate, n_mfcc=mfcc_dim, n_fft=512)
    return mfcc.transpose()


def mk_parent_dir_if_necessary(img_save_path):
    folder = pathlib.Path(img_save_path).parent
    if not os.path.exists(folder):
        os.makedirs(folder)


def mk_dir_if_necessary(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def handle_single_obj(wav_path):
    mfcc = read_mfcc(wav_path, 13)
    end_part = wav_path.split("split_wav/")[-1].replace(".wav", ".mfcc")
    new_path = os.path.join(save_root, end_part)
    mk_parent_dir_if_necessary(new_path)
    mfcc.dump(new_path)


def fun(task_index):
    wav_path = voice_list[task_index]
    try:
        handle_single_obj(wav_path)
    except Exception as e:
        print(e)
        return

    if task_index > 0 and task_index % 100 == 0:
        time_cost_h = (time.time() - start_time) / 3600
        p = task_index / len(voice_list)
        total_time = time_cost_h / p
        print("%d progress:%.4f total_time:%.1f" % (task_index, p, total_time))


if __name__ == '__main__':
    # This script may take 3~4 hours

    # 1. find all .wav files:
    pattern = "./dataset/split_wav/*/*.wav"
    voice_list = glob.glob(pattern)
    random.shuffle(voice_list)
    print("count:%d" % (len(voice_list) / 10000))

    # 2.define output file folder
    save_root = "/ssd/1_libri_flac/splitted_mfcc/"

    # 3.start!
    start_time = time.time()
    pool_size = 8
    pool = ProcessPoolExecutor(pool_size)
    tasks = [pool.submit(fun, i) for i in range(len(voice_list))]
    wait(tasks)

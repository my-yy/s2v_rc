from concurrent.futures import wait, ProcessPoolExecutor, ThreadPoolExecutor

from utils import pickle_util
import os
import librosa
import soundfile


def mk_dir_if_necessary(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def handle_single_obj(obj):
    key = obj["key"]
    tmp = key.split("-")
    flac_path = "./dataset/flac/%s/%s/%s.flac" % (tmp[0], tmp[1], key)
    float32_array, sample_rate = librosa.load(flac_path, sr=None)
    save_folder_path = os.path.join(save_root, key)
    mk_dir_if_necessary(save_folder_path)
    counter = 0
    for w, tup in zip(obj["word_array"], obj["time_array"]):
        f, t = tup
        f = max(0, int(f * sample_rate))
        t = min(len(float32_array), int(t * sample_rate))
        sub_arr = float32_array[f:t]
        counter += 1
        save_name = "%02d_%s.wav" % (counter, w)
        save_full_path = os.path.join(save_folder_path, save_name)
        soundfile.write(save_full_path, sub_arr, sample_rate)


def fun(task_index):
    obj = all_objs[task_index]
    try:
        handle_single_obj(obj)
    except Exception as e:
        print(e)
        return

    if task_index % 100 == 0:
        print(task_index, task_index / len(all_objs))


if __name__ == '__main__':
    all_objs = pickle_util.read_pickle("./dataset/info/500h_word_split.pkl")
    save_root = "/ssd/1_libri_flac/splitted_results/"

    # start
    pool_size = 4
    pool = ProcessPoolExecutor(pool_size)
    tasks = [pool.submit(fun, i) for i in range(len(all_objs))]
    wait(tasks)

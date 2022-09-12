import os
import collections


def get_all_transcript_file_paths(dataset_path):
    result = []
    for dir_name_1th in os.listdir(dataset_path):
        for dir_name_2th in os.listdir(os.path.join(dataset_path, dir_name_1th)):
            transcript_text_file_name = dir_name_1th + "-" + dir_name_2th + ".trans.txt"
            transcript_text_file_path = os.path.join(dataset_path, dir_name_1th, dir_name_2th, transcript_text_file_name)
            assert os.path.exists(transcript_text_file_path)
            result.append(transcript_text_file_path)

            # check:
            # do not have child folder and the only txt file is  xxx-xxx-trans.txt :
            for name in os.listdir(os.path.join(dataset_path, dir_name_1th, dir_name_2th)):
                full_path = os.path.join(dataset_path, dir_name_1th, dir_name_2th, name)
                assert not os.path.isdir(full_path)
                if "txt" in name:
                    assert full_path == transcript_text_file_path
    return result


def get_big_string_of_dataset(dataset_path):
    arr = get_all_transcript_file_paths(dataset_path)
    string_in_file = []
    for filepath in arr:
        with open(filepath) as f:
            text = f.read().lower()
            string_in_file.append(text)

    big_string = "\n".join(string_in_file)
    return big_string


def get_lines_in_transcript_file(filepath):
    answer = []
    with open(filepath) as f:
        for line in f:
            line = line.lower().strip()
            # 89-219-0000 INSTANTANEOUSLY WITH THE CONSCIOUSNESS OF....
            if not line:
                continue

            tmp = line.split()
            key = tmp[0]
            words = tmp[1:]
            answer.append(words)
    return answer


def get_all_lines_of_dataset(dataset_pat):
    arr = get_all_transcript_file_paths(dataset_pat)
    lines = []
    for filepath in arr:
        lines.extend(get_lines_in_transcript_file(filepath))
    return lines


def create_word_counter(lines):
    counter = collections.Counter()
    for line in lines:
        for word in line:
            counter[word] += 1
    return counter

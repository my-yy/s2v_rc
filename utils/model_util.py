import torch
import os
import json
import sys
from utils import pickle_util
import numpy as np

history_array = []


def save_model(epoch, model, optimizer, file_save_path):
    dirpath = os.path.abspath(os.path.join(file_save_path, os.pardir))
    if not os.path.exists(dirpath):
        print("mkdir:", dirpath)
        os.makedirs(dirpath)

    opti = None
    if optimizer is not None:
        opti = optimizer.state_dict()

    torch.save(obj={
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': opti,
    }, f=file_save_path)

    history_array.append(file_save_path)


def delete_last_saved_model():
    if len(history_array) == 0:
        return
    last_path = history_array.pop()
    if os.path.exists(last_path):
        os.remove(last_path)
        print("delete model:", last_path)

    if os.path.exists(last_path + ".json"):
        os.remove(last_path + ".json")


def load_model(resume_path, model, optimizer=None, strict=True):
    checkpoint = torch.load(resume_path)
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model'], strict=strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print("checkpoint loaded!")
    return start_epoch


def save_model_v2(model, args, model_save_name):
    model_save_path = os.path.join(args.model_save_folder, args.project, args.name, model_save_name)
    save_model(0, model, None, model_save_path)
    print("save:", model_save_path)


def save_project_info(args):
    run_info = {
        "cmd_str": ' '.join(sys.argv[1:]),
        "args": vars(args),
    }

    name = "run_info.json"
    folder = os.path.join(args.model_save_folder, args.project, args.name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    json_file_path = os.path.join(folder, name)
    with open(json_file_path, "w") as f:
        json.dump(run_info, f)

    print("save_project_info:", json_file_path)


def get_pkl_json(folder):
    names = [i for i in os.listdir(folder) if ".pkl.json" in i]
    assert len(names) == 1
    json_path = os.path.join(folder, names[0])
    obj = pickle_util.read_json(json_path)
    return obj

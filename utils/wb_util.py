# used for late-initialize wandb in case of crash before a full evaluation (which add an idle log on the monitoring panel)
import wandb
import os
from utils import pickle_util

history_logs = []
history_configs = []
is_inited = False


def update_config(obj):
    history_configs.append(obj)
    if not is_inited:
        print("wb_util temporarily cache config")
        return
    wandb.config.update(obj)


def log(obj):
    history_logs.append(obj)
    if not is_inited:
        print("wb_util temporarily cache log")
        return
    wandb.log(obj)


def init(args):
    init_core(args.project, args.name, args.dryrun)


def init_core(project, name, dryrun):
    global is_inited
    if is_inited:
        return
    is_inited = True

    if dryrun:
        os.environ['WANDB_MODE'] = 'dryrun'
        wandb.log = do_nothing
        wandb.save = do_nothing
        wandb.watch = do_nothing
        wandb.config = {}
        print("wb dryrun mode")
        return

    config_path = ".wb_config.json"
    assert os.path.exists(config_path), "do not have wandb config file"

    # assert have WB_KEY
    json_dict = pickle_util.read_json(config_path)
    assert "WB_KEY" in json_dict, "wb_config.json do not have WB_KEY"
    WB_KEY = json_dict["WB_KEY"]

    # use self-hosted wb server
    if "WANDB_BASE_URL" in json_dict:
        os.environ["WANDB_BASE_URL"] = json_dict["WANDB_BASE_URL"]

    # login
    wandb.login(key=WB_KEY)
    wandb.init(project=project, name=name)
    print("wandb inited")

    # supplement config and logs
    for obj in history_configs:
        wandb.config.update(obj)

    for log in history_logs:
        wandb.log(log)


def do_nothing(v):
    pass

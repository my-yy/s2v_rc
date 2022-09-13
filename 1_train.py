import time
from loaders import s2v_loader
from loaders.s2v_create_emb_dict_loader import get_word2arrays
from models import s2v_official
from utils import myparser, seed_util, wb_util, pickle_util
from torch.utils.data import DataLoader
import os
import numpy as np
from utils.vec_eval import my_sim


def train_step(epoch, step, data):
    input_padded, input_lengths, target_padded, target_lengths, word_in_list, word_out_list = data
    _, loss_item = model.train_step(input_padded.cuda(), input_lengths, target_padded.cuda(), target_lengths)
    return loss_item


def do_test(key2emb):
    # similarity benchmark test
    res = my_sim.get_result_v2(key2emb, use_interest_set=False)
    res2 = {}
    for k, v in res.items():
        res2["test/" + k] = v
    return res2


# save embedding file after each epoch
def after_one_epoch(epoch, train_epoch_loss_arr):
    # Regenerate Embedding:
    _, final_word_dict = get_word2arrays(model,
                                         list(train_iter.dataset.vocab_set),
                                         s2v_loader.mfcc_dict,
                                         worker=args.worker,
                                         batch_size=args.batch_size)
    # Benchmark Eval
    test_result = do_test(final_word_dict)
    wb_util.log(test_result)
    mean_loss = np.mean(train_epoch_loss_arr)
    add_obj = {
        "train/epoch": epoch,
        "train/epoch_loss": mean_loss
    }
    # Save Model
    model_name = "epoch%03d_ws%.2f_men%.2f_loss%.6f.pkl" % (epoch, test_result['test/WS-353-SIM'], test_result["test/MEN"], float(mean_loss))
    model_save_path = os.path.join(args.model_save_folder, args.project, args.name, model_name)
    model.save_model(epoch, model_save_path)
    print("save model:", model_save_path)
    pickle_util.save_json(model_save_path + ".json", test_result)

    # Save Embedding
    pickle_util.save_pickle(model_save_path + ".emb", final_word_dict)
    eval_log = {**test_result, **add_obj}
    wb_util.log(eval_log)
    print(eval_log)


def train():
    start_time = time.time()
    global step
    for epoch in range(start_epoch, args.epoch):
        model.train()
        train_epoch_loss_arr = []
        for batch in train_iter:
            step += 1
            loss_item = train_step(epoch, step, batch)
            train_epoch_loss_arr.append(loss_item)
            if step % args.log_step == 0:
                time_cost_d = (time.time() - start_time) / 3600 / 24
                progress = (step - start_step) / ((args.epoch - start_epoch) * len(train_iter))
                total_time = time_cost_d / progress
                obj = {
                    "train/batch": step,
                    "train/loss": loss_item,
                    "train/progress": progress,
                    "train/total_time": total_time,
                }
                wb_util.log(obj)
                wb_util.init(args)
                print(obj)
        after_one_epoch(epoch, train_epoch_loss_arr)


if __name__ == "__main__":
    parser = myparser.MyParser(epoch=500, seed=2526, batch_size=4096, lr=1e-3, model_save_folder="./outputs/", worker=4)
    parser.custom({
        "emb_dim": 50,
        "mfcc_dim": 13,
        "window_size": 3,
        "word_min_count": 4,
        "optimizer": "sgd",
        "log_step": 100,
        "skip_init_eval": False,
    })
    parser.use_wb(project="Speech2Vec", name="run1", dryrun=True)
    args = parser.parse()
    seed_util.set_seed(args.seed)

    # 1.data
    train_iter = DataLoader(s2v_loader.Dataset(args.window_size, args.word_min_count), batch_size=args.batch_size, shuffle=False, num_workers=args.worker, pin_memory=True)

    # 2.model
    model = s2v_official.ModelWrapper(args.mfcc_dim, args.emb_dim, args.lr, args.optimizer)
    start_step = 0
    start_epoch = 0
    step = 0
    if not args.skip_init_eval:
        # Eval the random init model:
        after_one_epoch(-1, [-1])

    train()

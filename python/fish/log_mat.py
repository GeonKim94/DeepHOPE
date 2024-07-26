import os
import json
import matplotlib.pyplot as plt
from collections import OrderedDict
from cycler import cycler
import numpy as np
from scipy import io

LOG_KEYS = {
    "train": "epoch",
    "valid": "epoch",
    "test": "fname"
}

LOG_VALUES = {
    "train": ["loss"],
    "valid": ["acc", "test_acc"],
    "test": ["train_acc", "valid_acc", "test_acc", "domain_acc", "test_src_acc", "test_tgt_acc", "time"]
}


class Logger:

    def __init__(self, save_dir):
        self.log_file = save_dir + "/log.txt"
        self.buffers = []

    def log_parse(self, log_key):
        log_dict = OrderedDict()
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if len(line) == 1 or not line.startswith("[%s]" % (log_key)):
                    continue
                # line : ~~
                line = line[line.find("] ") + 2:]  # ~~
                line_log = json.loads(line)

                train_log_key = line_log[LOG_KEYS[log_key]]
                line_log.pop(LOG_KEYS[log_key], None)
                log_dict[train_log_key] = line_log

        return log_dict

    def plot_mat(self):
        log_key = 'train'
        log_dict = self.log_parse(log_key)
        x = log_dict.keys()
        x2 = []
        for i in range(len(x)):
            x2.append(i)
        x2 = np.array(x2)
        for keys in LOG_VALUES[log_key]:
            y = [v[keys] for v in log_dict.values()]
            loss = np.array(y)

        acc = []
        log_key = 'valid'
        log_dict = self.log_parse(log_key)
        for keys in LOG_VALUES[log_key]:
            y = [v[keys] for v in log_dict.values()]
            acc_temp = np.array(y)
            acc.append(acc_temp)
        acc = np.array(acc)
        data1 = {'x': x2, 'loss': loss, 'acc1' : acc[0], 'acc2' : acc[1]}
        io.savemat('data.mat',data1)


if __name__ == "__main__":
    logger = Logger("outs")
    logger.plot_mat()


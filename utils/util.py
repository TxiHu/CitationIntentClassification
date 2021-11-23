# -*- coding: utf-8 -*-
import torch
import pandas as pd
import numpy as np
import time
import logging
import random
import os


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_submission(pre_list, filename, test_f1):
    test_unique = pd.read_csv('/content/citation_classification/dataset/SDP_test.csv')
    submission = pd.DataFrame(columns=['unique_id', 'citation_class_label'])
    pre_label = pd.Series(pre_list)
    submission['unique_id'] = test_unique['unique_id']
    submission['citation_class_label'] = pre_label
    submission.to_csv('/content/citation_classification/{}_F1_{:.4f}_Time_{}.csv'.format(filename, test_f1,
                                                                                        time.strftime('%m_%d_%H',
                                                                                              time.localtime(time.time()))),
                      sep=',', index=False, encoding='utf-8')


def log_result(test_f1, best_model_f1=0, c_matrix=None, per_eval=None, logfile='output.log', lr=0.1, epoch=None, fun_name=None):
    folder = os.path.exists(logfile)
    if not folder:
        os.mknod(logfile)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(logfile, mode='a')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: \n %(message)s")
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    correct_num = np.diagonal(c_matrix)
    information = {'lr': lr, 'n_epoch': epoch, "Test F1": '%.4f' % test_f1, 'Best Val F1': '%.4f' % best_model_f1,
                   'Correct num': correct_num, "fun_name": fun_name}
    logger.info(information)
    logger.info(c_matrix)
    logger.info(per_eval)
# -*- coding: utf-8 -*-
from transformers import AutoTokenizer
import torch.optim as optim
from model.citation_model import *
from utils.scheduler import WarmupMultiStepLR
from train_valid.dataset_train import dataset_train
from train_valid.dataset_valid import dataset_valid
from utils.dataload import *
from utils.util import *
from sklearn.metrics import classification_report, confusion_matrix
import optuna
import time


def run_optuna(path, dev):
    print('Run optuna')
    setup_seed(0)
    token = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    criterion = nn.CrossEntropyLoss()
    # dataset = load_data(16, reverse=True, multi=True, mul_num=2400)
    dataset = load_data(16)

    def objective(trial):
        model = Model('allenai/scibert_scivocab_uncased')
        n_epoch = trial.suggest_int('n_epoch',140, 170, log=True)
        lr = trial.suggest_float('lr', 1e-4, 1e-3, log=True)
        au_weight = trial.suggest_float('au_weight', 0.001, 0.01, log=True)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
        scheduler = WarmupMultiStepLR(optimizer, [90, 110], gamma=0.1, warmup_epochs=5)
        best_model_f1, best_epoch = dataset_train(model, token, dataset, criterion, optimizer, n_epoch, au_weight, dev,
                                                    scheduler, model_path=path)

        return best_model_f1
    study = optuna.create_study(study_name='studyname', direction='maximize', storage='sqlite:///optuna.db', load_if_exists=True)
    study.optimize(objective, n_trials=6)
    print("Best_Params:{} \t Best_Value:{}".format(study.best_params, study.best_value))
    history = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
    print(history)


def main_run(path, dev):
    setup_seed(0)
    token = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    model = Model('allenai/scibert_scivocab_uncased')
    criterion = nn.CrossEntropyLoss()
    lr = 0.000184
    au_weight = 0.007413
    n_epoch = 151
    # dataset = load_data(16, reverse=True, multi=True, mul_num=2400)
    dataset = load_data(16)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=2e-4)
    scheduler = WarmupMultiStepLR(optimizer, [90, 110], gamma=0.1, warmup_epochs=5)
    best_model_f1, best_epoch = dataset_train(model, token, dataset, criterion, optimizer, n_epoch, au_weight, dev,
                                                scheduler, model_path=path)
    print("best_model_f1:{} \t best_epoch:{}".format(best_model_f1, best_epoch))

    test_f1, test_micro_f1, test_true_label, test_pre_label = dataset_valid(model, token,
                                                                         dataset['test'], device,
                                                                         mode='test', path=path)
    print('Test'.center(20, '='))
    print('Test_True_Label:', collections.Counter(test_true_label))
    print('Test_Pre_Label:', collections.Counter(test_pre_label))
    print('Test macro F1: %.4f \t Test micro F1: %.4f' % (test_f1, test_micro_f1))
    print('Test'.center(20, '='))
    test_true = torch.Tensor(test_true_label).tolist()
    test_pre = torch.Tensor(test_pre_label).tolist()
    generate_submission(test_pre, 'mul_rev_val_f1_{:.5}_best_epoch_{}'.format(best_model_f1, best_epoch), test_f1)
    c_matrix = confusion_matrix(test_true, test_pre, labels=[0, 1, 2, 3, 4, 5])
    per_eval = classification_report(test_true, test_pre, labels=[0, 1, 2, 3, 4, 5])
    log_result(test_f1, best_model_f1,  c_matrix, per_eval, lr=lr, epoch=n_epoch, fun_name='main_multi_rev')


if __name__ == "__main__":
    tst = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    run_optuna('/content/citation_classification/citation_mul_rev_model.pth', device)
    # main_run('/content/citation_classification/citation_mul_rev_model.pth', device)
    ten = time.time()
    print('Total time: {}'.format((ten - tst)))

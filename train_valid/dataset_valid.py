# -*- coding: utf-8 -*-
import torch
from sklearn.metrics import f1_score


def dataset_valid(model, tokenizer, valid, device, mode=None, path=None, criterion=None):

    if mode == 'test':
        model_state = torch.load(path)
        model.load_state_dict(model_state)
        model.to(device)
    true_label = []
    pre_label = []
    avg_loss = 0
    model.eval()
    with torch.no_grad():
        for index, (sentence, target) in enumerate(zip(valid['sen'], valid['tar'])):
            sentences = tokenizer(sentence, return_tensors='pt', is_split_into_words=True, padding=True,
                                  return_length=True)
            sentences = sentences.to(device)
            output = model(sentences)
            if criterion is not None:
                loss_target = torch.LongTensor(target)
                loss = criterion(output, loss_target.to(device))
                avg_loss += loss.item()
                if (index + 1) % 10 == 0:
                    print('Batch: %d \t Valid_Loss: %.4f' % (index + 1, avg_loss / 10))
                    avg_loss = 0
            output = torch.softmax(output.cpu(), dim=1)
            predict_test_value, predict_test_label = torch.max(output, dim=1)
            pre_label.extend(predict_test_label.tolist())
            true_label.extend(target)
    macro_f1 = f1_score(torch.LongTensor(true_label), torch.LongTensor(pre_label), average='macro')
    micro_f1 = f1_score(torch.LongTensor(true_label), torch.LongTensor(pre_label), average='micro')
    return (macro_f1, micro_f1, true_label, pre_label) if mode == 'test' else (macro_f1, micro_f1)
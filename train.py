import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
import logging as lgg

from utils import get_time_dif
from pytorch_pretrained_bert.optimization import BertAdam


def train_attention(tran_iter, test_iter, dev_iter, model, criterion, cf):
    total_batch = 0
    dev_best_acc_top = 0.0
    dev_best_f1_top = 0.0
    last_improve = 0
    flag = False

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.00}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=cf.learning_rate,
                         warmup=0.05,
                         t_total=len(tran_iter) * cf.num_epochs)

    for epoch in range(cf.num_epochs):
        model.train()
        start_time = time.time()
        lgg.info('Epoch [{}/{}]'.format(epoch + 1, cf.num_epochs))

        for i, (trains, y1, y2, arg_mask) in enumerate(tran_iter):

            optimizer.zero_grad()
            pre1 = model(trains)
            # pre1 = model(trains)
            if cf.class_n == 4:
                y = y1[0]
            else:
                y = y1[1]

            # loss1 = model.crf(emissions=pre1, tags=y, mask=None)
            loss1 = criterion(pre1, y)
            loss = loss1
            # loss = loss1
            loss.backward(retain_graph=True)
            optimizer.step()
            total_batch += 1

            # if total_batch % 100 == 0:
            if total_batch % 100 == 0:
                if cf.need_clc_loss:
                    y_predit_top = torch.max(pre1.data, dim=1)[1].data.cpu()
                    # outputs_top_em = torch.div(outputs_top + outputs_top_reverse, 2)
                    # outputs_sec_em = torch.div(outputs_sec + outputs_sec_reverse, 2)
                    # outputs_conn_em = torch.div(outputs_conn + outputs_conn_reverse, 2)
                    # y_predit_top_reverse = torch.max(outputs_top_reverse.data, 1)[1].cpu()
                    # y_predit_sec_reverse = torch.max(outputs_sec_reverse.data, 1)[1].cpu()
                    # y_predit_conn_reverse = torch.max(outputs_conn_reverse.data, 1)[1].cpu()
                else:
                    y_predit_top = pre1.data.cpu()

                train_acc_top = metrics.accuracy_score(y.data.cpu(), y_predit_top)

                loss_dev, acc_top, f1_top = evaluate(cf, model, dev_iter, criterion)
                if f1_top > dev_best_f1_top:
                    dev_best_f1_top = f1_top
                    torch.save(model.state_dict(), cf.save_path_top)
                    improve = '*，update model'
                    last_improve = total_batch
                else:
                    improve = ''

                time_dif = get_time_dif(start_time)

                msg = 'top-down:TOP: Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' + \
                      'Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Val F1: {5:>6.2%} Time: {6} {7}'
                lgg.info(msg.format(total_batch, loss.item(), train_acc_top, loss_dev, acc_top, f1_top, time_dif,
                                    improve))
                lgg.info(' ')
            model.train()
        time_dif = get_time_dif(start_time)
        lgg.info("Train time usage: {}".format(time_dif))
    test_loss, acc_top_test, f1_top_test = test(cf, model, test_iter, criterion)


def test(config, model, test_iter, criterion, reverse=False, ensemble=False):
    model.load_state_dict(torch.load(config.save_path_top))
    model.eval()
    start_time = time.time()

    test_loss, acc_top, f1_top = evaluate(config, model, test_iter, criterion, test=True,
                                          reverse=reverse, ensemble=ensemble)

    time_dif = get_time_dif(start_time)
    lgg.info("Test time usage: {}".format(time_dif))
    msg = 'TOP: Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}, Test F1: {2:>6.2%}'
    lgg.info(msg.format(test_loss, acc_top, f1_top))

    return test_loss, acc_top, f1_top


def evaluate(config, model, data_iter, criterion, test=False, reverse=False, ensemble=False):
    model.eval()
    loss_total = 0
    predict_all_top = np.array([], dtype=int)
    labels1_all_top = np.array([], dtype=int)
    labels2_all_top = np.array([], dtype=int)

    with torch.no_grad():
        for i, (trains, y1, y2, arg_mask) in enumerate(data_iter):
            if config.class_n == 4:
                y = y1[0]
                y_pre = y2[0]
            else:
                y = y1[1]
                y_pre = y2[1]
            pre1 = model(trains)
            # pre1 = model(trains)
            loss_top1 = F.cross_entropy(pre1, y)
            loss_total += loss_top1
            y1_true_top = y.data.cpu().numpy()
            y2_true_top = y_pre.data.cpu().numpy()
            y_predit_top = torch.max(pre1.data, dim=1)[1].data.cpu()
            labels1_all_top = np.append(labels1_all_top, y1_true_top)
            labels2_all_top = np.append(labels2_all_top, y2_true_top)
            predict_all_top = np.append(predict_all_top, y_predit_top)

    predict_sense_top = predict_all_top
    gold_sense_top = labels1_all_top
    mask = (predict_sense_top == labels2_all_top)
    gold_sense_top[mask] = labels2_all_top[mask]

    acc_top = metrics.accuracy_score(gold_sense_top, predict_sense_top)
    f1_top = metrics.f1_score(gold_sense_top, predict_sense_top, average='macro')

    return loss_total / len(data_iter), acc_top, f1_top

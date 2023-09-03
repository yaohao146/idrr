import random

import torch
import transformers
from torch import nn
from tqdm import tqdm
import time

from transformers import RobertaTokenizer, RobertaModel
import yh_model

PAD, CLS, SEP = ['[PAD]'], ['[CLS]'], ['[SEP]']


# def roberta_embeding(tokenizer, model, text1, text2):
#     text = "[CLS] " + text1 + " [SEP] " + text2 + " [EOS]"
#     encoded_input = tokenizer(text, return_tensors='pt').to(device='cuda')
#     with torch.no_grad():
#         output = model(**encoded_input)
#     output_embeding = output['last_hidden_state']
#     return output_embeding[0]


def build_dataset(config):


    def load_data_atten(path):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue

                labels1, labels2, arg1, arg2 = [_.strip() for _ in lin.split('|||')]
                labels1, labels2 = eval(labels1), eval(labels2)
                if config.dataset == "corpus/pdtb2/":
                    labels1[0] = config.top2id[labels1[0]] if labels1[0] is not None else -1
                    labels1[1] = config.sec2id[labels1[1]] if labels1[1] is not None else -1
                    labels1[2] = config.conn2id[labels1[2]] if labels1[2] is not None else -1
                    labels2[0] = config.top2id[labels2[0]] if labels2[0] is not None else -1
                    labels2[1] = config.sec2id[labels2[1]] if labels2[1] is not None else -1
                    labels2[2] = config.conn2id[labels2[2]] if labels2[2] is not None else -1

                else:
                    labels1[0] = config.top2id[labels1[0]] if labels1[0] is not None else -1
                    if labels1[1] in config.sec2id:
                        labels1[1] = config.sec2id[labels1[1]] if labels1[1] is not None else -1
                    else:
                        continue
                    labels1[2] = config.conn2id[labels1[2]] if labels1[2] is not None else -1
                    labels2[0] = config.top2id[labels2[0]] if labels2[0] is not None else -1
                    if labels2[1] in config.sec2id:
                        labels2[1] = config.sec2id[labels2[1]] if labels2[1] is not None else -1
                    else:
                        labels2[1] = -1
                    labels2[2] = config.conn2id[labels2[2]] if labels2[2] is not None else -1

                arg1_token = config.tokenizer.tokenize(arg1)
                arg2_token = config.tokenizer.tokenize(arg2)
                token = CLS + arg1_token + SEP + arg2_token + SEP

                token_type_ids = [0] * (len(arg1_token) + 2) + [1] * (len(arg2_token) + 1)
                arg1_mask = [1] * (len(arg1_token) + 2)
                arg2_mask = [0] * (len(arg1_token) + 1) + [1] * (len(arg2_token) + 2)
                input = config.tokenizer(arg1, arg2, truncation=True, max_length=101, padding='max_length')
                input_ids = input['input_ids']
                attention_mask = input['attention_mask']

                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                if config.pad_size:
                    if seq_len < config.pad_size:
                        mask = [1] * len(token_ids) + [0] * (config.pad_size - seq_len)
                        token_ids += ([0] * (config.pad_size - seq_len))
                        token_type_ids += ([0] * (config.pad_size - seq_len))
                    else:
                        mask = [1] * config.pad_size
                        token_ids = token_ids[:config.pad_size]
                        token_type_ids = token_type_ids[:config.pad_size]

                    if len(arg1_mask) < config.pad_size:
                        arg1_mask += [0] * (config.pad_size - len(arg1_mask))
                    else:
                        arg1_mask = arg1_mask[:config.pad_size]

                    if len(arg2_mask) < config.pad_size:
                        arg2_mask += [0] * (config.pad_size - len(arg2_mask))
                    else:
                        arg2_mask = arg2_mask[:config.pad_size]

                contents.append([input_ids, seq_len, attention_mask, token_type_ids,
                                 labels1[0], labels1[1], labels1[2],
                                 labels2[0], labels2[1], labels2[2],
                                 arg1_mask, arg2_mask])
        return contents

    train = load_data_atten(config.train_path)
    dev = load_data_atten(config.dev_path)
    test = load_data_atten(config.test_path)

    return train, dev, test
    # return dev


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  #
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    # def _to_tensor(self, datas):
    #     x1 = torch.LongTensor([_[0] for _ in datas]).to(self.device)
    #     #
    #     top = torch.tensor([_[1] for _ in datas]).to(self.device)
    #     conn = torch.tensor([_[3].numpy().tolist() for _ in datas]).to(self.device)
    #     top_index = torch.tensor([_[4] for _ in datas]).to(self.device)
    #
    #     return x1, (top, sec, conn, top_index)

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        seq_len = torch.LongTensor([_[1] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        token_type = torch.LongTensor([_[3] for _ in datas]).to(self.device)

        y1_top = torch.LongTensor([_[4] for _ in datas]).to(self.device)
        y1_sec = torch.LongTensor([_[5] for _ in datas]).to(self.device)
        y1_conn = torch.LongTensor([_[6] for _ in datas]).to(self.device)
        y2_top = torch.LongTensor([_[7] for _ in datas]).to(self.device)
        y2_sec = torch.LongTensor([_[8] for _ in datas]).to(self.device)
        y2_conn = torch.LongTensor([_[9] for _ in datas]).to(self.device)

        arg1_mask = torch.LongTensor([_[10] for _ in datas]).to(self.device)
        arg2_mask = torch.LongTensor([_[11] for _ in datas]).to(self.device)

        return (x, seq_len, mask, token_type), \
               (y1_top, y1_sec, y1_conn), \
               (y2_top, y2_sec, y2_conn), \
               (arg1_mask, arg2_mask)

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index > self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """ """
    end_time = time.time()
    time_dif = end_time - start_time
    return time_dif
    # return timedelta(seconds=int(round(time_dif)))


if __name__ == '__main__':
    from config import Config

    cf = Config('corpus/pdtb2')
    cf.soft_n = "top"

    model = yh_model.model_all(cf.embed_dim, cf).to(device='cuda')

    criterion = nn.CrossEntropyLoss().to(device='cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=cf.learning_rate)
    train, dev, test = build_dataset(cf)
    # dev = build_gram(cf)
    train_iter = build_iterator(train, cf)
    dev_iter = build_iterator(dev, cf)
    test_iter = build_iterator(test, cf)
    train.train_attention(train_iter, test_iter, dev_iter, model, criterion, cf, optimizer)
    # selfattention_model.train_attention(dev_iter, model, criterion, cf, optimizer)

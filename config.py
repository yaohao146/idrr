import datetime
import json
import time
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel


class Config:
    def __init__(self, args, cuda=0, finetune=True, base='robert'):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        self.finetune_bert = finetune
        self.l1 = config['l1']
        self.l2 = config['l2']
        self.l3 = config['l3']
        if self.finetune_bert:
            self.num_epochs = config['num_epochs']
            self.learning_rate = config['learning_rate']
        else:
            self.num_epochs = 20
            self.learning_rate = 0.000005
        self.class_n = config['num_classes']
        self.dataset = config['dataset']
        self.model_name = config['model_name']
        self.train_path = self.dataset + '/train.txt'
        self.dev_path = self.dataset + '/dev.txt'
        self.test_path = self.dataset + '/test.txt'
        self.embed_dim = config['embed_dim']
        self.require_improment = config['require_improment']

        self.id2conn = [x.strip() for x in open(self.dataset + '/data/conn.txt').readlines()]
        self.conn2id = dict((x, xid) for xid, x in enumerate(self.id2conn))

        self.id2top = [x.strip() for x in open(self.dataset + '/data/class4.txt').readlines()]
        self.top2id = dict((x, xid) for xid, x in enumerate(self.id2top))

        self.id2sec = [x.strip() for x in open(self.dataset + '/data/class_n.txt').readlines()]
        self.sec2id = dict((x, xid) for xid, x in enumerate(self.id2sec))

        self.ffn_hidden = config['ffn_hidden']
        self.mlm_probability = config['mlm_probability']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.bert = base
        self.pad_size = config['pad_size']
        self.gram_size = config['gram_size']
        self.drop_prob = config['drop_prob']
        self.drop_cls = config['drop_cls']
        self.device = "cuda"
        # self.learning_rate = 0.1
        self.n_transformer = config['n_transformer']

        # self.num_epochs = 10

        self.tokenizer = RobertaTokenizer.from_pretrained("./roberta_base")

        self.batch_size = config['batch_size']
        self.bert_path = 'roberta_base'
        self.embed_dim = config['embed_dim']

        self.save_path_top = self.dataset + '/saved_dict/' + self.model_name + '_top.ckpt'  #
        self.save_path_sec = self.dataset + '/saved_dict/' + self.model_name + '_sec.ckpt'  #
        self.save_path_conn = self.dataset + '/saved_dict/' + self.model_name + '_conn.ckpt'
        t = time.strftime('%B%d-%H:%M:%S')
        self.log = self.dataset + 'log/' + self.model_name + '_' + str(t) + '.log'
        self.device = torch.device('cuda:{0}'.format(cuda) if torch.cuda.is_available() else 'cpu')  #

        # show training and test time
        self.show_time = True
        self.need_clc_loss = True

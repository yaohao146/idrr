import torch
import torch.nn as nn

from layers.transformer import Transformer
from transformers import RobertaModel
from layers.sublayer import EncoderLayer
from layers.crf import CRF


class model_all(nn.Module):
    def __init__(self, embed_dim, conf):
        super(model_all, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=2, padding=0).to(
            device='cuda')
        self.conv2 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=2, padding=0).to(
            device='cuda')
        self.conv3 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim, kernel_size=2, padding=0).to(
            device='cuda')
        self.bert = RobertaModel.from_pretrained("./roberta_base")
        for param in self.bert.parameters():
            param.requires_grad = conf.finetune_bert
        self.transformer1 = Transformer(
            d_model=conf.embed_dim,
            pad_size=conf.pad_size,
            max_len=conf.gram_size,
            ffn_hidden=conf.ffn_hidden,
            n_head=conf.n_heads,
            n_layers=conf.n_layers,
            drop_prob=conf.drop_prob,
            device=conf.device
        ).to(device='cuda')
        self.W1 = nn.Linear(embed_dim, embed_dim)
        self.U1 = nn.Linear(embed_dim, 1)
        self.linear_cls_H_1 = nn.Linear(1536, 4)
        self.linear_cls_H_2 = nn.Linear(768, 4)
        self.linear_cls_cls = nn.Linear(768, 4)
        self.linear_H_H = nn.Linear(768, 4)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        context = x[0]
        mask = x[2]
        bert_out = self.bert(context, attention_mask=mask)
        hidden_last, pooled, hidden_all = bert_out.last_hidden_state, bert_out.pooler_output, bert_out.hidden_states
        cls, encode_out = hidden_last[:, 0:1, :], hidden_last[:, 1:, :]
        gram_2 = self.relu(self.conv1(encode_out.transpose(1, 2)))
        gram_3 = self.relu(self.conv2(gram_2))
        gram_4 = self.relu(self.conv3(gram_3))
        output = torch.cat((encode_out.transpose(1, 2), gram_2), dim=2)
        output = torch.cat((output, gram_3), dim=2)
        output = torch.cat((output, gram_4), dim=2)
        output = output.transpose(1, 2)
        trans = self.transformer1(output)
        linear_transform = self.W1(trans)
        attention_weights = self.U1(linear_transform)
        attention_weights = torch.nn.functional.softmax(attention_weights, dim=1)
        weighted_sum = trans * attention_weights
        output_tensor = torch.sum(weighted_sum, dim=1)
        cls = cls.squeeze(1)
        output_tensor = self.drop(output_tensor)

        # cat(cls,n_gram)的分类
        # cat = torch.cat((cls, output_tensor), dim=1)
        # pre1 = self.linear_cls_H_1(cat)

        # cls的分类
        # pre2 = self.linear_cls_cls(cls)

        # n-gram的分类
        pre1 = self.linear_H_H(output_tensor)

        # return pre1, pre2, pre3
        return pre1

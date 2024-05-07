import os
import torch
from torch_geometric.nn import Linear
from egnn_pytorch import EGNN
from torch.nn import Sequential, BatchNorm1d, Dropout, Sigmoid, Conv1d, LSTM, LayerNorm, ReLU
from torchdrug import models
from CrossAttention import CrossAttention

class MIPE(torch.nn.Module):
    def __init__(self, share_weight=False, dropout=0.5, heads=4):
        super(MIPE, self).__init__()
        self.node_attr_dim = 62
        self.hidden_dim = 64
        self.esm_dim = 1280
        self.prott5_dim = 1024
        self.ablang_dim = 768
        self.hidden_dim_cnn = 64
        self.hidden_dim_cnn2 = 128
        self.hidden_dim_cnn3 = 256
        self.h1_dim = 64
        self.h2_dim = 64
        self.share_weight = share_weight
        self.dropout = dropout
        self.heads = 1
        self.multiheads = 16
        # GearNet
        self.gearnet = models.GearNet(input_dim=62, hidden_dims=[64, 64, 64], num_relation=3)
        # #CNN
        self.ag_cnn1 = Conv1d(in_channels=self.esm_dim, out_channels=self.hidden_dim_cnn, kernel_size=5, padding='same')
        if self.share_weight:
            self.ab_cnn1 = self.ag_cnn1
        else:
            self.ab_cnn1 = Conv1d(in_channels=self.ablang_dim, out_channels=self.hidden_dim_cnn, kernel_size=5,padding='same')
        # DilatedCNN
        self.ag_cnn2 = Conv1d(self.hidden_dim_cnn, self.hidden_dim_cnn2, 3, dilation=2, padding='same')
        self.ag_cnn3 = Conv1d(self.hidden_dim_cnn2, self.hidden_dim_cnn3, 3, dilation=4, padding='same')
        if self.share_weight:
            self.ab_cnn2 = self.ag_cnn2
            self.ab_cnn3 = self.ag_cnn3
        else:
            self.ab_cnn2 = Conv1d(self.hidden_dim_cnn, self.hidden_dim_cnn2, 3, dilation=2, padding='same')
            self.ab_cnn3 = Conv1d(self.hidden_dim_cnn2, self.hidden_dim_cnn3, 3, dilation=4, padding='same')
        # BiLSTM
        self.ag_lstm = LSTM(input_size=self.hidden_dim_cnn3, hidden_size=self.h1_dim // 2, num_layers=1,batch_first=True, bidirectional=True)
        if self.share_weight:
            self.ab_lstm = self.ag_lstm
        else:
            self.ab_lstm = LSTM(input_size=self.hidden_dim_cnn3, hidden_size=self.h1_dim // 2, num_layers=1,batch_first=True, bidirectional=True)
        self.linear_ag_seq = Sequential(Linear(self.h1_dim, 1), Sigmoid())
        self.linear_ag_strc = Sequential(Linear(self.h1_dim, 1), Sigmoid())
        self.linear_ab_seq = Sequential(Linear(self.h1_dim, 1), Sigmoid())
        self.linear_ab_strc = Sequential(Linear(self.h1_dim, 1), Sigmoid())
        self.linear_ag = Linear(self.h1_dim + self.h1_dim, self.h1_dim)
        self.linear_ab = Linear(self.h1_dim + self.h1_dim, self.h1_dim)
        self.ag_bnorm1 = Sequential(BatchNorm1d(self.hidden_dim_cnn, track_running_stats=False), Dropout(0.5))
        self.ag_bnorm2 = Sequential(BatchNorm1d(self.hidden_dim_cnn2, track_running_stats=False), Dropout(0.5))
        self.ag_bnorm3 = Sequential(BatchNorm1d(self.hidden_dim_cnn3, track_running_stats=False), Dropout(0.5))
        self.ag_bnorm4 = Sequential(BatchNorm1d(self.hidden_dim, track_running_stats=False), Dropout(0.5))
        self.ab_bnorm1 = Sequential(BatchNorm1d(self.hidden_dim_cnn, track_running_stats=False), Dropout(0.5))
        self.ab_bnorm2 = Sequential(BatchNorm1d(self.hidden_dim_cnn2, track_running_stats=False), Dropout(0.5))
        self.ab_bnorm3 = Sequential(BatchNorm1d(self.hidden_dim_cnn3, track_running_stats=False), Dropout(0.5))
        self.ab_bnorm4 = Sequential(BatchNorm1d(self.hidden_dim, track_running_stats=False), Dropout(0.5))
        # CrossAttention
        self.ag_crossattention = CrossAttention(self.h1_dim, self.multiheads)
        self.ab_crossattention = CrossAttention(self.h1_dim, self.multiheads)
        self.ag_feed_forward = Sequential(
            Linear(self.h1_dim, 4 * self.h1_dim),
            ReLU(),
            Linear(4 * self.h1_dim, self.h1_dim)
        )
        self.ag_norm1 = LayerNorm(self.h1_dim)
        self.ag_norm2 = LayerNorm(self.h1_dim)
        self.ab_feed_forward = Sequential(
            Linear(self.h1_dim, 4 * self.h1_dim),
            ReLU(),
            Linear(4 * self.h1_dim, self.h1_dim)
        )
        self.ab_norm1 = LayerNorm(self.h1_dim)
        self.ab_norm2 = LayerNorm(self.h1_dim)
        self.ag_linearsigmoid_linear = Linear(self.h1_dim, 1)
        self.ag_linearsigmoid_sigmoid = Sigmoid()
        self.ab_linearsigmoid_linear = Linear(self.h1_dim, 1)
        self.ab_linearsigmoid_sigmoid = Sigmoid()
        self.ag_linearsigmoid = Sequential(Linear(self.h1_dim, 1), Sigmoid())
        self.ab_linearsigmoid = Sequential(Linear(self.h1_dim, 1), Sigmoid())

    def forward(self, *agab):
        ag_x = agab[0]
        ag_edge_index = agab[1]
        ab_x = agab[2]
        ab_edge_index = agab[3]
        ag_esm = agab[4]
        ab_esm = agab[5]
        # Ag
        ag_edge_index[1] = torch.nn.functional.normalize(ag_edge_index[1])
        ag_h1 = self.gearnet(ag_edge_index[0], ag_edge_index[1])["node_feature"]
        ag_out_strc = self.linear_ag_strc(ag_h1)
        ag_esm = torch.nn.functional.normalize(ag_esm)
        ag_esm = self.ag_bnorm1((self.ag_cnn1(ag_esm)))
        ag_esm = self.ag_bnorm2((self.ag_cnn2(ag_esm)))
        ag_esm = self.ag_bnorm3((self.ag_cnn3(ag_esm)))
        ag_esm = ag_esm.transpose(1, 2)
        output_tensor, _ = self.ag_lstm(ag_esm)
        ag_h2 = torch.cat((output_tensor[:, :, :64 // 2], output_tensor[:, :, 64 // 2:]), dim=2)
        ag_h2 = torch.squeeze(ag_h2, dim=0)
        ag_h2 = self.ag_bnorm4(ag_h2)
        ag_out_seq = self.linear_ag_seq(ag_h2)
        ag_h1 = self.linear_ag(torch.cat((ag_h1, ag_h2), dim=1))
        # Ab
        ab_edge_index[1] = torch.nn.functional.normalize(ab_edge_index[1])
        ab_h1 = self.gearnet(ab_edge_index[0], ab_edge_index[1])["node_feature"]
        ab_out_strc = self.linear_ab_strc(ab_h1)
        ab_esm = torch.nn.functional.normalize(ab_esm)
        ab_esm = self.ab_bnorm1((self.ab_cnn1(ab_esm)))
        ab_esm = self.ab_bnorm2((self.ab_cnn2(ab_esm)))
        ab_esm = self.ab_bnorm3((self.ab_cnn3(ab_esm)))
        ab_esm = ab_esm.transpose(1, 2)
        output_tensor, _ = self.ab_lstm(ab_esm)
        ab_h2 = torch.cat((output_tensor[:, :, :64 // 2], output_tensor[:, :, 64 // 2:]), dim=2)
        ab_h2 = torch.squeeze(ab_h2, dim=0)
        ab_h2 = self.ab_bnorm4(ab_h2)
        ab_out_seq = self.linear_ab_seq(ab_h2)
        ab_h1 = self.linear_ab(torch.cat((ab_h1, ab_h2), dim=1))
        # CrossAttention
        ag_attention, ag_attention_weights = self.ag_crossattention(ag_h1, ab_h1)
        ab_attention, ab_attention_weights = self.ab_crossattention(ab_h1, ag_h1)
        ag_res1 = self.ag_norm1(ag_h1 + ag_attention)
        ab_res1 = self.ab_norm1(ab_h1 + ab_attention)
        ag_res2 = self.ag_norm2(ag_res1 + self.ag_feed_forward(ag_res1))
        ab_res2 = self.ab_norm2(ab_res1 + self.ab_feed_forward(ab_res1))
        ag_out_0 = self.ag_linearsigmoid_linear(ag_res2)
        ag_out = self.ag_linearsigmoid_sigmoid(ag_out_0)
        ab_out_0 = self.ab_linearsigmoid_linear(ab_res2)
        ab_out = self.ab_linearsigmoid_sigmoid(ab_out_0)
        return ag_out, ab_out, ag_attention_weights, ab_attention_weights, ag_h1, ag_attention, ab_h1, ab_attention,\
               ag_h2, ab_h2, ag_out_seq, ag_out_strc, ab_out_seq, ab_out_strc
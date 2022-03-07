import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GATConv, GatedGraphConv


class SR_GNN(nn.Module):
    def __init__(self, opt):
        super(SR_GNN, self).__init__()
        # hyper parameters
        self.device = opt.device
        self.hidden_size = opt.hidden_size
        self.num_item = opt.num_item
        self.batch_size = opt.batch_size
        self.hybrid = opt.hybrid

        # network
        self.embedding = nn.Embedding(self.num_item + 1, self.hidden_size)  # Add mask item (index = 0)
        self.ggnn = GatedGraphConv(out_channels=self.hidden_size, num_layers=opt.layer_gnn)
        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.fc3 = nn.Linear(self.hidden_size, 1, bias=False)
        if self.hybrid:
            self.fc4 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.loss_function = nn.CrossEntropyLoss()

        # init parameters
        std_var = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std_var, std_var)

    def forward(self, A, items, seq_alias):
        num_sample = len(seq_alias)
        h_items = self.ggnn(self.embedding(items), A)
        def embed_seq(i): return h_items[i][seq_alias[i]]
        h_seqs = torch.stack([embed_seq(i) for i in torch.arange(num_sample).long()])
        return h_seqs

    def predict(self, data):
        A_batch, items_batch, seq_alias_batch, mask_batch = map(lambda x: x.to(self.device), data)

        # learning item embeddings on session graphs
        h_seqs_batch = self.forward(A_batch, items_batch, seq_alias_batch)
        h_tail_batch = h_seqs_batch[torch.arange(mask_batch.shape[0]).long(), torch.sum(mask_batch, 1) - 1]
        # generating session embedding
        q1 = self.fc1(h_tail_batch).unsqueeze(1)  # batch_size x 1 x latent_size
        q2 = self.fc2(h_seqs_batch)  # batch_size x seq_length x latent_size
        alpha_att = self.fc3(torch.sigmoid(q1 + q2))
        seq_emb = torch.sum(alpha_att * h_seqs_batch * mask_batch.unsqueeze(-1), 1)  # global embedding
        if self.hybrid:
            seq_emb = self.fc4(torch.cat([seq_emb, h_tail_batch], 1))
        item_emb = self.embedding.weight[1:].transpose(1, 0)
        scores_batch = torch.matmul(seq_emb, item_emb)
        return scores_batch

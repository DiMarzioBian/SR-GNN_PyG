from abc import ABC
from argparse import Namespace
from typing import Union, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, Dataset, DataLoader


class MultiSessionsGraph(Dataset):
    """Every session is a graph."""

    def __init__(self, f, transform=None, pre_transform=None):
        """
        Args:
            f: pickle loaded file
            phrase: 'train' or 'test'
        """
        self.f = f
        super(MultiSessionsGraph, self).__init__(f, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        pass

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        pass

    def len(self) -> int:
        pass

    def get(self, idx: int) -> Data:
        pass

    def download(self):
        pass

    def process(self):
        data_list = []
        for sequences, y in zip(self.f[0], self.f[1]):
            i = 0
            nodes = {}  # dict{15: 0, 16: 1, 18: 2, ...}
            senders = []
            x = []
            for node in sequences:
                if node not in nodes:
                    nodes[node] = i
                    x.append([node])
                    i += 1
                senders.append(nodes[node])
            receivers = senders[:]
            del senders[-1]  # the last item is a receiver
            del receivers[0]  # the first item is a sender
            edge_index = torch.tensor([senders, receivers], dtype=torch.long)
            x = torch.tensor(x, dtype=torch.long)
            y = torch.tensor([y], dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index, y=y))

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def get_sample_dataloader(opt: Namespace,
                          train_data: tuple,
                          test_data: tuple,
                          valid_data: tuple = (None, None)):
    """ Load data and prepare dataloader. """

    # Instancelize dataloader
    train_loader = DataLoader(MultiSessionsGraph(train_data), batch_size=opt.batch_size,
                              num_workers=opt.num_workers, shuffle=True)
    test_loader = DataLoader(MultiSessionsGraph(test_data), batch_size=opt.batch_size,
                             num_workers=opt.num_workers, shuffle=False)
    # Validation set
    if opt.val_split_rate > 0:
        valid_loader = DataLoader(MultiSessionsGraph(valid_data), batch_size=opt.batch_size,
                                  num_workers=opt.num_workers, shuffle=False)
    else:
        valid_loader = test_loader

    return train_loader, valid_loader, test_loader

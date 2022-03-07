from typing import Union, List, Tuple
import pickle
import torch
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader


class MultiSessionsGraphInMemory(InMemoryDataset):
    """Every session is a graph."""

    def __init__(self, opt, mode, transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            opt: args parser
            mode: 'train', 'test' or 'valid'
        """
        self.mode = mode
        self.root = '_data/'+opt.benchmark
        super(MultiSessionsGraphInMemory, self).__init__(self.root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.length = self.data.shape[0]

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return [self.root+'/processed_in_memory/'+str(self.mode)+'.txt']

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return [self.root+'/processed_in_memory/'+str(self.mode)+'.pt']

    # def len(self) -> int:
    #     return self.length

    def process(self):
        data = pickle.load(open(self.raw_dir + '/' + self.raw_file_names[0], 'rb'))
        data_list = []
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

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

    # def get(self, idx: int) -> Data:
    #     pass


class MultiSessionsGraph(Dataset):
    """ Every session is a graph. """
    def __init__(self, opt, mode, transform=None, pre_transform=None):
        """
        Args:
            opt: args parser
            mode: 'train', 'test' or 'valid'
        """
        self.name = opt.benchmark
        self.mode = mode
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.length = self.data.shape[0]
        super(MultiSessionsGraph, self).__init__(opt, mode, transform, pre_transform)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return ['_data/'+self.name+'processed'+str(self.mode)+'.txt']

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['_data/'+self.name+'processed'+str(self.mode)+'.pt']

    def download(self):
        pass

    def len(self) -> int:
        return self.length

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

    def get(self, idx: int) -> Data:
        pass


def get_sample_dataloader(opt):
    """ Load data and prepare dataloader. """

    if opt.in_memory:
        # Instancelize dataloader
        # Entire data in memory
        train_loader = DataLoader(MultiSessionsGraphInMemory(opt, mode='train'), batch_size=opt.batch_size,
                                  num_workers=opt.num_workers, shuffle=True)
        test_loader = DataLoader(MultiSessionsGraphInMemory(opt, mode='test'), batch_size=opt.batch_size,
                                 num_workers=opt.num_workers, shuffle=False)
        if opt.val_split_rate > 0:
            valid_loader = DataLoader(MultiSessionsGraphInMemory(opt, mode='valid'), batch_size=opt.batch_size,
                                      num_workers=opt.num_workers, shuffle=False)
        else:
            valid_loader = test_loader
    else:
        train_loader = DataLoader(MultiSessionsGraph(opt, mode='train'), batch_size=opt.batch_size,
                                  num_workers=opt.num_workers, shuffle=True)
        test_loader = DataLoader(MultiSessionsGraph(opt, mode='test'), batch_size=opt.batch_size,
                                 num_workers=opt.num_workers, shuffle=False)
        if opt.val_split_rate > 0:
            valid_loader = DataLoader(MultiSessionsGraph(opt, mode='valid'), batch_size=opt.batch_size,
                                      num_workers=opt.num_workers, shuffle=False)
        else:
            valid_loader = test_loader

    return train_loader, valid_loader, test_loader

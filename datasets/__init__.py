from datasets.dataloader_sample import get_sample_dataloader


def get_dataloader(opt):
    if opt.benchmark == 'sample':
        train_loader, val_loader, test_loader = get_sample_dataloader(opt)
    else:
        raise RuntimeError('Dataset ' + opt.benchmark + ' not found!')
    return train_loader, val_loader, test_loader

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# def get_imagenet_dataloaders(data_folder, batch_size=128, num_workers=8, distributed=False, rank=0, world_size=1):
#     """
#     ImageNet data loader
#     """
#     train_transform = transforms.Compose([
#         transforms.RandomResizedCrop(224),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])
#     val_transform = transforms.Compose([
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#     ])

#     train_set = datasets.ImageFolder(os.path.join(data_folder, 'train'), transform=train_transform)
#     val_set = datasets.ImageFolder(os.path.join(data_folder, 'val'), transform=val_transform)

#     # 如果启用了分布式训练(distributed),则需要使用 DistributedSampler.
#     # 它确保每个进程只加载自己负责的数据分片，避免数据重复加载.
#     if distributed:
#         train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank)
#         shuffle = False  # DistributedSampler handles shuffling
#     else:
#         train_sampler = None
#         shuffle = True

#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, sampler=train_sampler)
#     val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

#     return train_loader, val_loader, train_sampler

# 没加 DistributedSampler 前的版本
def get_imagenet_dataloaders(data_folder, batch_size=128, num_workers=8):
    """
    ImageNet data loader
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    train_set = datasets.ImageFolder(os.path.join(data_folder, 'train'), transform=train_transform)
    val_set = datasets.ImageFolder(os.path.join(data_folder, 'val'), transform=val_transform)


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader
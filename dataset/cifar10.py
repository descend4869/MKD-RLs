from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloaders(data_folder, batch_size=128, num_workers=8):
    """
    CIFAR-10 数据加载器
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    # 采用标准的CIFAR-10数据集的统计值，训练和测试使用同样的均值和标准差

    train_set = datasets.CIFAR10(root=data_folder,
                                 download=True,
                                 train=True,
                                 transform=train_transform)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    test_set = datasets.CIFAR10(root=data_folder,
                                download=True,
                                train=False,
                                transform=test_transform)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    return train_loader, test_loader
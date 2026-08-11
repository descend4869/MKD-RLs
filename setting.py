

# ------------- teacher net --------------------#
teacher_model_path_dict = {
    'RegNetY_400MF': '/data/myh/checkpoints/mkd_checkpoints/teachers/pretrained_models/RegNetY_400MF_best.pth',
    'RegNetX_400MF': '/data/myh/checkpoints/mkd_checkpoints/teachers/pretrained_models/RegNetX_400MF_best.pth',
    'resnet32x4': '/data/myh/checkpoints/mkd_checkpoints/teachers/pretrained_models/resnet32x4_best.pth',
    'resnet110x2': '/data/myh/checkpoints/mkd_checkpoints/teachers/pretrained_models/resnet110x2_best.pth',
    'wrn_28_4': '/data/myh/checkpoints/mkd_checkpoints/teachers/pretrained_models/wrn_28_4_best.pth',
    # 'ResNet50': '/data/winycg/imagenet_pretrained/resnet50-0676ba61.pth',
    # 'ResNet101': '/data/winycg/imagenet_pretrained/resnet101-63fe2227.pth',
    # 'wide_resnet50_2': '/data/winycg/imagenet_pretrained/wide_resnet50_2-95faca4d.pth',
    # 'resnext50_32x4d': '/data/winycg/imagenet_pretrained/resnext50_32x4d-7cdf4587.pth',
    }


# New: 针对CIFAR-10数据集，预训练好的teacher的存放位置
cifar10_teacher_model_path_dict = {
    'RegNetY_400MF': '/data/myh/checkpoints/mkd_checkpoints/teachers_cifar10/pretrained_models/RegNetY_400MF_best.pth',
    'RegNetX_400MF': '/data/myh/checkpoints/mkd_checkpoints/teachers_cifar10/pretrained_models/RegNetX_400MF_best.pth',
    'resnet32x4': '/data/myh/checkpoints/mkd_checkpoints/teachers_cifar10/pretrained_models/resnet32x4_best.pth',
    # 'resnet110x2': '/data/myh/checkpoints/mkd_checkpoints/teachers_cifar10/pretrained_models/resnet110x2_best.pth',
    'wrn_28_4': '/data/myh/checkpoints/mkd_checkpoints/teachers_cifar10/pretrained_models/wrn_28_4_best.pth',
    }
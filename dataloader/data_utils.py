import numpy as np
import torch

def set_up_datasets(args):
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset =="manyshotcifar":
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = args.shot_num
        args.sessions = 9
    if args.dataset == 'cub200':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11
    
    if args.dataset == 'manyshotcub':
        import dataloader.cub200.cub200 as Dataset
        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = args.shot_num
        args.sessions = 11

    if args.dataset == 'mini_imagenet':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    if args.dataset == 'mini_imagenet_withpath':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    
    
    if args.dataset == 'manyshotmini':
        import dataloader.miniimagenet.miniimagenet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = args.shot_num
        args.sessions = 9
    
    if args.dataset == 'imagenet100':
        import dataloader.imagenet100.ImageNet as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    if args.dataset == 'imagenet1000':
        import dataloader.imagenet1000.ImageNet as Dataset
        args.base_class = 600
        args.num_classes=1000
        args.way = 50
        args.shot = 5
        args.sessions = 9
    
    if args.dataset == 'cicids2017_improved':
        import dataloader.cicids2017_improved.cicids2017_improved as Dataset
        args.base_class = 4  # 基本クラス数（最初の4クラス: 0,1,2,3）
        args.num_classes = 10  # 総クラス数（0-9の10クラス）
        args.way = 1  # 各セッションで追加するクラス数
        args.shot = 5  # 各クラスのサンプル数
        args.sessions = 6  # セッション数 (base + 6 incremental sessions)

    args.Dataset=Dataset
    return args

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args, session)
    return trainset, trainloader, testloader

def get_base_dataloader(args):
    # txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)

    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(
            root=args.dataroot, train=True, download=True,
            index=class_index, base_sess=True
        )
        testset = args.Dataset.CIFAR100(
            root=args.dataroot, train=False, download=False,
            index=class_index, base_sess=True
        )
    elif args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(
            root=args.dataroot, train=True,
            index=class_index, base_sess=True
        )
        testset = args.Dataset.CUB200(
            root=args.dataroot, train=False, index=class_index
        )
    elif args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(
            root=args.dataroot, train=True,
            index=class_index, base_sess=True
        )
        testset = args.Dataset.MiniImageNet(
            root=args.dataroot, train=False, index=class_index
        )
    elif args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = args.Dataset.ImageNet(
            root=args.dataroot, train=True,
            index=class_index, base_sess=True
        )
        testset = args.Dataset.ImageNet(
            root=args.dataroot, train=False, index=class_index
        )
    elif args.dataset == 'cicids2017_improved':
        # For base session, use session_0.txt file
        txt_path = "data/index_list/" + args.dataset + "/session_0.txt"
        trainset = args.Dataset.CICIDS2017Improved(
            root=args.dataroot, train=True, max_samples=getattr(args, 'max_samples', None),
            index=txt_path
        )
        testset = args.Dataset.CICIDS2017Improved(
            root=args.dataroot, train=False, max_samples=getattr(args, 'max_samples', None),
            index=txt_path
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    trainloader = torch.utils.data.DataLoader(
        dataset=trainset, batch_size=args.batch_size_base, shuffle=True,
        num_workers=8, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )

    return trainset, trainloader, testloader


def get_new_dataloader(args,session):
    txt_path = "data/index_list/" + args.dataset + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(
            root=args.dataroot, train=True, download=False,
            index=class_index, base_sess=False
        )
    elif args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(
            root=args.dataroot, train=True,
            index_path=txt_path
        )
    elif args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(
            root=args.dataroot, train=True,
            index_path=txt_path
        )
    elif args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = args.Dataset.ImageNet(
            root=args.dataroot, train=True,
            index_path=txt_path
        )
    elif args.dataset == 'cicids2017_improved':
        trainset = args.Dataset.CICIDS2017Improved(
            root=args.dataroot, train=True, max_samples=getattr(args, 'max_samples', None),
            index=txt_path
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(
            dataset=trainset, batch_size=batch_size_new, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
            num_workers=args.num_workers, pin_memory=True
        )

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(
            root=args.dataroot, train=False, download=False,
            index=class_new, base_sess=False
        )
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(
            root=args.dataroot, train=False,
            index=class_new
        )
    elif args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(
            root=args.dataroot, train=False,
            index=class_new
        )
    elif args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        testset = args.Dataset.ImageNet(
            root=args.dataroot, train=False,
            index=class_new
        )
    elif args.dataset == 'cicids2017_improved':
        # For testing, include all classes seen so far (base classes + new classes)
        all_classes = list(range(args.base_class + session * args.way))
        testset = args.Dataset.CICIDS2017Improved(
            root=args.dataroot, train=False, max_samples=getattr(args, 'max_samples', None),
            index=all_classes
        )
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")

    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list
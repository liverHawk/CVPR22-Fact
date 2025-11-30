import numpy as np
import torch
import os
from dataloader.sampler import CategoriesSampler


def set_up_datasets(args):
    if args.dataset == "cifar100":
        import dataloader.cifar100.cifar as Dataset

        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    # if args.dataset == "manyshotcifar":
    #     import dataloader.cifar100.manyshot_cifar as Dataset

    #     args.base_class = 60
    #     args.num_classes = 100
    #     args.way = 5
    #     args.shot = args.shot_num
    #     args.sessions = 9
    if args.dataset == "cub200":
        import dataloader.cub200.cub200 as Dataset

        args.base_class = 100
        args.num_classes = 200
        args.way = 10
        args.shot = 5
        args.sessions = 11

    # if args.dataset == "manyshotcub":
    #     import dataloader.cub200.manyshot_cub as Dataset

    #     args.base_class = 100
    #     args.num_classes = 200
    #     args.way = 10
    #     args.shot = args.shot_num
    #     args.sessions = 11

    if args.dataset == "mini_imagenet":
        import dataloader.miniimagenet.miniimagenet as Dataset

        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    # if args.dataset == "mini_imagenet_withpath":
    #     import dataloader.miniimagenet.miniimagenet_with_img as Dataset

    #     args.base_class = 60
    #     args.num_classes = 100
    #     args.way = 5
    #     args.shot = 5
    #     args.sessions = 9

    # if args.dataset == "manyshotmini":
    #     import dataloader.miniimagenet.manyshot_mini as Dataset

    #     args.base_class = 60
    #     args.num_classes = 100
    #     args.way = 5
    #     args.shot = args.shot_num
    #     args.sessions = 9

    if args.dataset == "imagenet100":
        import dataloader.imagenet100.ImageNet as Dataset

        args.base_class = 60
        args.num_classes = 100
        args.way = 5
        args.shot = 5
        args.sessions = 9

    if args.dataset == "imagenet1000":
        import dataloader.imagenet1000.ImageNet as Dataset

        args.base_class = 600
        args.num_classes = 1000
        args.way = 50
        args.shot = 5
        args.sessions = 9

    if args.dataset == "CICIDS2017_improved":
        import dataloader.cicids2017.cicids2017 as Dataset

        # CICIDS2017_improved has 10 classes after preprocessing (label consolidation)
        # Following research_data_drl preprocessing: labels are consolidated
        # Configuration: base_class + (way * sessions) = num_classes

        # Load from params.yaml if not set via command line
        from utils import load_params_yaml
        try:
            params = load_params_yaml("params.yaml")
            create_sessions = params.get("create_sessions", {})

            if not hasattr(args, 'base_class') or args.base_class is None:
                args.base_class = create_sessions.get("base_class", 4)
            if not hasattr(args, 'num_classes') or args.num_classes is None:
                args.num_classes = create_sessions.get("num_classes", 10)
            if not hasattr(args, 'way') or args.way is None:
                args.way = create_sessions.get("way", 1)
            if not hasattr(args, 'shot') or args.shot is None:
                args.shot = create_sessions.get("shot", 5)
        except:
            # Fallback to defaults if params.yaml loading fails
            if not hasattr(args, 'base_class') or args.base_class is None:
                args.base_class = 4
            if not hasattr(args, 'num_classes') or args.num_classes is None:
                args.num_classes = 10
            if not hasattr(args, 'way') or args.way is None:
                args.way = 1
            if not hasattr(args, 'shot') or args.shot is None:
                args.shot = 5

        # Calculate sessions based on configuration
        args.sessions = (args.num_classes - args.base_class) // args.way + 1

    args.Dataset = Dataset
    return args


def get_dataloader(args, session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args, session)
    return trainset, trainloader, testloader


def get_base_dataloader(args):
    # txt_path = "data/index_list/" + args.dataset + "/session_" + str(0 + 1) + ".txt"
    class_index = np.arange(args.base_class)
    if args.dataset == "cifar100":
        trainset = args.Dataset.CIFAR100(
            root=args.dataroot,
            train=True,
            download=True,
            index=class_index,
            base_sess=True,
        )
        testset = args.Dataset.CIFAR100(
            root=args.dataroot,
            train=False,
            download=False,
            index=class_index,
            base_sess=True,
        )

    if args.dataset == "cub200":
        trainset = args.Dataset.CUB200(
            root=args.dataroot, train=True, index=class_index, base_sess=True
        )
        testset = args.Dataset.CUB200(
            root=args.dataroot, train=False, index=class_index
        )

    if args.dataset == "mini_imagenet":
        trainset = args.Dataset.MiniImageNet(
            root=args.dataroot, train=True, index=class_index, base_sess=True
        )
        testset = args.Dataset.MiniImageNet(
            root=args.dataroot, train=False, index=class_index
        )

    if args.dataset == "imagenet100" or args.dataset == "imagenet1000":
        trainset = args.Dataset.ImageNet(
            root=args.dataroot, train=True, index=class_index, base_sess=True
        )
        testset = args.Dataset.ImageNet(
            root=args.dataroot, train=False, index=class_index
        )

    if args.dataset == "CICIDS2017_improved":
        trainset = args.Dataset.CICIDS2017_improved(
            root=args.dataroot, train=True, index=class_index, base_sess=True
        )
        testset = args.Dataset.CICIDS2017_improved(
            root=args.dataroot, train=False, index=class_index, base_sess=True
        )

    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size_base,
        shuffle=True,
        num_workers=8,
        pin_memory=args.pin_memory,
    )
    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=args.pin_memory,
    )

    return trainset, trainloader, testloader


def get_base_dataloader_meta(args):
    txt_path = os.path.join(os.path.dirname(args.dataroot), "data/index_list/" + args.dataset + "/session_0.txt")
    class_index = np.arange(args.base_class)
    if args.dataset == "cifar100":
        trainset = args.Dataset.CIFAR100(
            root=args.dataroot,
            train=True,
            download=True,
            index=class_index,
            base_sess=True,
        )
        testset = args.Dataset.CIFAR100(
            root=args.dataroot,
            train=False,
            download=False,
            index=class_index,
            base_sess=True,
        )

    if args.dataset == "cub200":
        trainset = args.Dataset.CUB200(
            root=args.dataroot, train=True, index_path=txt_path
        )
        testset = args.Dataset.CUB200(
            root=args.dataroot, train=False, index=class_index
        )
    if args.dataset == "mini_imagenet":
        trainset = args.Dataset.MiniImageNet(
            root=args.dataroot, train=True, index_path=txt_path
        )
        testset = args.Dataset.MiniImageNet(
            root=args.dataroot, train=False, index=class_index
        )

    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=args.pin_memory)
    sampler = CategoriesSampler(
        trainset.targets,
        args.train_episode,
        args.episode_way,
        args.episode_shot + args.episode_query,
    )

    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    return trainset, trainloader, testloader


def get_new_dataloader(args, session):
    txt_path = os.path.join(os.path.dirname(args.dataroot), "data/index_list/" + args.dataset + "/new_sessions/session_" + str(session) + ".txt")
    if args.dataset == "cifar100":
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(
            root=args.dataroot,
            train=True,
            download=False,
            index=class_index,
            base_sess=False,
        )
    if args.dataset == "cub200":
        trainset = args.Dataset.CUB200(
            root=args.dataroot, train=True, index_path=txt_path
        )
    if args.dataset == "mini_imagenet":
        trainset = args.Dataset.MiniImageNet(
            root=args.dataroot, train=True, index_path=txt_path
        )
    if args.dataset == "imagenet100" or args.dataset == "imagenet1000":
        trainset = args.Dataset.ImageNet(
            root=args.dataroot, train=True, index_path=txt_path
        )
    if args.dataset == "CICIDS2017_improved":
        trainset = args.Dataset.CICIDS2017_improved(
            root=args.dataroot, train=True, index_path=txt_path
        )

    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=batch_size_new,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
    else:
        trainloader = torch.utils.data.DataLoader(
            dataset=trainset,
            batch_size=args.batch_size_new,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == "cifar100":
        testset = args.Dataset.CIFAR100(
            root=args.dataroot,
            train=False,
            download=False,
            index=class_new,
            base_sess=False,
        )
    if args.dataset == "cub200":
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_new)
    if args.dataset == "mini_imagenet":
        testset = args.Dataset.MiniImageNet(
            root=args.dataroot, train=False, index=class_new
        )
    if args.dataset == "imagenet100" or args.dataset == "imagenet1000":
        testset = args.Dataset.ImageNet(
            root=args.dataroot, train=False, index=class_new
        )
    if args.dataset == "CICIDS2017_improved":
        testset = args.Dataset.CICIDS2017_improved(
            root=args.dataroot, train=False, index=class_new, base_sess=False
        )

    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    return trainset, trainloader, testloader


def get_session_classes(args, session):
    class_list = np.arange(args.base_class + session * args.way)
    return class_list

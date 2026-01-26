import os
import numpy as np
import torch
from dataloader.sampler import CategoriesSampler

def set_up_datasets(args):
    # dataset_nameが指定されていない場合、dataset_typeと同じ値を使用
    if not hasattr(args, 'dataset_name') or args.dataset_name is None:
        args.dataset_name = getattr(args, 'dataset_type', args.dataset)
    
    # 後方互換性のため、args.datasetが存在しない場合はargs.dataset_typeを使用
    if not hasattr(args, 'dataset'):
        args.dataset = getattr(args, 'dataset_type', 'cifar100')
    
    if args.dataset == 'cifar100':
        import dataloader.cifar100.cifar as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    if args.dataset =="manyshotcifar":
        import dataloader.cifar100.manyshot_cifar as Dataset
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
        import dataloader.cub200.manyshot_cub as Dataset
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
        import dataloader.miniimagenet.miniimagenet_with_img as Dataset
        args.base_class = 60
        args.num_classes=100
        args.way = 5
        args.shot = 5
        args.sessions = 9
    
    
    if args.dataset == 'manyshotmini':
        import dataloader.miniimagenet.manyshot_mini as Dataset
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

    if args.dataset == 'CICIDS2017_improved':
        # CICIDS2017_improvedデータセットの設定
        import dataloader.cicids2017.cicids2017 as Dataset
        
        # パラメータが設定されていない場合はデフォルト値を使用
        if not hasattr(args, 'base_class') or args.base_class is None:
            args.base_class = 5  # デフォルト値
        if not hasattr(args, 'num_classes') or args.num_classes is None:
            args.num_classes = 10  # デフォルト値
        if not hasattr(args, 'way') or args.way is None:
            args.way = 1  # デフォルト値
        if not hasattr(args, 'shot') or args.shot is None:
            args.shot = 5  # デフォルト値
        # セッション数を計算（base_class + way * sessions = num_classes）
        if not hasattr(args, 'sessions') or args.sessions is None:
            num_new_classes = args.num_classes - args.base_class
            args.sessions = (num_new_classes // args.way) + 1  # +1はベースセッション
        
        print(f"CICIDS2017_improved設定: base_class={args.base_class}, num_classes={args.num_classes}, way={args.way}, shot={args.shot}, sessions={args.sessions}")
        
        # 入力次元数を設定（特徴量の数）
        # データローダーから実際の特徴量数を取得
        if not hasattr(args, 'input_dim') or args.input_dim is None:
            # 一時的にデータローダーを作成して特徴量数を取得
            try:
                # rootは data/dataset_name の形式
                dataset_name = getattr(args, 'dataset_name', args.dataset)
                root = os.path.join(args.dataroot, dataset_name)
                temp_dataset = Dataset.CICIDS2017(
                    root=root,
                    train=True,
                    index=[0],  # ダミーインデックス（存在しないクラスでもエラーにならないように）
                    base_sess=True,
                    label_column=getattr(args, 'label_column', 'Label'),
                    normalize_method=getattr(args, 'normalize_method', 'standard'),
                    window_size=getattr(args, 'window_size', 1000)
                )
                args.input_dim = temp_dataset.data.shape[1]
                print(f"自動検出された特徴量数: {args.input_dim}")
                del temp_dataset
            except Exception as e:
                raise ValueError(f"特徴量数の自動取得に失敗しました: {e}")
                # デフォルト値: 67（column_names.txtから推測）


    # Datasetが定義されていない場合のエラーチェック
    if 'Dataset' not in locals():
        raise ValueError(
            f"データセット '{args.dataset}' の設定が見つかりません。\n"
            f"data_utils.pyのset_up_datasets関数に'{args.dataset}'の処理を追加してください。"
        )
    
    args.Dataset=Dataset
    return args

def get_dataloader(args,session):
    if session == 0:
        trainset, trainloader, testloader = get_base_dataloader(args)
    else:
        trainset, trainloader, testloader = get_new_dataloader(args)
    return trainset, trainloader, testloader

def get_base_dataloader(args):
    # dataset_nameが指定されていない場合、dataset_typeと同じ値を使用
    if not hasattr(args, 'dataset_name') or args.dataset_name is None:
        args.dataset_name = args.dataset_type
    
    txt_path = "data/index_list/" + args.dataset_name + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    print(f"get_base_dataloader: base_class={args.base_class}, class_index={class_index}")
    if args.dataset == 'cifar100':

        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index=class_index, base_sess=True)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = args.Dataset.ImageNet(root=args.dataroot, train=True,
                                             index=class_index, base_sess=True)
        testset = args.Dataset.ImageNet(root=args.dataroot, train=False, index=class_index)

    if args.dataset == 'CICIDS2017_improved':
        # CICIDS2017_improvedデータローダー
        normalize_method = getattr(args, 'normalize_method', 'standard')
        label_column = getattr(args, 'label_column', 'Label')
        window_size = getattr(args, 'window_size', 1000)
        
        # rootは data/dataset_name の形式
        root = os.path.join(args.dataroot, args.dataset_name)
        
        trainset = args.Dataset.CICIDS2017(
            root=root,
            train=True,
            index=class_index,
            base_sess=True,
            label_column=label_column,
            normalize_method=normalize_method,
            window_size=window_size
        )
        testset = args.Dataset.CICIDS2017(
            root=root,
            train=False,
            index=class_index,
            base_sess=True,
            label_column=label_column,
            normalize_method=normalize_method,
            window_size=window_size
        )

    pin_memory = getattr(args, 'pin_memory', args.num_gpu > 0 if hasattr(args, 'num_gpu') else False)
    trainloader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size_base,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )
    testloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory
    )

    return trainset, trainloader, testloader



def get_base_dataloader_meta(args):
    # dataset_nameが指定されていない場合、dataset_typeと同じ値を使用
    if not hasattr(args, 'dataset_name') or args.dataset_name is None:
        args.dataset_name = args.dataset_type
    
    txt_path = "data/index_list/" + args.dataset_name + "/session_" + str(0 + 1) + '.txt'
    class_index = np.arange(args.base_class)
    if args.dataset == 'cifar100':
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=True,
                                         index=class_index, base_sess=True)
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_index, base_sess=True)

    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_index)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                             index_path=txt_path)
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                            index=class_index)


    # DataLoader(test_set, batch_sampler=sampler, num_workers=8, pin_memory=True)
    sampler = CategoriesSampler(trainset.targets, args.train_episode, args.episode_way,
                                args.episode_shot + args.episode_query)

    pin_memory = getattr(args, 'pin_memory', args.num_gpu > 0 if hasattr(args, 'num_gpu') else False)
    trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_sampler=sampler, num_workers=args.num_workers,
                                              pin_memory=pin_memory)

    pin_memory = getattr(args, 'pin_memory', args.num_gpu > 0 if hasattr(args, 'num_gpu') else False)
    testloader = torch.utils.data.DataLoader(
        dataset=testset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin_memory)

    return trainset, trainloader, testloader

def get_new_dataloader(args,session):
    # dataset_nameが指定されていない場合、dataset_typeと同じ値を使用
    if not hasattr(args, 'dataset_name') or args.dataset_name is None:
        args.dataset_name = args.dataset_type
    
    txt_path = "data/index_list/" + args.dataset_name + "/session_" + str(session + 1) + '.txt'
    if args.dataset == 'cifar100':
        class_index = open(txt_path).read().splitlines()
        trainset = args.Dataset.CIFAR100(root=args.dataroot, train=True, download=False,
                                         index=class_index, base_sess=False)
    if args.dataset == 'cub200':
        trainset = args.Dataset.CUB200(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'mini_imagenet':
        trainset = args.Dataset.MiniImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        trainset = args.Dataset.ImageNet(root=args.dataroot, train=True,
                                       index_path=txt_path)

    if args.dataset == 'CICIDS2017_improved':
        # CICIDS2017_improvedデータローダー
        normalize_method = getattr(args, 'normalize_method', 'standard')
        label_column = getattr(args, 'label_column', 'Label')
        window_size = getattr(args, 'window_size', 1000)
        
        # rootは data/dataset_name の形式
        root = os.path.join(args.dataroot, args.dataset_name)
        
        # セッションファイルからデータインデックスを読み込む
        data_indices = [int(line.strip()) for line in open(txt_path).read().splitlines() if line.strip()]
        
        trainset = args.Dataset.CICIDS2017(
            root=root,
            train=True,
            index=data_indices,
            base_sess=False,
            label_column=label_column,
            normalize_method=normalize_method,
            window_size=window_size
        )

    pin_memory = getattr(args, 'pin_memory', args.num_gpu > 0 if hasattr(args, 'num_gpu') else False)
    if args.batch_size_new == 0:
        batch_size_new = trainset.__len__()
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batch_size_new, shuffle=False,
                                                  num_workers=args.num_workers, pin_memory=pin_memory)
    else:
        trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=args.batch_size_new, shuffle=True,
                                                  num_workers=args.num_workers, pin_memory=pin_memory)

    # test on all encountered classes
    class_new = get_session_classes(args, session)

    if args.dataset == 'cifar100':
        testset = args.Dataset.CIFAR100(root=args.dataroot, train=False, download=False,
                                        index=class_new, base_sess=False)
    if args.dataset == 'cub200':
        testset = args.Dataset.CUB200(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'mini_imagenet':
        testset = args.Dataset.MiniImageNet(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'imagenet100' or args.dataset == 'imagenet1000':
        testset = args.Dataset.ImageNet(root=args.dataroot, train=False,
                                      index=class_new)
    if args.dataset == 'CICIDS2017_improved':
        normalize_method = getattr(args, 'normalize_method', 'standard')
        label_column = getattr(args, 'label_column', 'Label')
        window_size = getattr(args, 'window_size', 1000)
        
        # rootは data/dataset_name の形式
        root = os.path.join(args.dataroot, args.dataset_name)
        
        testset = args.Dataset.CICIDS2017(
            root=root,
            train=False,
            index=class_new,
            base_sess=True,  # treat index as class list to include all seen classes
            label_column=label_column,
            normalize_method=normalize_method,
            window_size=window_size
        )

    pin_memory = getattr(args, 'pin_memory', args.num_gpu > 0 if hasattr(args, 'num_gpu') else False)
    testloader = torch.utils.data.DataLoader(dataset=testset, batch_size=args.test_batch_size, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=pin_memory)

    return trainset, trainloader, testloader

def get_session_classes(args,session):
    class_list=np.arange(args.base_class + session * args.way)
    return class_list
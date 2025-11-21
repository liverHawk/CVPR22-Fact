"""
FACT-DQN強化学習用のヘルパー関数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from utils import Averager, count_acc, count_acc_topk, confmatrix
import os


def rl_train_session(
    model,
    dqn_head,
    target_head,
    trainloader,
    replay_buffer,
    optimizer,
    args,
    session,
    reward_calculator,
):
    """
    強化学習でincremental sessionを訓練

    Args:
        model: FACTモデル（エンコーダ固定）
        dqn_head: DQNヘッド（学習対象）
        target_head: ターゲットDQNヘッド
        trainloader: 訓練データローダー
        replay_buffer: Experience Replay Buffer
        optimizer: オプティマイザ
        args: 訓練パラメータ
        session: 現在のセッション番号
        reward_calculator: 報酬計算器
    """
    model.eval()  # エンコーダは固定
    dqn_head.train()

    tl = Averager()  # 損失
    ta = Averager()  # 精度
    tr = Averager()  # 平均報酬

    # 現在のクラス数
    num_current_classes = args.base_class + session * args.way

    global_step = 0
    epsilon_start = getattr(args, 'rl_epsilon_start', 1.0)
    epsilon_end = getattr(args, 'rl_epsilon_end', 0.05)
    epsilon_decay = getattr(args, 'rl_epsilon_decay', 5000.0)
    batch_size = getattr(args, 'rl_batch_size', 64)
    update_interval = getattr(args, 'rl_update_interval', 4)
    target_update_interval = getattr(args, 'rl_target_update', 500)
    gamma = getattr(args, 'rl_gamma', 0.99)

    # データをバッファに収集
    print(f"Collecting experiences for session {session}...")
    device = getattr(args, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    from utils import get_model_module
    model_module = get_model_module(model)
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(trainloader)):
            data, labels = [_.to(device) for _ in batch]

            # エンコーディング（固定）
            embeddings = model_module.encode(data)

            # Epsilon-greedy探索
            epsilon = exponential_decay(epsilon_start, epsilon_end, epsilon_decay, global_step)

            for i in range(len(data)):
                emb = embeddings[i]
                label = labels[i].item()

                # 行動選択
                if np.random.random() < epsilon:
                    action = np.random.randint(0, num_current_classes)
                else:
                    q_values = dqn_head.q_values(emb.unsqueeze(0))[:, :num_current_classes]
                    action = q_values.argmax(dim=1).item()

                # 報酬計算
                logits = dqn_head(emb.unsqueeze(0), include_virtual=False)[:, :num_current_classes]
                prototypes = dqn_head.real_classifier.weight[:num_current_classes]

                reward = reward_calculator.calculate(
                    action=action,
                    target=label,
                    embeddings=emb,
                    prototypes=prototypes,
                    logits=logits.squeeze(0),
                )

                # バッファに追加
                replay_buffer.push(
                    state=emb,
                    action=action,
                    reward=reward,
                    next_state=emb,  # 1-step環境なので同じ
                    done=False,
                )

                tr.add(reward)
                global_step += 1

    # DQN訓練
    print(f"Training DQN for session {session}...")
    num_updates = getattr(args, 'rl_num_updates', 1000)

    for update_step in tqdm(range(num_updates)):
        if len(replay_buffer) < batch_size:
            continue

        # バッチサンプリング
        batch = replay_buffer.sample(batch_size)

        # Q値計算
        q_values = dqn_head(batch.states, include_virtual=False)[:, :num_current_classes]
        q_values = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)

        # ターゲットQ値計算
        with torch.no_grad():
            next_q_values = target_head(batch.next_states, include_virtual=False)[:, :num_current_classes]
            next_q_values = next_q_values.max(1).values
            target_q = batch.rewards + gamma * (1 - batch.dones) * next_q_values

        # TD損失
        loss = F.mse_loss(q_values, target_q)

        # 最適化
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dqn_head.parameters(), 5.0)
        optimizer.step()

        tl.add(loss.item())

        # ターゲットネットワーク更新
        if (update_step + 1) % target_update_interval == 0:
            target_head.load_state_dict(dqn_head.state_dict())

    # 訓練データでの精度計算
    dqn_head.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for batch in trainloader:
            data, labels = [_.to(device) for _ in batch]
            embeddings = model_module.encode(data)
            q_values = dqn_head.q_values(embeddings)[:, :num_current_classes]
            preds = q_values.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        accuracy = correct / total if total > 0 else 0.0
        ta.add(accuracy)

    return tl.item(), ta.item(), tr.item()


def rl_test(model, dqn_head, testloader, args, session, validation=True, wandb_logger=None):
    """
    強化学習モデルのテスト

    Args:
        model: FACTモデル
        dqn_head: DQNヘッド
        testloader: テストデータローダー
        args: パラメータ
        session: セッション番号
        validation: バリデーションモードかどうか
        wandb_logger: Weights & Biases logger
    """
    test_class = args.base_class + session * args.way
    model.eval()
    dqn_head.eval()

    vl = Averager()
    va = Averager()
    va5 = Averager()
    lgt = torch.tensor([])
    lbs = torch.tensor([])

    device = getattr(args, 'device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    from utils import get_model_module
    model_module = get_model_module(model)
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.to(device) for _ in batch]

            # エンコーディング
            embeddings = model_module.encode(data)

            # Q値を使って分類
            q_values = dqn_head.q_values(embeddings)[:, :test_class]

            # 損失と精度
            loss = F.cross_entropy(q_values, test_label)
            acc = count_acc(q_values, test_label)
            top5acc = count_acc_topk(q_values, test_label)

            vl.add(loss.item())
            va.add(acc)
            va5.add(top5acc)
            lgt = torch.cat([lgt, q_values.cpu()])
            lbs = torch.cat([lbs, test_label.cpu()])

    vl = vl.item()
    va = va.item()
    va5 = va5.item()

    print(f'Session {session}, test, loss={vl:.4f} acc={va:.4f}, acc@5={va5:.4f}')

    # Confusion matrix
    lgt = lgt.view(-1, test_class)
    lbs = lbs.view(-1)

    if not validation:
        save_model_dir = os.path.join(args.save_path, f'session{session}_rl_confusion_matrix')

        # ラベル名を取得
        label_names = None
        if args.dataset == 'CICIDS2017_improved' and hasattr(testloader.dataset, 'label_encoder'):
            label_names = list(testloader.dataset.label_encoder.classes_)

        cm = confmatrix(lgt, lbs, save_model_dir, label_names=label_names)
        perclassacc = cm.diagonal()
        seenac = np.mean(perclassacc[:args.base_class])
        unseenac = np.mean(perclassacc[args.base_class:]) if session > 0 else 0.0

        print('Seen Acc:', seenac, 'Unseen ACC:', unseenac)

        # Classification report
        from utils import save_classification_report
        save_classification_report(lgt, lbs, save_model_dir)

        if wandb_logger is not None:
            wandb_logger.log_image(f'session_{session}_rl_confusion_matrix', save_model_dir + '.png')

    return vl, va


def exponential_decay(start: float, end: float, decay: float, step: int) -> float:
    """Epsilon-greedyのための指数減衰"""
    import math
    return end + (start - end) * math.exp(-1.0 * step / decay)

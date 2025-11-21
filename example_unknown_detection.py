"""
埋め込み空間上での距離ベース未知クラス検出の使用例（拡張可能な設計）
"""
import torch
import torch.nn.functional as F
from models.fact.Network import (
    MYNET, DistanceMetric, CosineDistance, EuclideanDistance,
    NormalizedEuclideanDistance, ManhattanDistance, ChebyshevDistance,
    DISTANCE_METRICS
)
from utils import *

def example_unknown_detection():
    """未知クラス検出の使用例"""
    
    # モデルとデータの準備（実際のargsを使用）
    # args = ... (実際の引数オブジェクト)
    # model = MYNET(args, mode='ft_cos')
    # model = model.cuda()
    
    # 例: バッチデータで未知クラス検出
    # data = ...  # (batch_size, ...)
    
    # ===== 方法1: 基本的な未知クラス検出 =====
    # is_unknown, distances, nearest_class = model.detect_unknown_by_distance(
    #     data,
    #     known_class_indices=None,  # Noneの場合はbase_classまで
    #     distance_threshold=None,   # Noneの場合は自動計算
    #     distance_type='cosine'     # 文字列で指定
    # )
    
    # ===== 方法2: 分類と未知クラス検出を同時に実行 =====
    # logits, is_unknown, distances, nearest_class = model.forward(
    #     data,
    #     enable_unknown_detection=True,
    #     known_class_indices=None,
    #     distance_threshold=0.5,  # 手動で閾値を設定
    #     distance_type='cosine'
    # )
    
    # ===== 方法3: 複数の距離メトリクスを試す =====
    # 利用可能な距離メトリクス:
    # - 'cosine': コサイン距離（推奨、FACTモデルに最適）
    # - 'euclidean': ユークリッド距離
    # - 'euclidean_normalized': 正規化ユークリッド距離
    # - 'manhattan': マンハッタン距離（L1距離）
    # - 'chebyshev': チェビシェフ距離（L∞距離）
    # - 'mahalanobis': マハラノビス距離（簡易版）
    
    # for dist_type in ['cosine', 'euclidean', 'manhattan']:
    #     is_unknown, distances, nearest_class = model.detect_unknown_by_distance(
    #         data, distance_type=dist_type
    #     )
    #     print(f"{dist_type}: {is_unknown.sum().item()} unknown samples detected")
    
    # ===== 方法4: カスタム距離メトリクスを使用 =====
    # カスタム距離メトリクスを実装する場合:
    # class CustomDistance(DistanceMetric):
    #     @staticmethod
    #     def compute_distance(embeddings, prototypes):
    #         # カスタム距離計算を実装
    #         return custom_distances
    #     
    #     @staticmethod
    #     def compute_inter_class_distance(prototypes):
    #         # クラス間距離計算を実装
    #         return inter_distances
    # 
    # is_unknown, distances, nearest_class = model.detect_unknown_by_distance(
    #     data, distance_metric=CustomDistance()
    # )
    
    # ===== 方法5: forwardメソッドで統合使用 =====
    # 分類と未知クラス検出を1回のフォワードパスで実行
    # result = model(data, enable_unknown_detection=True, distance_type='cosine')
    # logits, is_unknown, distances, nearest_class = result
    # 
    # # 分類結果を使用
    # pred_classes = logits.argmax(dim=1)
    # 
    # # 未知クラスの場合は特別な処理
    # for i in range(len(is_unknown)):
    #     if is_unknown[i]:
    #         print(f"Sample {i}: Unknown class detected (distance: {distances[i]:.4f}, nearest: {nearest_class[i]})")
    #     else:
    #         print(f"Sample {i}: Known class {pred_classes[i].item()} (distance: {distances[i]:.4f})")
    
    pass


def test_with_unknown_detection(model, testloader, args, session, distance_threshold=None, distance_type='cosine'):
    """
    未知クラス検出を含むテスト関数の例
    
    Args:
        model: 学習済みモデル
        testloader: テストデータローダー
        args: 引数オブジェクト
        session: 現在のセッション
        distance_threshold: 距離の閾値（Noneの場合は自動計算）
        distance_type: 距離の種類
    """
    model = model.eval()
    
    # 既知クラスのインデックスを決定
    known_class_indices = list(range(args.base_class + session * args.way))
    
    total_samples = 0
    unknown_detected = 0
    correct_classified = 0
    unknown_correctly_detected = 0  # 実際に未知クラスで、正しく検出された数
    
    all_distances = []
    all_is_unknown = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(testloader, 1):
            data, test_label = [_.cuda() for _ in batch]
            
            # 未知クラス検出
            is_unknown, distances, nearest_class = model.module.detect_unknown_by_distance(
                data,
                known_class_indices=known_class_indices,
                distance_threshold=distance_threshold,
                distance_type=distance_type
            )
            
            # 通常の分類
            logits = model(data)
            logits = logits[:, :args.base_class + session * args.way]
            pred = torch.argmax(logits, dim=1)
            
            # 統計を更新
            batch_size = data.size(0)
            total_samples += batch_size
            unknown_detected += is_unknown.sum().item()
            
            # 既知クラスとして分類された場合の精度
            known_mask = ~is_unknown
            if known_mask.sum() > 0:
                correct_classified += (pred[known_mask] == test_label[known_mask]).sum().item()
            
            # 実際に未知クラス（既知クラス範囲外）のサンプルを正しく検出
            actual_unknown_mask = test_label >= (args.base_class + session * args.way)
            unknown_correctly_detected += (is_unknown & actual_unknown_mask).sum().item()
            
            # データを保存
            all_distances.extend(distances.cpu().tolist())
            all_is_unknown.extend(is_unknown.cpu().tolist())
            all_labels.extend(test_label.cpu().tolist())
    
    # 結果を表示
    print(f"\n=== Unknown Detection Results (Session {session}) ===")
    print(f"Total samples: {total_samples}")
    print(f"Unknown detected: {unknown_detected} ({100*unknown_detected/total_samples:.2f}%)")
    print(f"Known class accuracy: {100*correct_classified/(total_samples-unknown_detected):.2f}%" if unknown_detected < total_samples else "N/A")
    print(f"Unknown detection accuracy: {100*unknown_correctly_detected/max(1, sum(1 for l in all_labels if l >= args.base_class + session * args.way)):.2f}%")
    print(f"Average distance: {sum(all_distances)/len(all_distances):.4f}")
    
    return {
        'total_samples': total_samples,
        'unknown_detected': unknown_detected,
        'known_accuracy': correct_classified / max(1, total_samples - unknown_detected),
        'unknown_detection_accuracy': unknown_correctly_detected / max(1, sum(1 for l in all_labels if l >= args.base_class + session * args.way)),
        'avg_distance': sum(all_distances) / len(all_distances)
    }


if __name__ == '__main__':
    print("Unknown detection example script")
    print("This script demonstrates how to use distance-based unknown class detection")
    print("in the embedding space within the neural network.")
    print("\nKey features:")
    print("1. detect_unknown_by_distance(): Detects unknown classes based on distance")
    print("2. forward(enable_unknown_detection=True): Classification + unknown detection in one pass")
    print("3. Extensible distance metrics:")
    print("   - cosine: Cosine distance (recommended for FACT)")
    print("   - euclidean: Euclidean distance")
    print("   - euclidean_normalized: Normalized Euclidean distance")
    print("   - manhattan: Manhattan distance (L1)")
    print("   - chebyshev: Chebyshev distance (L∞)")
    print("   - mahalanobis: Mahalanobis distance (simplified)")
    print("4. Automatic threshold calculation based on inter-class distances")
    print("5. Custom distance metrics can be easily added by extending DistanceMetric class")
    print("\n" + "="*70)
    print("Usage with train.py:")
    print("="*70)
    print("\n1. Basic usage with unknown detection enabled:")
    print("   python train.py --enable_unknown_detection --distance_type cosine")
    print("\n2. With custom threshold:")
    print("   python train.py --enable_unknown_detection --distance_type euclidean --distance_threshold 0.5")
    print("\n3. With all options (example):")
    print("   python train.py \\")
    print("     -project fact \\")
    print("     -dataset CICIDS2017_improved \\")
    print("     -encoder mlp \\")
    print("     --enable_unknown_detection \\")
    print("     --distance_type cosine \\")
    print("     --distance_threshold 0.6")
    print("\n4. Available distance types:")
    print("   --distance_type cosine (default, recommended)")
    print("   --distance_type euclidean")
    print("   --distance_type euclidean_normalized")
    print("   --distance_type manhattan")
    print("   --distance_type chebyshev")
    print("   --distance_type mahalanobis")
    print("\nNote: When --enable_unknown_detection is enabled, the test function will")
    print("      automatically detect unknown classes and display statistics during training.")


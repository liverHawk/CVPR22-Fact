import yaml
import subprocess
import itertools
from pathlib import Path
import time
# from datetime import datetime


class GridExperimentRunner:
    def __init__(self, config_path, command, files_to_commit, git_enabled=True):
        """
        Args:
            config_path: YAMLファイルのパス
            command: 実行するコマンド（文字列またはリスト）
            files_to_commit: コミットするファイルのリスト
            git_enabled: Gitコミットを有効にするかどうか
        """
        self.config_path = Path(config_path)
        self.command = command
        self.files_to_commit = files_to_commit
        self.git_enabled = git_enabled

    def load_yaml(self):
        """YAMLファイルを読み込む"""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def save_yaml(self, config):
        """YAMLファイルに保存"""
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    def update_yaml(self, updates):
        """YAMLファイルの値を更新"""
        config = self.load_yaml()

        for key, value in updates.items():
            keys = key.split(".")
            temp = config
            for k in keys[:-1]:
                temp = temp[k]
            temp[keys[-1]] = value

        self.save_yaml(config)

    def run_command(self):
        """コマンドを実行"""
        print(f"▶ コマンド実行: {self.command}")
        try:
            if isinstance(self.command, str):
                # リアルタイムで出力を表示
                process = subprocess.Popen(
                    self.command,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # 行バッファリング
                    universal_newlines=True
                )
            else:
                process = subprocess.Popen(
                    self.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
            
            # リアルタイムで出力を表示
            output_lines = []
            for line in process.stdout:
                line = line.rstrip()
                print(line, flush=True)  # 即座に表示
                output_lines.append(line)
            
            # プロセスの終了を待つ
            return_code = process.wait()
            
            if return_code == 0:
                print("✓ コマンドが正常に終了しました")
                return True
            else:
                print(f"✗ コマンドがエラーコード {return_code} で終了しました")
                return False

        except subprocess.CalledProcessError as e:
            print(f"✗ コマンド実行エラー: {e}")
            if e.stderr:
                print(f"エラー出力: {e.stderr}")
            return False

    def git_commit(self, commit_message):
        """指定されたファイルをGitコミット"""
        if not self.git_enabled:
            print("⊘ Gitコミットは無効です")
            return True

        try:
            for file in self.files_to_commit:
                if Path(file).exists():
                    subprocess.run(["git", "add", file], check=True)
                else:
                    print(f"⚠ ファイルが見つかりません: {file}")

            subprocess.run(
                ["git", "commit", "-m", commit_message],
                check=True,
                capture_output=True,
                text=True,
            )
            print(f"✓ コミット完了: {commit_message}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"⚠ Gitコミットエラー（変更なしの可能性）: {e}")
            return True

    def generate_experiments_from_grid(self, param_grid, experiment_name_template=None):
        """
        パラメータグリッドから全組み合わせの実験を生成

        Args:
            param_grid: パラメータの辞書（各値は配列）
                例: {
                    'model.learning_rate': [0.001, 0.01, 0.1],
                    'model.batch_size': [16, 32, 64],
                    'data.augmentation': [True, False]
                }
            experiment_name_template: 実験名のテンプレート（Noneの場合は自動生成）

        Returns:
            experiments: 実験設定のリスト
        """
        # パラメータ名と値のリストを取得
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        # 全組み合わせを生成
        experiments = []
        for idx, combination in enumerate(itertools.product(*param_values), 1):
            # 現在の組み合わせをディクショナリに変換
            updates = dict(zip(param_names, combination))

            # 実験名を生成
            if experiment_name_template:
                exp_name = experiment_name_template.format(idx=idx, **updates)
            else:
                # デフォルトの実験名（パラメータ値を含む）
                param_str = "_".join(
                    [f"{k.split('.')[-1]}={v}" for k, v in updates.items()]
                )
                exp_name = f"exp{idx:03d}_{param_str}"

            # コミットメッセージを生成
            param_desc = ", ".join([f"{k}={v}" for k, v in updates.items()])
            commit_message = f"Experiment {idx}: {param_desc}"

            experiments.append(
                {"name": exp_name, "updates": updates, "commit_message": commit_message}
            )

        return experiments

    def run_grid_experiments(
        self,
        param_grid,
        experiment_name_template=None,
        stop_on_error=True,
        dry_run=False,
    ):
        """
        グリッドサーチ実験を実行

        Args:
            param_grid: パラメータグリッド
            experiment_name_template: 実験名のテンプレート
            stop_on_error: エラー時に停止するか
            dry_run: 実際には実行せず、実験リストのみ表示
        """
        # 実験を生成
        experiments = self.generate_experiments_from_grid(
            param_grid, experiment_name_template
        )
        total = len(experiments)

        print(f"\n{'=' * 80}")
        print(f"グリッドサーチ実験: {total}個の実験を実行します")
        print(f"{'=' * 80}\n")

        if dry_run:
            print("【ドライラン】以下の実験が実行されます:\n")
            for idx, exp in enumerate(experiments, 1):
                print(f"{idx}. {exp['name']}")
                print(f"   パラメータ: {exp['updates']}")
                print()
            return

        # 実験を実行
        successful = 0
        failed = 0

        for idx, exp in enumerate(experiments, 1):
            print(f"\n{'=' * 80}")
            print(f"実験 {idx}/{total}: {exp['name']}")
            print(f"パラメータ: {exp['updates']}")
            print(f"{'=' * 80}")

            # 1. YAMLファイルを更新
            self.update_yaml(exp["updates"])
            print("✓ YAMLファイルを更新しました")

            # 2. コマンド実行
            start_time = time.time()
            success = self.run_command()
            end_time = time.time()
            duration = end_time - start_time
            print(f"about {duration * (total - idx)} seconds left")
            

            if not success:
                failed += 1
                print(f"\n✗ 実験 {exp['name']} が失敗しました")
                if stop_on_error:
                    print("エラーにより中断します。")
                    break
                else:
                    print("次の実験に進みます。")
                    continue

            # 3. ファイルをコミット
            self.git_commit(exp["commit_message"])

            successful += 1
            print(f"\n✓ 実験 {exp['name']} が完了しました")

        # サマリー
        print(f"\n{'=' * 80}")
        print("実験完了サマリー")
        print(f"{'=' * 80}")
        print(f"総実験数: {total}")
        print(f"成功: {successful}")
        print(f"失敗: {failed}")
        print(f"{'=' * 80}\n")


def main(params_grids):
    runner = GridExperimentRunner(
        config_path="params.yaml",
        command="dvc repro",
        files_to_commit=[
            "dvc.lock",
            "params.yaml"
        ],
        git_enabled=True,
    )

    # 実際に実行
    for grid in params_grids:
        runner.run_grid_experiments(
            grid,
            stop_on_error=False,  # エラーが起きても続行
        )

# 使用例1: 基本的なグリッドサーチ
if __name__ == "__main__":
    input_grids = [
        {
            "create_sessions.shot": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        },
        {
            "create_sessions.shot": [ 5 ],
            "train.epochs_new": [1, 10, 100, 1000, 10000],
        },
        {
            "train.epochs_base": [ 1000 ],
            "train.temperature": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
        }
    ]

    main(input_grids)

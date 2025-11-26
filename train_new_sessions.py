from train import get_command_line_parser, initialize_trainer
import comet_ml


def _prepare_new_session_list(args):
    """
    0（ベース）を除いたセッション番号の配列を返す
    """
    if args.select_sessions:
        selected = sorted({sess for sess in args.select_sessions if sess != 0})
        if not selected:
            raise ValueError("新規セッションを1つ以上指定してください（0は除外）")
        return selected
    # args.sessions はトレーナー初期化後に決定するため、この関数では決定しない
    return None


def main():
    parser = get_command_line_parser()
    args = parser.parse_args()
    config = comet_ml.ExperimentConfig(
        tags=[f"new_sessions_{'-'.join(map(str, args.select_sessions))}"],
    )
    args.comet = comet_ml.start(project_name="NIDS on FACT", experiment_config=config)

    pending_sessions = _prepare_new_session_list(args)
    trainer = initialize_trainer(args)

    total_sessions = trainer.args.sessions
    if pending_sessions is None:
        pending_sessions = list(range(1, total_sessions))
    else:
        invalid = [sess for sess in pending_sessions if sess >= total_sessions]
        if invalid:
            raise ValueError(
                f"指定されたセッション {invalid} は最大 {total_sessions - 1} を超えています"
            )

    trainer.args.select_sessions = pending_sessions
    trainer.train()


if __name__ == "__main__":
    main()


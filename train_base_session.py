from train import get_command_line_parser, initialize_trainer
import comet_ml


def main():
    parser = get_command_line_parser()
    args = parser.parse_args()
    config = comet_ml.ExperimentConfig(
        tags=["base_session"],
    )
    args.comet = comet_ml.start(project_name="NIDS on FACT", experiment_config=config)

    # ベースセッションのみを強制的に実行
    args.select_sessions = [0]
    trainer = initialize_trainer(args)
    trainer.args.select_sessions = [0]
    trainer.train()


if __name__ == "__main__":
    main()


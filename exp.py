import marimo

__generated_with = "0.19.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import polars as pl
    return mo, pl


@app.cell
def _():

    DATABASE_URL = "experiments.db"

    import peewee

    db = peewee.SqliteDatabase(DATABASE_URL)
    return db, peewee


@app.cell
def _(db, peewee):
    from peewee import (
        BooleanField,
        DateTimeField,
        FloatField,
        ForeignKeyField,
        IntegerField,
        Model,
        TextField,
    )
    import json
    from datetime import datetime

    class BaseModel(peewee.Model):
        class Meta:
            database = db


    class Experiment(BaseModel):
        """One row per training run. Every train.py argument is stored as its own column."""

        created_at = DateTimeField(default=datetime.now)
        save_path = TextField()

        # Basic
        project = TextField()
        dataset_type = TextField()
        dataset_name = TextField()
        dataroot = TextField()
        encoder = TextField()

        # Training
        epochs_base = IntegerField()
        epochs_new = IntegerField()
        lr_base = FloatField()
        lr_new = FloatField()
        batch_size_base = IntegerField()
        batch_size_new = IntegerField()
        test_batch_size = IntegerField()

        # Mode
        base_mode = TextField()
        new_mode = TextField()

        # Schedule
        schedule = TextField()
        step = IntegerField()
        decay = FloatField()
        gamma = FloatField()
        momentum = FloatField()
        temperature = FloatField()
        milestones_json = TextField(null=True)  # JSON list e.g. "[50,100,150]"

        # FACT
        balance = FloatField()
        alpha = FloatField()
        eta = FloatField()
        loss_iter = IntegerField()

        # Other
        start_session = IntegerField()
        model_dir = TextField(null=True)
        num_workers = IntegerField()
        gpu = TextField()
        seed = IntegerField()

        # Flags
        not_data_init = BooleanField()
        set_no_val = BooleanField()
        debug = BooleanField()

        # CICIDS / FSCIL
        normalize_method = TextField()
        label_column = TextField()
        base_class = IntegerField(null=True)
        num_classes = IntegerField(null=True)
        way = IntegerField(null=True)
        shot = IntegerField()
        base_labels_json = TextField(null=True)  # JSON list

        @classmethod
        def from_args(cls, args):
            """Build Experiment from argparse args (or namespace with same attributes)."""
            milestones_json = json.dumps(getattr(args, "milestones", [])) if getattr(args, "milestones", None) else None
            base_labels = getattr(args, "base_labels", None)
            base_labels_json = json.dumps(base_labels) if base_labels is not None else None
            return cls.create(
                save_path=getattr(args, "save_path", "") or "",
                project=getattr(args, "project", "fact"),
                dataset_type=getattr(args, "dataset_type", ""),
                dataset_name=getattr(args, "dataset_name", "") or "",
                dataroot=getattr(args, "dataroot", "data/"),
                encoder=getattr(args, "encoder", "mlp"),
                epochs_base=getattr(args, "epochs_base", 0),
                epochs_new=getattr(args, "epochs_new", 0),
                lr_base=getattr(args, "lr_base", 0.0),
                lr_new=getattr(args, "lr_new", 0.0),
                batch_size_base=getattr(args, "batch_size_base", 0),
                batch_size_new=getattr(args, "batch_size_new", 0),
                test_batch_size=getattr(args, "test_batch_size", 0),
                base_mode=getattr(args, "base_mode", ""),
                new_mode=getattr(args, "new_mode", ""),
                schedule=getattr(args, "schedule", ""),
                step=getattr(args, "step", 0),
                decay=getattr(args, "decay", 0.0),
                gamma=getattr(args, "gamma", 0.0),
                momentum=getattr(args, "momentum", 0.0),
                temperature=getattr(args, "temperature", 0.0),
                milestones_json=milestones_json,
                balance=getattr(args, "balance", 0.0),
                alpha=getattr(args, "alpha", 0.0),
                eta=getattr(args, "eta", 0.0),
                loss_iter=getattr(args, "loss_iter", 0),
                start_session=getattr(args, "start_session", 0),
                model_dir=getattr(args, "model_dir", None),
                num_workers=getattr(args, "num_workers", 0),
                gpu=getattr(args, "gpu", ""),
                seed=getattr(args, "seed", 0),
                not_data_init=getattr(args, "not_data_init", False),
                set_no_val=getattr(args, "set_no_val", False),
                debug=getattr(args, "debug", False),
                normalize_method=getattr(args, "normalize_method", ""),
                label_column=getattr(args, "label_column", ""),
                base_class=getattr(args, "base_class", None),
                num_classes=getattr(args, "num_classes", None),
                way=getattr(args, "way", None),
                shot=getattr(args, "shot", 5),
                base_labels_json=base_labels_json,
            )


    class EpochMetric(BaseModel):
        """One row per epoch (base session only)."""

        experiment = ForeignKeyField(Experiment, backref="epoch_metrics")
        session = IntegerField()
        epoch = IntegerField()
        train_loss = FloatField()
        train_acc = FloatField()
        test_loss = FloatField()
        test_acc = FloatField()
        learning_rate = FloatField()


    class SessionMetric(BaseModel):
        """One row per session (final evaluation: max_acc, F1, etc.)."""

        experiment = ForeignKeyField(Experiment, backref="session_metrics")
        session = IntegerField()
        max_acc = FloatField()
        test_loss = FloatField()
        accuracy = FloatField()
        precision_macro = FloatField()
        recall_macro = FloatField()
        f1_macro = FloatField()
        seen_acc = FloatField()
        unseen_acc = FloatField()

    return EpochMetric, Experiment, SessionMetric


@app.cell
def _(Experiment, mo, pl):
    all_exp = Experiment.select().dicts()
    df_exp = pl.DataFrame(list(all_exp))

    mo.ui.dataframe(df_exp)
    return


@app.cell
def _(EpochMetric, mo, pl):
    all_epoch = EpochMetric.select().dicts()
    df_epoch = pl.DataFrame(list(all_epoch))

    mo.ui.dataframe(df_epoch)
    return


@app.cell
def _(SessionMetric, mo, pl):
    all_session = SessionMetric.select().dicts()
    df_session = pl.DataFrame(list(all_session))

    mo.ui.dataframe(df_session)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()

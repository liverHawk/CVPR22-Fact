"""Database connection and initialization."""

import os
from peewee import SqliteDatabase

# Default: single DB at project root so all experiments are queryable in one place
_DEFAULT_DB_NAME = "experiments.db"


def get_db_path(save_path=None):
    """Return path to SQLite DB file. If save_path is given, can be used for per-dataset DB."""
    if save_path:
        return os.path.join(save_path, _DEFAULT_DB_NAME)
    # Project root: go up from db/ to project root
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(_root, _DEFAULT_DB_NAME)


db = SqliteDatabase(None)


def init_db(save_path=None):
    """Initialize DB connection and create tables. Uses project root DB by default."""
    path = get_db_path(save_path)
    db.init(path)
    from db import models  # noqa: F401

    db.create_tables([models.Experiment, models.EpochMetric, models.SessionMetric])
    return db

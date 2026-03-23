from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


def init_database(db_path: str | Path) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")

    return path


def write_table(df: pd.DataFrame, db_path: str | Path, table_name: str) -> None:
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False)


def write_json_record(
    db_path: str | Path,
    table_name: str,
    payload: dict[str, object],
) -> None:
    frame = pd.DataFrame(
        [{"payload_json": json.dumps(payload, indent=2, default=str)}]
    )
    write_table(frame, db_path, table_name)


def read_table(db_path: str | Path, query: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)

#!/usr/bin/env python
"""Stage 00: parse downloaded gold datasets (data/datasets/*) and write them to data/fnp.duckdb.

Downloads are done once by hand/curl into fresh dirs (see docs/review/01_DATA_GAPS.md);
this script only parses. Default: print dataset x split x label5 summary. --apply writes.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import duckdb
import pandas as pd

from fnp.config import DUCKDB_PATH, ensure_dirs
from fnp.datasets import GOLD_COLS, load_all_gold, load_ptc


def summary(g: pd.DataFrame) -> pd.DataFrame:
    return (g.assign(label5=g.label5.fillna("(none)")).groupby(["dataset", "split", "label5"])
            .size().unstack(fill_value=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    gold = load_all_gold()
    arts, spans = load_ptc()
    pd.set_option("display.width", 200)
    print(summary(gold).to_string())
    print(f"\nPTC: {len(arts)} articles, {len(spans)} spans")
    if not a.apply:
        print("\n(dry run; pass --apply to write)")
        return
    ensure_dirs()
    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("""CREATE TABLE IF NOT EXISTS gold_items_v1(item_id VARCHAR, dataset VARCHAR, split VARCHAR,
        text VARCHAR, claim VARCHAR, lang VARCHAR, label5 VARCHAR, raw_label VARCHAR, speaker VARCHAR,
        date VARCHAR, url VARCHAR, meta_json VARCHAR)""")
    con.execute("CREATE TABLE IF NOT EXISTS ptc_articles_v1(article_id VARCHAR, split VARCHAR, text VARCHAR)")
    con.execute('CREATE TABLE IF NOT EXISTS ptc_spans_v1(article_id VARCHAR, split VARCHAR, start BIGINT, "end" BIGINT, technique VARCHAR)')
    for ds, df in gold.groupby("dataset"):
        n = con.execute("SELECT count(*) FROM gold_items_v1 WHERE dataset=?", [ds]).fetchall()[0][0]
        if n == len(df):
            print(f"skip {ds}: {n} rows already present")
            continue
        if n:
            print(f"WARN {ds}: {n} existing rows != {len(df)}; v1 is never overwritten, skipping "
                  f"(write a *_v2 table to change)")
            continue
        d = df[GOLD_COLS]
        con.register("_d", d)
        con.execute("INSERT INTO gold_items_v1 SELECT * FROM _d")
        con.unregister("_d")
        print(f"wrote {ds}: {len(df)} rows")
    for tbl, df in (("ptc_articles_v1", arts), ("ptc_spans_v1", spans)):
        n = con.execute(f"SELECT count(*) FROM {tbl}").fetchall()[0][0]
        if n == len(df):
            print(f"skip {tbl}: {n} rows present")
        elif n == 0 and len(df):
            con.register("_t", df)
            con.execute(f"INSERT INTO {tbl} SELECT * FROM _t")
            con.unregister("_t")
            print(f"wrote {tbl}: {len(df)}")
        else:
            print(f"WARN {tbl}: {n} existing != {len(df)}; skipped")
    con.close()


if __name__ == "__main__":
    main()

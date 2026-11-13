r"""ComEM test-anchor list and entity-disjoint dev split. Run in the ComEM checkout.

    cd comem && python build_comem_dev.py <out.json>

Rebuilds `candidates` with the official `blocking()` (saved retriv index) and the official
candidates_K rule, checks that the official test sample (data/llm4em/<ds>.csv) is a subset of it,
and draws up to 200 dev anchors not in test whose 10 right-side records never occur in any test
candidate list. Dev labels are kept for dev selection only.
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
import src.blocking as B  # noqa: E402

out = {}
for dataset, (path, files, sep) in B.Datasets.items():
    dfs = [pd.read_csv(path / f, index_col="id", sep=sep, dtype=str) for f in files]
    for df in dfs:
        df.fillna("", inplace=True)
        df.drop(columns=["description"], inplace=True, errors="ignore")
        df.drop(columns=["http://dbpedia.org/ontology/abstract"], inplace=True, errors="ignore")
    matches = pd.read_csv(path / "gt.csv", sep=sep)
    matches.columns = ["id_left", "id_right"]
    match_set = set(matches.itertuples(index=False, name=None))
    order = {}
    for l, r in B.blocking(dataset, dfs, match_set):          # official order: score-descending per anchor
        order.setdefault(str(l), []).append(str(r))
    ldf, rdf = dfs[0], dfs[-1]
    lrec = ldf.astype(str).agg(" ".join, axis=1).str.strip(); lrec.index = lrec.index.astype(str)
    rrec = rdf.astype(str).agg(" ".join, axis=1).str.strip(); rrec.index = rrec.index.astype(str)
    ms = {(str(a), str(b)) for a, b in match_set}
    rows = [(l, r) for l, rs in order.items() if len(rs) > 10 for r in rs[:10]]   # official candidates_K
    candK = pd.DataFrame(rows, columns=["id_left", "id_right"])
    candK["record_left"] = candK.id_left.map(lrec); candK["record_right"] = candK.id_right.map(rrec)
    candK["label"] = [(a, b) in ms for a, b in zip(candK.id_left, candK.id_right)]
    test = pd.read_csv(Path("data/llm4em") / f"{dataset}.csv", dtype=str)
    t_left = set(test.id_left.astype(str))
    t_right = set(test.id_right.astype(str))
    pairs_k = set(zip(candK.id_left, candK.id_right))
    test_in_k = sum((a, b) in pairs_k for a, b in zip(test.id_left, test.id_right))
    pool = candK[~candK.id_left.isin(t_left)]
    bad = set(pool[pool.id_right.isin(t_right)].id_left)
    ok = sorted(set(pool.id_left) - bad, key=lambda i: hashlib.sha256(f"comem-dev-20261105:{dataset}:{i}".encode()).hexdigest())[:200]
    dev = pool[pool.id_left.isin(ok)]
    out[dataset] = dict(test_anchors=sorted(t_left), test_pairs=len(test), test_pairs_in_candidatesK=test_in_k,
                        dev_pool_anchors=int(pool.id_left.nunique()), dev_excluded_entity_overlap=len(bad),
                        dev=[dict(id_left=a, id_right=g.id_right.tolist(), record_left=g.record_left.iloc[0],
                                  record_right=g.record_right.tolist(), label=g.label.tolist())
                             for a, g in dev.groupby("id_left", sort=False)])
    print(dataset, "test", len(t_left), "test pairs in candK", test_in_k, "/", len(test), "dev pool", out[dataset]["dev_pool_anchors"],
          "excluded", len(bad), "dev", len(out[dataset]["dev"]), "dev pos", int(dev.label.sum()), flush=True)
json.dump(out, open(sys.argv[1], "w"))

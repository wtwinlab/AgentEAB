# BIRD and Spider databases

The BIRD and Spider database files are distributed by their authors and are not in this repository. No
paper-object computation needs them: SQL outcomes are part of the scored inputs in `paper_objects/`.
The collection and scoring code in `experiments/*/code/` reads them when it executes queries.

- **BIRD development databases** (11 SQLite databases, 101 files, 1.39 GiB): the `dev.zip` package of
  the 2024-06-27 development release at `https://bird-bench.github.io/`. Unpack to
  `<dir>/dev_20240627/dev_databases/<db_id>/<db_id>.sqlite` and set
  `BIRD_DB_DIR=<dir>/dev_20240627/dev_databases`.
- **Spider databases** (166 SQLite databases, 391 files, 0.82 GiB): the release at
  `https://yale-lily.github.io/spider`. Unpack to `<dir>/spider_data/database/<db_id>/<db_id>.sqlite`
  and set `SPIDER_DB_DIR=<dir>/spider_data/database`.

`CHECKSUMS.json` records, per corpus, the file count, total size, a digest over the (relative path,
size) listing, and a SHA-256 for every `.sqlite` file. Compare what you obtained with:

```
python3 - <<'PY'
import hashlib, json, os, pathlib
want = json.load(open("data/external/CHECKSUMS.json"))
for lab, env in (("bird_databases", "BIRD_DB_DIR"), ("spider_databases", "SPIDER_DB_DIR")):
    root = os.environ.get(env)
    if not root:
        print("%s: %s not set, skipped" % (lab, env)); continue
    ok = bad = 0
    for rel, meta in want[lab]["sqlite_files"].items():
        p = pathlib.Path(root) / rel
        h = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        ok += h == meta["sha256"]; bad += h != meta["sha256"]
    print("%s: %d databases match, %d do not" % (lab, ok, bad))
PY
```

Queries are executed read-only: each database is opened with `mode=ro` and `PRAGMA query_only=ON`
under a 5-second timeout.

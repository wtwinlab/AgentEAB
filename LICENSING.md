# Licensing

## The authors' material

The code is released under the MIT License (`LICENSE`). The data in this repository and in the linked
Google Drive folder, and the documentation, are released under CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/; `LICENSE-DATA`).

The analysis code (`paper_objects/*/code/`, `paper_objects/check_objects.py`, `verify.py`), the
collection code (`code/`, `experiments/*/code/`), the SHARE-Eval code (`share_eval/`), the derived tables in `paper_objects/*/inputs/` and
`output/`, the supplement in `paper/` and these documents are the authors' own work.

## Third-party content in the inputs

The scored tables carry identifiers, labels and outcomes; a few inputs also carry benchmark text. None
of it is relicensed here; each source keeps its own terms.

| Source | Where it appears | Terms |
|---|---|---|
| BIRD (development set) | question ids, database ids and outcomes throughout the SQL objects; question text, evidence and model replies in the extractor-audit samples of `tab_s-reader-sensitivity/inputs/audit_*/` | CC BY-SA 4.0, as published at `https://bird-bench.github.io/` |
| Spider | question and database ids and outcomes in the SQL pool objects | CC BY-SA 4.0, as published at `https://yale-lily.github.io/spider` |
| GSM8K, StrategyQA, MMLU-Pro, GPQA | task ids and correctness only (no question text) in the QA objects | each benchmark's own terms; GPQA question and option text is not redistributed, here or in the hosted data |
| WDC Products, Geographic Settlements, DBLP–ACM, Amazon–Google | pair ids, reference labels and judgments in the matching objects | WDC Products and the Leipzig benchmark collection (Geographic Settlements CC BY 4.0; DBLP–ACM and Amazon–Google as stated on their dataset pages) |
| MusicBrainz 20K | aggregate grouping results only; its records are obtained from the Leipzig benchmark collection | as stated on its dataset page |

The BIRD and Spider database files are not included (`data/external/README.md`).

## Upstream protocol implementations

Some collection adapters read the released implementations at run time; `code/README.md`
lists each repository and commit to clone into `vendor/`. Their licences, as found in the clones used:

| Path | Repository | Licence file |
|---|---|---|
| `vendor/reconcile` | https://github.com/dinobby/ReConcile | MIT |
| `vendor/consensus_literature/moa` | https://github.com/togethercomputer/MoA | Apache 2.0 |
| `vendor/consensus_literature/mallm` | https://github.com/Multi-Agent-LLMs/mallm | Apache 2.0 |
| `vendor/mad` | https://github.com/composable-models/llm_multiagent_debate | as stated by the upstream repository |
| `vendor/delib` | task-allocation games | MIT |

`code/consensus_delivery/patches/` holds a patch applied to the MALLM clone used by the MALLM adapter; it is distributed with the upstream licence of the file it modifies.

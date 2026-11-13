"""The stage evaluation, comparisons and reports run on their own code.

Only the protocol registry (which loads the collection code of each protocol on request) and the
command line (which runs the scripts in `share_eval/`) use the repository's other implementations.
"""
import ast
import pathlib
import subprocess
import sys

PKG = pathlib.Path(__file__).resolve().parents[1]
ALLOWED = {"_repo.py", "adapters/protocols.py", "cli.py", "__main__.py"}
OTHER_CODE = ("paper_objects", "share_eval/", "experiments/", "code/consensus_delivery")


def _imports_repo(path: pathlib.Path) -> bool:
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.names and any(a.name == "_repo" for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("_repo"):
            return True
        if isinstance(node, ast.Import) and any(a.name.endswith("._repo") for a in node.names):
            return True
    return False


def test_no_module_imports_repo():
    offenders = [str(p.relative_to(PKG)) for p in PKG.rglob("*.py")
                 if "tests" not in p.parts and str(p.relative_to(PKG)) not in ALLOWED and _imports_repo(p)]
    assert not offenders, offenders


SCRIPT = r"""
import sys, importlib, pkgutil
sys.dont_write_bytecode = True
import share_eval_suite
for m in pkgutil.walk_packages(share_eval_suite.__path__, "share_eval_suite."):
    if ".tests" in m.name or m.name.endswith(("cli", "__main__", "_repo")):
        continue
    importlib.import_module(m.name)
from share_eval_suite.comparisons import peers, replacement, interactions, dependency, computation
from share_eval_suite.stages import availability, changes, grouping
peers.peer_effect("protocol", "exact", tie="first", configurations="both")
replacement.replacement_cost("refiner", "qwen3-32b", "bird")
replacement.stage_decomposition("refiner", "qwen3-32b", "bird")
interactions.shared_reference_ledger()
dependency.replay_bound("dpeer_EI", "mean4")
computation.fixed_response_votes()
availability.candidate_flow()
changes.revision_changes()
grouping.sharing_effect_on_groups()
loaded = [getattr(m, "__file__", "") or "" for m in list(sys.modules.values())]
bad = sorted({f for f in loaded if any(s in f for s in OTHER)})
print("\n".join(bad))
"""


def test_runs_without_other_implementations():
    repo = PKG.parent
    code = "OTHER = %r\n" % (OTHER_CODE,) + SCRIPT
    r = subprocess.run([sys.executable, "-B", "-c", code], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip() == "", "loaded other implementations:\n" + r.stdout

r"""Run the existing share_eval CLI (unchanged) on the formal spec with the pilot adapter (copied unchanged).

    python3 code/run_cli.py validate code/chess_formal_spec.json
    python3 code/run_cli.py analyze  code/chess_formal_spec.json
Outputs go to output/cli/, not into the share_eval directory.
"""
import pathlib
import sys

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
SE_CODE = HERE.parents[1] / "../common" / "code"
sys.path.insert(0, str(SE_CODE))
sys.path.insert(0, str(HERE))
import share_eval as SE                                                     # noqa: E402
import chess_b1_adapter as A                                                # noqa: E402

SE.OUT = HERE.parent / "output" / "cli"
SE._adapter = lambda spec: A
sys.exit(SE.main())

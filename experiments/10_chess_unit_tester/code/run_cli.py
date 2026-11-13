import pathlib
import sys

sys.dont_write_bytecode = True
HERE = pathlib.Path(__file__).resolve().parent
SE_CODE = HERE.parents[1] / "../common" / "code"
sys.path.insert(0, str(SE_CODE))
sys.path.insert(0, str(HERE))
import share_eval as SE
import chess_b1_adapter as A

SE.OUT = HERE.parent / "output" / "cli"
SE._adapter = lambda spec: A
sys.exit(SE.main())

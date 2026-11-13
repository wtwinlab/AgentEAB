import sys

sys.dont_write_bytecode = True

from share_eval_suite.cli import main  # noqa: E402

sys.exit(main())

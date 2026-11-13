r"""Interface-only adaptation of official MAC-SQL @31a9df5e. Prompts, parsers, agent
logic and the Refiner trigger rule are unchanged. Every replacement asserts its expected count.

    python3 apply_patch_macsql.py ${EXT_WORKDIR}/macsql
    git -C ${EXT_WORKDIR}/macsql diff > macsql_local.patch

P1 api_config.py  OpenAI-compatible local endpoint instead of Azure; model qwen3-32b.
P2 llm.py         GPT-path call parameters kept (temperature 0.1); `model=` instead of Azure
                  `engine=`; max_tokens 4096 (upstream unset); thinking off.
P3 agents.py      both sqlite3 connections (Selector schema read, Refiner execution) read-only.
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])


def sub(rel, old, new, count=1):
    p = ROOT / rel
    s = p.read_text()
    assert s.count(old) == count, (rel, s.count(old), old[:70])
    p.write_text(s.replace(old, new))


sub("core/api_config.py",
    'openai.api_type = "azure"\nopenai.api_base = OPENAI_API_BASE\n# set your own api_version\nopenai.api_version = "2023-07-01-preview"\n',
    'openai.api_type = "open_ai"  # LOCAL PATCH P1: local OpenAI-compatible server\nopenai.api_base = OPENAI_API_BASE\nopenai.api_version = None\n')
sub("core/api_config.py",
    "MODEL_NAME = 'gpt-4-1106-preview' # 128k 版本\n",
    "MODEL_NAME = os.getenv('MACSQL_MODEL', 'qwen3-32b')  # LOCAL PATCH P1\n")
sub("core/llm.py",
    "        response = openai.ChatCompletion.create(\n            engine=MODEL_NAME,\n            messages=[{\"role\": \"user\", \"content\": prompt}],\n            temperature=0.1\n        )\n",
    "        response = openai.ChatCompletion.create(  # LOCAL PATCH P2\n            model=MODEL_NAME,\n            messages=[{\"role\": \"user\", \"content\": prompt}],\n            temperature=0.1,\n            max_tokens=4096,\n            chat_template_kwargs={\"enable_thinking\": False}\n        )\n")
sub("core/agents.py",
    "        conn = sqlite3.connect(db_path)\n",
    "        conn = sqlite3.connect(f\"file:{db_path}?mode=ro\", uri=True)  # LOCAL PATCH P3: read-only\n",
    count=2)
print("MAC-SQL patched")

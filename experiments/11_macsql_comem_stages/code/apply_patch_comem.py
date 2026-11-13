import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1])


def sub(rel, old, new):
    p = ROOT / rel
    s = p.read_text()
    assert s.count(old) == 1, (rel, s.count(old), old[:70])
    p.write_text(s.replace(old, new))


sub("src/utils.py",
    "    response = client.chat.completions.create(messages=messages, model=model, **kwargs)\n",
    "    kwargs.setdefault(\"extra_body\", {\"chat_template_kwargs\": {\"enable_thinking\": False}})  # LOCAL PATCH C1\n"
    "    response = client.chat.completions.create(messages=messages, model=model, **kwargs)\n")
sub("src/utils.py",
    "        if model_name not in self._model_cost_per_1m_tokens:\n            raise ValueError(f\"Unknown model name: {model_name}\")\n",
    "        if model_name not in self._model_cost_per_1m_tokens:  # LOCAL PATCH C2: local model, no money\n"
    "            self._model_cost_per_1m_tokens = {**self._model_cost_per_1m_tokens, model_name: {\"prompt\": 0, \"completion\": 0}}\n")
print("ComEM patched")

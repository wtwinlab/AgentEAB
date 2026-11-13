r"""Interface-only adaptation of official ComEM @5b43c5b8. Templates, ranking,
selection, parsing and metrics are unchanged. Every replacement asserts its expected count.

    python3 apply_patch_comem.py ${EXT_WORKDIR}/comem
    git -C ${EXT_WORKDIR}/comem diff > comem_local.patch

C1 utils.py openai_chat_complete: thinking off for the local Qwen3 selector (request body only;
            the diskcache key is unchanged).
C2 utils.py APICostCalculator: an unknown (local) model costs 0 money instead of raising.
"""
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

r"""Smoke gate: exit 0 only if every cell has >=1 run, 0 peer-text calls, 0 markers, 0 placeholders."""
import json, subprocess, sys
out = json.loads(subprocess.run([sys.executable, __file__.replace("cn_gate.py", "cn_peer_text_check.py"), sys.argv[1], sys.argv[2]],
                                capture_output=True, text=True).stdout)
bad = [k for k, v in out.items() if v.get("peer_text_calls") or v.get("marker_calls") or v.get("placeholder_calls") or not v.get("runs")]
print("GATE", "PASS" if out and not bad else "FAIL", json.dumps(out))
sys.exit(0 if out and not bad else 1)

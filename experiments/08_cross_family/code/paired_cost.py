"""Price task-weighted paired token differences using the manuscript deployment scenarios. No calls."""
from pathlib import Path
from collections import defaultdict
from statistics import mean
import json,csv
S=Path(__file__).resolve().parents[1]; O=S/'output'
inputs={r['unit']:r for r in map(json.loads,(S/'data/qa_replay_inputs.jsonl').open())}
rows=[]
for model,prices in [('qwen3-32b',(0.08,0.28)),('gpt-oss-120b',(0.15,0.60)),('gemma-4-26b',None)]:
 cells=defaultdict(dict)
 for r in map(json.loads,(S/f'output/raw/qa_{model}.jsonl').open()):
  if r.get('status')!='ok': continue
  u=inputs[r['unit_id']]; key=(u['panel'],u['prefix_id'],u['agent'],u['decode'])
  cells[key][u['condition']]=(u['task'],r['response']['usage'])
 for a,b in [('answer_only','full'),('full','no_peers'),('answer_only','no_peers')]:
  tasks=defaultdict(list)
  for v in cells.values():
   if a in v and b in v:
    task,ua=v[a]; _,ub=v[b]
    tasks[task].append((ua['prompt_tokens']-ub['prompt_tokens'],ua['completion_tokens']-ub['completion_tokens']))
  ti=mean(mean(x[0] for x in v) for v in tasks.values());to=mean(mean(x[1] for x in v) for v in tasks.values())
  rows.append(dict(model=model,contrast=a+' - '+b,tasks=len(tasks),paired_cells=sum(map(len,tasks.values())),delta_input=ti,delta_output=to,usd_per_1000=None if prices is None else (ti*prices[0]+to*prices[1])/1000,output_input_break_even=None if to==0 else -ti/to))
(O/'paired_costs.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))

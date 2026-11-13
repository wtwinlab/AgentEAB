"""Aggregate saved input/output usage for development evaluation and held-out deployment. Published rates define pricing scenarios; labels and CPU time are not priced."""
import json,collections
from pathlib import Path
R=Path(__file__).resolve().parents[1]
S=collections.defaultdict(lambda:dict(calls=0,input_tokens=0,output_tokens=0))
for p in sorted((R/'output/raw').glob('e2_*.jsonl')):
 if 'devprobe' in p.name:continue
 for l in p.open():
  d=json.loads(l);assert d['status']=='ok'
  parts=d['unit_id'].split('|'); key=(d['model'],parts[1],parts[-1]);u=d['response']['usage'];v=S[key]
  v['calls']+=1;v['input_tokens']+=u['prompt_tokens'];v['output_tokens']+=u['completion_tokens']
rows=[dict(model=k[0],split=k[1],condition=k[2],**v) for k,v in sorted(S.items())]
prices={'qwen3-32b':(.08,.28),'gpt-oss-120b':(.15,.60)}
res=[]
for m in ['qwen3-32b','gpt-oss-120b','gemma-4-26b']:
 dev=[v for k,v in S.items() if k[0]==m and k[1]=='dev'];n=S[m,'test','no_peer'];f=S[m,'test','full_explanation'];assert n['calls']==f['calls']==1000
 z=dict(model=m,development_recipient_calls=sum(x['calls'] for x in dev),development_recipient_input=sum(x['input_tokens'] for x in dev),development_recipient_output=sum(x['output_tokens'] for x in dev),development_peer=S['qwen2.5-7b','dev','peer'],test_peer=S['qwen2.5-7b','test','peer'],no_peer=n,full_explanation=f,delta_no_peer_minus_full_recipient_input=n['input_tokens']-f['input_tokens'],delta_no_peer_minus_full_recipient_output=n['output_tokens']-f['output_tokens'])
 if m in prices:
  pi,po=prices[m];z['recipient_pricing_scenario']=dict(input_per_million=pi,output_per_million=po,dev_menu_usd=(z['development_recipient_input']*pi+z['development_recipient_output']*po)/1e6,no_peer_test_usd=(n['input_tokens']*pi+n['output_tokens']*po)/1e6,full_test_usd=(f['input_tokens']*pi+f['output_tokens']*po)/1e6)
 res.append(z)
out=dict(scope='All 500 development and 1000 test pairs; four development inputs per recipient. Development peer calls shared across the three evaluated families, not repeated per family. Pilot excluded from policy development cost. No-peer deployment has no peer call. CPU and gold-label acquisition not priced.',rows=rows,comparisons=res)
(R/'output/paper_e2_cost.json').write_text(json.dumps(out,indent=2)+'\n')
for z in res:print(json.dumps(z))

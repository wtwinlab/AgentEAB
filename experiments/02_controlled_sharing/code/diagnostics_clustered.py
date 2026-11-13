#!/usr/bin/env python3
"""Offline diagnostics with explicit readout semantics and task-clustered intervals. Raw
records and historical synthesis outputs are not overwritten. No inference/DB calls.
"""
import collections, csv, hashlib, json, pathlib, sys
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE))
import inputs as IN
import analyze_research_questions as A
OUT=ROOT/'analysis/candidate_diagnostics'
BOOT=5000

def ratio_ci(pairs, seed=20261023):
    """Resample tasks, retaining all repeat/agent counts inside each task."""
    a=np.asarray(pairs,dtype=float)
    den=a[:,1].sum()
    if not den: return {'rate':None,'lo':None,'hi':None,'valid_bootstraps':0}
    rng=np.random.default_rng(seed); vals=[]
    for start in range(0,BOOT,100):
        ix=rng.integers(0,len(a),size=(min(100,BOOT-start),len(a)))
        sums=a[ix].sum(axis=1); ok=sums[:,1]>0
        vals.extend((sums[ok,0]/sums[ok,1]).tolist())
    lo,hi=np.quantile(vals,[.025,.975]) if vals else (None,None)
    return {'rate':float(a[:,0].sum()/den),'lo':None if lo is None else float(lo),
            'hi':None if hi is None else float(hi),'valid_bootstraps':len(vals)}

def classify(raw, bench, record, gold, kind):
    x=A.norm(raw,kind)
    if x is None: return 'absent'
    if bench in IN.EXT_SQL:
        db=(record.get('config',{}).get('unit_meta') or {}).get('db_id')
        v=IN.sql_label(db,raw) if db else None
        return 'unscorable' if v is None else v
    if record['unit'] not in gold: return 'unscorable'
    return 'correct' if x==A.norm(gold[record['unit']],kind) else 'incorrect'

def counts(r,bench,gold,kind):
    rec=r.get('record') or {}; proto=r['protocol']; c=collections.Counter(N=1)
    initial={}; post={}
    for call in rec.get('calls') or []:
        ai=call.get('agent_index')
        # Aggregator is a downstream generation, not an initial independent candidate.
        if ai is None or (proto=='moa' and call.get('role')!='reference'): continue
        raw=call.get('extracted_answer')
        if raw is None: raw=call.get('prediction')
        s=classify(raw,bench,r,gold,kind)
        initial.setdefault(ai,s); post[ai]=s
    final=classify(rec.get('majority_answer'),bench,r,gold,kind)
    cand=[classify(x,bench,r,gold,kind) for x in (rec.get('per_agent_final_answer') or [])]
    c['final_'+final]=1
    c['candidate_unknown']=int('unscorable' in cand)
    if proto!='moa':
        for ai,s in initial.items():
            if s in ('incorrect','correct') and post[ai]=='unscorable':
                c['post_unknown']+=1; continue
            if s=='incorrect':
                c['W']+=1; c['Wc']+=int(post[ai]=='correct'); c['post_unknown']+=int(post[ai]=='unscorable')
            if s=='correct':
                c['C']+=1; c['R']+=int(post[ai] in ('incorrect','absent')); c['post_unknown']+=int(post[ai]=='unscorable')
        had='correct' in initial.values()
        # Unknown candidate outcomes cannot establish a loss; abstentions can.
        if had and cand and 'unscorable' not in cand:
            c['A']=1; c['L']=int('correct' not in cand)
        c['initial_unknown']=int('unscorable' in initial.values())
    if 'correct' in cand:
        c['O']=1
        c['opportunity_'+final]=1
        c['B']=int(final in ('incorrect','absent'))
        c['O_scored']=int(final!='unscorable')
    elif 'unscorable' not in cand and cand:
        c['no_correct_candidate']=1
        c['correct_without_candidate']=int(final=='correct')
    return c

METRICS={'correction':('Wc','W'),'regression':('R','C'),
 'availability_loss':('L','A'),'failure_given_available':('B','O_scored'),
 'wrong_given_available':('opportunity_incorrect','O_scored'),
 'abstention_given_available':('opportunity_absent','O_scored'),
 'loss_share_all_executions':('L','N'),'failure_share_all_executions':('B','N')}

def main():
    OUT.mkdir(parents=True,exist_ok=True); summaries=[]; rates=[]; audit={}; total=0
    for bench in IN.BENCHMARKS:
        kind=IN.kind_for(bench); gold=IN.gold_for(bench)
        rows,drop=IN.read(bench,apply_transform=False)
        seen=set(); groups=collections.defaultdict(lambda:collections.defaultdict(collections.Counter))
        for r in rows:
            key=tuple(r[k] for k in ('benchmark','protocol','backbone','arm','unit','repeat'))
            if key in seen: raise ValueError(('duplicate',key))
            seen.add(key)
            groups[(r['protocol'],r['backbone'],r['arm'])][r['unit']].update(counts(r,bench,gold,kind))
        audit[bench]={'records':len(rows),'excluded':drop}; total+=len(rows)
        for (proto,bb,arm),units in sorted(groups.items()):
            s=sum(units.values(),collections.Counter())
            meta={'benchmark':bench,'protocol':proto,'backbone':bb,'arm':arm,
                  'readout':'generation' if proto=='moa' else 'selection',
                  'candidate_window':'three references' if proto=='moa' else ('last two predictions' if proto=='ford' else 'final agent answers'),
                  'n_tasks':len(units)}
            summaries.append(dict(meta,**s))
            assert s['opportunity_correct']+s['opportunity_incorrect']+s['opportunity_absent']+s['opportunity_unscorable']==s['O']
            assert s['B']==s['opportunity_incorrect']+s['opportunity_absent']
            for name,(num,den) in METRICS.items():
                applicable=not(proto=='moa' and name in ('correction','regression','availability_loss','loss_share_all_executions'))
                pairs=[(u[num],u[den]) for u in units.values()]
                ci=ratio_ci(pairs) if applicable else {'rate':None,'lo':None,'hi':None,'valid_bootstraps':0}
                rates.append(dict(meta,metric=name,applicable=applicable,numerator=s[num] if applicable else None,
                                  denominator=s[den] if applicable else None,**ci))
        print(bench,len(rows),'records',len(groups),'groups',flush=True)
        del rows
    for name,data in [('counts',summaries),('rates',rates)]:
        keys=list(dict.fromkeys(k for x in data for k in x))
        with (OUT/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,keys);w.writeheader();w.writerows(data)
    old=list(csv.DictReader((A.OUT/'rq_diagnostics.csv').open()))
    lookup={(r['benchmark'],r['protocol'],r['backbone'],r['condition']):r for r in old}
    diffs=[]
    for r in rates:
        name=r['metric']; oldnames={'correction':('correction_rate','correction_lo','correction_hi'),'regression':('regression_rate','regression_lo','regression_hi'),'availability_loss':('availability_loss','loss_lo','loss_hi'),'failure_given_available':('selection_miss','miss_lo','miss_hi')}
        if name not in oldnames:continue
        o=lookup.get((r['benchmark'],r['protocol'],r['backbone'],r['arm']))
        if not o:continue
        p,l,h=oldnames[name]
        diffs.append(dict(r,old_rate=o[p],old_lo=o[l],old_hi=o[h]))
    with (OUT/'before_after.csv').open('w') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for x in diffs for k in x)));w.writeheader();w.writerows(diffs)
    manifest={'records':total,'groups':len(summaries),'bootstrap':BOOT,'seed':20261023,'sources':audit,
      'notes':['Ratios pool counts; bootstrap resamples tasks, retaining repeats and agents.',
               'Matching task is pair: intervals do not account for dependence across pairs sharing entities.',
               'MoA reference availability is static; revision/loss not applicable.',
               'FORD last-two window is not a complete-history candidate pool.',
               'Absent and unscorable outcomes are separate; no DB execution or model calls.',
               'Descriptive diagnostic intervals; no hypothesis tests or new Holm family.']}
    manifest['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    assert total==161200,total
    print('COMPLETE',total,len(summaries),flush=True)

if __name__=='__main__':main()

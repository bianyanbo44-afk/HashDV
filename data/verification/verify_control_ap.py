"""Recompute subset AP and deletion contrasts from saved matched predictions, without a GPU.

Usage: python verify_control_ap.py supplementary_data [--bootstrap]
Requires NumPy >=2.0. True-positive matching at ten IoUs is saved; this checks AP aggregation and contrasts.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def unpack(p):
    po,go=p['prediction_offsets'],p['target_offsets']
    pred=[(p['tp'][po[i]:po[i+1]],p['confidence'][po[i]:po[i+1]],p['prediction_class'][po[i]:po[i+1]])
          for i in range(len(po)-1)]
    target=[p['target_class'][go[i]:go[i+1]] for i in range(len(go)-1)]
    return pred,target


def subset_ap(pred,target,indices):
    hit=np.concatenate([pred[i][0] for i in indices])
    confidence=np.concatenate([pred[i][1] for i in indices])
    cls=np.concatenate([pred[i][2] for i in indices])
    truth=np.concatenate([target[i] for i in indices])
    ordering=np.argsort(-confidence)
    hit,cls=hit[ordering],cls[ordering]
    classes,counts=np.unique(truth,return_counts=True)
    areas=np.zeros((len(classes),hit.shape[1]),float)
    grid=np.linspace(0.,1.,101)
    for row,(c,total) in enumerate(zip(classes,counts)):
        h=hit[cls==c]
        if not len(h):continue
        successes=h.cumsum(axis=0)
        failures=(1-h).cumsum(axis=0)
        recall=successes/(total+1e-16)
        precision=successes/(successes+failures)
        for col in range(hit.shape[1]):
            x=np.r_[0.,recall[:,col],recall[-1,col],1.]
            y=np.r_[1.,precision[:,col],0.,0.]
            envelope=np.maximum.accumulate(y[::-1])[::-1]
            areas[row,col]=np.trapezoid(np.interp(grid,x,envelope),grid)
    return np.array([areas[:,0].mean(),areas.mean()]) if len(classes) else np.zeros(2)


def main(root,bootstrap=False):
    root=Path(root)
    ref=json.loads((root/'additional_analyses/random_deletion_results.json').read_text())
    seeds=[2027,3407,8888]
    pred={};target=None;files=None
    for g in ['a','b','random']:
        for seed in seeds:
            p=np.load(root/'random_deletion/saved_predictions'/f'{g}_{seed}.npz',allow_pickle=False)
            pred[(g,seed)],t=unpack(p)
            if target is None:target=t;files=p['image_names'].tolist()
            else:
                assert files==p['image_names'].tolist()
                assert all(np.array_equal(a,b) for a,b in zip(target,t))
    leaked=set()
    import csv
    with (root/'leak_lists/neu_det_copy_pairs.csv').open(encoding='utf-8') as f:
        leaked.update(row['test_image'] for row in csv.DictReader(f))
    L=np.array([i for i,f in enumerate(files) if f in leaked])
    U=np.array([i for i,f in enumerate(files) if f not in leaked])
    assert len(L)==ref['n_leaked'] and len(U)==ref['n_other']

    def contrast(li,ui,check=False):
        values={};byseed={}
        for seed in seeds:
            m={(g,n):subset_ap(pred[(g,seed)],target,idx) for g in ['a','b','random']
               for n,idx in [('L',li),('U',ui)]}
            gap={g:m[(g,'L')]-m[(g,'U')] for g in ['a','b','random']}
            byseed[seed]={'targeted_effect':gap['a']-gap['b'],'random_effect':gap['a']-gap['random'],
                'specificity_contrast':gap['random']-gap['b']}
            if check:
                for k,metric in enumerate(['map50','map50_95']):
                    r=ref['metrics'][metric]['per_seed'][str(seed)]
                    for g,name in [('a','full'),('b','targeted'),('random','random')]:
                        for n in ['L','U']:
                            assert abs(m[(g,n)][k]-r[name+'_'+n])<1e-10,(seed,metric,g,n)
        return {key:np.mean([byseed[s][key] for s in seeds],axis=0) for key in byseed[seeds[0]]}

    point=contrast(L,U,True)
    for key,value in point.items():
        for k,metric in enumerate(['map50','map50_95']):
            assert abs(value[k]-ref['metrics'][metric]['mean'][key])<1e-10,(metric,key)
    print('Nine detectors, both subsets, both AP endpoints and all paired contrasts verified.')
    if bootstrap:
        rng=np.random.default_rng(271027)
        samples={key:[] for key in point}
        for i in range(1000):
            b=contrast(rng.choice(L,len(L)),rng.choice(U,len(U)))
            for key in samples:samples[key].append(b[key])
            if i%200==0:print('Bootstrap',i,flush=True)
        for key,values in samples.items():
            ci=np.quantile(values,[.025,.975],axis=0)
            for k,metric in enumerate(['map50','map50_95']):
                assert np.max(np.abs(ci[:,k]-ref['metrics'][metric]['ci95'][key]))<1e-10,(metric,key,'CI')
        print('All 1,000-resample confidence intervals independently reproduced.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--bootstrap',action='store_true')
    args=ap.parse_args();main(args.root,args.bootstrap)

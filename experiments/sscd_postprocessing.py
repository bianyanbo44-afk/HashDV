"""Published SSCD score normalization/PCA whitening using a training-only background pool."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
from common import context, background_indices, OUT
OUT.mkdir(parents=True, exist_ok=True)


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def metrics(s,y):
    order=np.argsort(-s,kind='mergesort');v=s[order];yy=y[order]
    tp=np.cumsum(yy);fp=np.cumsum(1-yy)
    last=np.r_[np.flatnonzero(np.diff(v)),len(v)-1]
    f=np.r_[0.,fp[last]/fp[-1]];t=np.r_[0.,tp[last]/tp[-1]]
    return {'tpr1':float(np.interp(.01,f,t)),'auroc':float(np.trapezoid(t,f))}


def main():
    out={}
    for ds in ['neu_det','gc10_det','pcb_det']:
        meta,p,_,_=context(ds)
        bg=background_indices(meta,ds)
        test=np.array([meta['queries'][int(i)]['role']=='test' for i in p['query']]);y=p['label'][test]
        out[ds]={};saved={}
        for prep in ['square320','short288']:
            f=np.load(OUT/f'{ds}_sscd_{prep}_features.npz')
            g=f['gallery'].astype(float);q=f['queries'].astype(float);b=g[bg]
            mu=b.mean(0,keepdims=True)
            _,sv,vt=np.linalg.svd(b-mu,full_matrices=False)
            w=vt.T/np.maximum(sv,1e-12)
            transforms={'raw':(unit(g),unit(q),unit(b)),
                'train_centered':(unit(g-mu),unit(q-mu),unit(b-mu)),
                'whitened':(unit((g-mu)@w),unit((q-mu)@w),unit((b-mu)@w))}
            for kind,(gg,qq,bb) in transforms.items():
                raw=np.einsum('ij,ij->i',qq[p['query']],gg[p['cand']])
                if kind=='whitened':
                    key=prep+'_'+kind
                    out[ds][key]=metrics(raw[test],y);saved[key]=raw
                sim=qq@bb.T
                adjustment=np.partition(sim,-3,axis=1)[:,-3:].mean(1)
                score=raw-adjustment[p['query']]
                key=prep+'_'+kind+'_score_norm'
                out[ds][key]=metrics(score[test],y);saved[key]=score
                print(ds,key,out[ds][key],flush=True)
        np.savez_compressed(OUT/f'{ds}_sscd_postprocessing_scores.npz',**saved)
        (OUT/'sscd_postprocessing.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('MEANS',{k:sum(out[d][k]['tpr1'] for d in out)/3 for k in out['neu_det']},flush=True)


if __name__=='__main__':main()

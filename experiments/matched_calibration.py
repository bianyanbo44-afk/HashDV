"""Apply the same fixed query-background calibration to every verifier."""
import json
from pathlib import Path
import numpy as np

from sscd_postprocessing import metrics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
from common import context, background_indices, OUT
OUT.mkdir(parents=True, exist_ok=True)


def main():
    primary=json.loads((ROOT/'data/source_data/tables_and_text/l1b_results.json').read_text())['per_dataset']
    out={}
    for ds in ['neu_det','gc10_det','pcb_det']:
        meta,p,_,_=context(ds)
        bg=np.isin(p['cand'],background_indices(meta,ds))
        role=np.array([meta['queries'][int(i)]['role']=='test' for i in p['query']])
        a=primary[ds]['alpha_B1']
        cache=OUT/f'{ds}_sscd_pair_scores.npz'
        sscd=dict(np.load(cache,allow_pickle=False)) if cache.exists() else {k[5:]:v for k,v in p.items() if k.startswith('sscd_')}
        scores={'HashDV':a*p['H']+(1-a)*p['D_B1'],'Dense':p['D_B1'].astype(float),'Hash':p['H'].astype(float),
            'SSCD_square320':sscd['square320'],'SSCD_short288':sscd['short288']}
        out[ds]={};saved={}
        for k,score in scores.items():
            calibrated=score.astype(float).copy()
            for q in np.unique(p['query']):
                rows=p['query']==q;b=score[rows&bg]
                assert len(b)>0,(ds,int(q),'no eligible background candidate')
                adjustment=np.sort(b)[-min(3,len(b)):].mean()
                calibrated[rows]-=adjustment
            out[ds][k]=metrics(calibrated[role],p['label'][role]);saved[k]=calibrated
            print(ds,k,out[ds][k],flush=True)
        np.savez_compressed(OUT/f'{ds}_matched_calibration_scores.npz',**saved)
    (OUT/'matched_calibration.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print('MEANS',{k:sum(out[d][k]['tpr1'] for d in out)/3 for k in out['neu_det']},flush=True)


if __name__=='__main__':main()

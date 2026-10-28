"""Verify the supplementary comparison scores and intervention contrasts without GPU inference.

Usage: python verify_saved_results.py /path/to/supplementary_data
Dependency: numpy.
"""
import json
from pathlib import Path
import sys
import numpy as np


def tpr_at_1pct(s,y):
    order=np.argsort(-s,kind='mergesort');ss=s[order];yy=y[order]
    tp=np.cumsum(yy);fp=np.cumsum(1-yy)
    last=np.r_[np.flatnonzero(np.diff(ss)),len(ss)-1]
    return float(np.interp(.01,np.r_[0.,fp[last]/fp[-1]],np.r_[0.,tp[last]/tp[-1]]))


def main(root):
    root=Path(root)
    external=json.loads((root/'additional_analyses/sscd_results.json').read_text())['per_dataset']
    primary=json.loads((root/'source_data/tables_and_text/l1b_results.json').read_text())['per_dataset']
    end=json.loads((root/'additional_analyses/end_to_end_results.json').read_text())
    sift=json.loads((root/'additional_analyses/sift_results.json').read_text())
    post=json.loads((root/'additional_analyses/sscd_postprocessing.json').read_text())
    matched=json.loads((root/'additional_analyses/matched_calibration.json').read_text())
    for ds in ['neu_det','gc10_det','pcb_det']:
        p=np.load(root/'additional_analyses'/f'{ds}_comparison_pairs.npz',allow_pickle=False)
        test=p['role']=='test';y=p['label'][test].astype(float)
        a=primary[ds]['alpha_B1'];f=a*p['H']+(1-a)*p['D_B1']
        got=tpr_at_1pct(f[test],y)
        assert abs(got-primary[ds]['test']['F_B1']['tpr1'])<1e-9,(ds,'HashDV')
        for key in ['square320','square320_centered','short288','short288_centered']:
            got=tpr_at_1pct(p['sscd_'+key][test],y)
            assert abs(got-external[ds][key]['tpr1'])<1e-9,(ds,key)
        got=tpr_at_1pct(p['sift'][test],y)
        assert abs(got-sift[ds]['tpr1'])<1e-9,(ds,'SIFT')
        for key,v in post[ds].items():
            got=tpr_at_1pct(p['post_'+key][test],y)
            assert abs(got-v['tpr1'])<1e-9,(ds,'post',key)
        for key,v in matched[ds].items():
            got=tpr_at_1pct(p['matched_'+key][test],y)
            assert abs(got-v['tpr1'])<1e-9,(ds,'matched',key)
        q=p['query'];counts=np.bincount(q);score=f.astype(float).copy()
        score[(p['label']==1)&(counts[q]==21)]=-1.
        got=tpr_at_1pct(score[test],y)
        assert abs(got-end[ds]['retrieval_aware_tpr1'])<1e-9,(ds,'retrieval-aware')
        print(ds,'HashDV, SIFT, all SSCD configurations, matched calibration and retrieval-aware scores verified')
    controls=json.loads((root/'additional_analyses/random_deletion_results.json').read_text())
    for metric,x in controls['metrics'].items():
        for seed,r in x['per_seed'].items():
            direct=(r['random_L']-r['random_U'])-(r['targeted_L']-r['targeted_U'])
            assert abs(direct-r['specificity_contrast'])<1e-12,(metric,seed)
            assert abs((r['targeted_effect']-r['random_effect'])-direct)<1e-12
    print('All paired-seed intervention contrast identities verified')


if __name__=='__main__':
    if len(sys.argv)!=2:raise SystemExit('Usage: python verify_saved_results.py /path/to/supplementary_data')
    main(sys.argv[1])

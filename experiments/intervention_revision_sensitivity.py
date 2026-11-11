"""Prespecified NEU-DET deletion-contrast sensitivities from saved predictions."""
from pathlib import Path
import argparse
import csv
import hashlib
import itertools
import json
import re
import time
import numpy as np

SEEDS = (2027, 3407, 8888)


def ap50(pred, targets, indices):
    hit = np.concatenate([pred[i][0] for i in indices])
    conf = np.concatenate([pred[i][1] for i in indices])
    cls = np.concatenate([pred[i][2] for i in indices])
    truth = np.concatenate([targets[i] for i in indices])
    order = np.argsort(-conf)
    hit, cls = hit[order], cls[order]
    classes, counts = np.unique(truth, return_counts=True)
    areas = []
    grid = np.linspace(0., 1., 101)
    for c, total in zip(classes, counts):
        h = hit[cls == c]
        if not len(h):
            areas.append(0.)
            continue
        tp = h.cumsum()
        fp = (1-h).cumsum()
        recall = tp/(total+1e-16)
        precision = tp/(tp+fp)
        x = np.r_[0., recall, recall[-1], 1.]
        y = np.r_[1., precision, 0., 0.]
        envelope = np.maximum.accumulate(y[::-1])[::-1]
        areas.append(np.trapezoid(np.interp(grid, x, envelope), grid))
    return float(np.mean(areas)) if areas else 0.


def main(data, output, n_image=1000, n_block=500):
    start = time.time()
    preds = {}
    weights = {}
    target_ref = None
    names_ref = None
    for arm in ('a', 'b', 'random'):
        for seed in SEEDS:
            path = data/'random_deletion/saved_predictions'/f'{arm}_{seed}.npz'
            with np.load(path, allow_pickle=False) as z:
                arrays = {k:z[k] for k in z.files}
            names = arrays['image_names'].tolist()
            po, go = arrays['prediction_offsets'], arrays['target_offsets']
            preds[(arm, seed)] = [(arrays['tp'][po[i]:po[i+1],0].astype(float),
                                   arrays['confidence'][po[i]:po[i+1]],
                                   arrays['prediction_class'][po[i]:po[i+1]]) for i in range(len(names))]
            targets = [arrays['target_class'][go[i]:go[i+1]] for i in range(len(names))]
            weights[f'{arm}_{seed}'] = str(arrays['weight_sha256'].item())
            if target_ref is None:
                target_ref, names_ref = targets, names
            else:
                assert names == names_ref
                assert all(np.array_equal(a,b) for a,b in zip(targets,target_ref))
    with (data/'leak_lists/neu_det_copy_pairs.csv').open(encoding='utf-8') as stream:
        flagged = {r['test_image'] for r in csv.DictReader(stream)}
    membership = np.array([name in flagged for name in names_ref])
    L, U = np.flatnonzero(membership), np.flatnonzero(~membership)
    assert (len(L),len(U)) == (17,343)

    def estimate(li, ui, sampled_seeds=SEEDS):
        unique = set(sampled_seeds)
        gaps = {(arm,seed):ap50(preds[(arm,seed)],target_ref,li)-ap50(preds[(arm,seed)],target_ref,ui)
                for arm in ('a','b','random') for seed in unique}
        dl = np.mean([gaps[('a',s)]-gaps[('b',s)] for s in sampled_seeds])
        gamma = np.mean([gaps[('random',s)]-gaps[('b',s)] for s in sampled_seeds])
        return np.array([dl,gamma])*100

    original = json.loads((data/'additional_analyses/random_deletion_results.json').read_text())
    point = estimate(L,U)
    expected = original['metrics']['map50']['mean']
    assert abs(point[0]/100-expected['targeted_effect']) < 1e-10
    assert abs(point[1]/100-expected['specificity_contrast']) < 1e-10
    result = {'endpoint':'mAP50 points','n_L':len(L),'n_U':len(U),'seed_pairs':list(SEEDS),
              'estimands':['adjusted_selected_content_removal_response','matched_execution_specificity_contrast'],
              'point':{'delta_L':point[0],'Gamma':point[1]},'weight_sha256':weights,
              'scope':'Saved matched TP indicators; no retraining, no new physical-instance labels.',
              'analysis_type':'Post-review prespecified sensitivity, not the original prospective experiment.'}
    result['per_seed'] = {str(s):dict(zip(['delta_L','Gamma'],estimate(L,U,[s]).tolist())) for s in SEEDS}
    result['leave_one_seed_out'] = {str(omitted):dict(zip(['delta_L','Gamma'],estimate(L,U,[s for s in SEEDS if s!=omitted]).tolist())) for omitted in SEEDS}
    seed_samples = np.array([estimate(L,U,ss) for ss in itertools.product(SEEDS,repeat=3)])
    result['seed_resampling_only'] = {'resamples':27,'ci95':dict(zip(['delta_L','Gamma'],np.quantile(seed_samples,[.025,.975],axis=0).T.tolist())),
        'interpretation':'Empirical support from only three observed seed/deletion pairs; not an independent significance test.'}

    def summary(values):
        a = np.asarray(values)
        return {'resamples':len(a),'ci95':dict(zip(['delta_L','Gamma'],np.quantile(a,[.025,.975],axis=0).T.tolist())),
                'median':dict(zip(['delta_L','Gamma'],np.median(a,axis=0).tolist()))}

    rng = np.random.default_rng(271027)
    image_values = []
    for i in range(n_image):
        image_values.append(estimate(rng.choice(L,len(L)),rng.choice(U,len(U))))
        if i%200 == 0: print('image',i,flush=True)
    result['stratified_image_fixed_seeds'] = summary(image_values)
    if n_image == 1000:
        for k,key in enumerate(['targeted_effect','specificity_contrast']):
            assert np.max(np.abs(np.quantile(image_values,[.025,.975],axis=0)[:,k]/100 - np.array(original['metrics']['map50']['ci95'][key]))) < 1e-10

    block_results = {}
    for width in (5,10,20):
        keys=[]
        for name in names_ref:
            match=re.fullmatch(r'(.+)_(\d+)\.\w+',name)
            assert match,name
            keys.append((match.group(1),(int(match.group(2))-1)//width))
        blocks = [np.array([i for i,k in enumerate(keys) if k==key]) for key in sorted(set(keys))]
        config = {'filename_index_block_width':width,'n_blocks':len(blocks),
                  'n_blocks_with_flagged_images':sum(bool(membership[b].any()) for b in blocks),
                  'block_definition':'class + floor((filename numeric index - 1)/width)',
                  'physical_acquisition_units_verified':False}
        rng = np.random.default_rng(11112026+width)
        draws=[]
        for i in range(n_block):
            while True:
                chosen=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))])
                li,ui=chosen[membership[chosen]],chosen[~membership[chosen]]
                if len(li) and len(ui): break
            draws.append(estimate(li,ui))
        config['fixed_seeds']=summary(draws)
        if width==10:
            rng=np.random.default_rng(11112036)
            draws=[]
            for i in range(n_block):
                while True:
                    chosen=np.concatenate([blocks[j] for j in rng.integers(len(blocks),size=len(blocks))])
                    li,ui=chosen[membership[chosen]],chosen[~membership[chosen]]
                    if len(li) and len(ui): break
                seeds=rng.choice(SEEDS,size=3,replace=True).tolist()
                draws.append(estimate(li,ui,seeds))
            config['paired_seed_and_block']=summary(draws)
        block_results[str(width)]=config
        print('blocks',width,config,flush=True)
    result['adjacency_block_sensitivity']=block_results
    result['seconds']=round(time.time()-start,2)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'point':result['point'],'image':result['stratified_image_fixed_seeds'],'seconds':result['seconds']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',type=Path,default=Path(__file__).resolve().parents[1]/'data')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'data/additional_analyses/intervention_revision_sensitivity.json')
    parser.add_argument('--image-resamples',type=int,default=1000)
    parser.add_argument('--block-resamples',type=int,default=500)
    args=parser.parse_args()
    main(args.data,args.output,args.image_resamples,args.block_resamples)

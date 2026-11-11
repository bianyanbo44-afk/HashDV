"""Post-hoc test-subset sensitivity for existing 17-image NEU deletion models.

All full/targeted/random models remain those of the original intervention.
The original U343 is held as reference. No new model is trained by this script.
"""
from pathlib import Path
import argparse,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hashdv.verification.verify_control_ap import unpack,subset_ap
DATA=ROOT/'data'
OUT=DATA/'additional_analyses'
SEEDS=(2027,3407,8888)


def main(nboot):
    geometry=json.loads((OUT/'real_pair_geometry.json').read_text())['neu_det']['pairs']
    flagged={r['test'] for r in geometry}
    corroborated={r['test'] for r in geometry if r['corroborated']}
    assert len(flagged)==17 and len(corroborated)==9
    pred={}
    target=None
    for arm in ['a','b','random']:
        for seed in SEEDS:
            with np.load(DATA/'random_deletion/saved_predictions'/f'{arm}_{seed}.npz',allow_pickle=False) as z:
                arrays={k:z[k] for k in z.files}
            p,t=unpack(arrays)
            # Primary endpoint only; AP uses the same supplied implementation.
            pred[(arm,seed)]=[(tp[:,:1],c,k) for tp,c,k in p]
            names=arrays['image_names'].tolist()
            if target is None:
                target=t;reference_names=names
            else:
                assert reference_names==names
                assert all(np.array_equal(a,b) for a,b in zip(target,t))
    groups={'corroborated_9':np.array([i for i,n in enumerate(names) if n in corroborated]),
            'uncertain_8':np.array([i for i,n in enumerate(names) if n in flagged-corroborated]),
            'all_flagged_17':np.array([i for i,n in enumerate(names) if n in flagged]),
            'original_U343':np.array([i for i,n in enumerate(names) if n not in flagged]),
            'original_T360':np.arange(len(names))}
    def accuracy(indices):
        return {(arm,seed):float(subset_ap(pred[(arm,seed)],target,indices)[0]) for arm in ['a','b','random'] for seed in SEEDS}
    def effects(l,u):
        records={}
        for seed in SEEDS:
            a,b,r=(l[(arm,seed)] for arm in ['a','b','random'])
            au,bu,ru=(u[(arm,seed)] for arm in ['a','b','random'])
            records[str(seed)]={'full_L':a,'targeted_L':b,'random_L':r,
                'full_U':au,'targeted_U':bu,'random_U':ru,
                'delta_L':(a-b)-(au-bu),'gamma':(r-b)-(ru-bu)}
        means={key:float(np.mean([d[key] for d in records.values()])) for key in records[str(SEEDS[0])]}
        return {'per_seed':records,'mean':means}
    u=accuracy(groups['original_U343'])
    point={group:effects(accuracy(indices),u) for group,indices in groups.items() if group!='original_U343'}
    draws={key:{'delta_L':[],'gamma':[]} for key in ['corroborated_9','uncertain_8']}
    rng=np.random.default_rng(11112027)
    for b in range(nboot):
        boot_u=accuracy(rng.choice(groups['original_U343'],size=343,replace=True))
        for key in draws:
            selected=groups[key]
            e=effects(accuracy(rng.choice(selected,size=len(selected),replace=True)),boot_u)['mean']
            for metric in draws[key]:draws[key][metric].append(e[metric])
        if b%100==0:print('bootstrap',b,flush=True)
    for key in draws:
        point[key]['image_bootstrap_ci95']={metric:np.quantile(values,[.025,.975]).tolist() for metric,values in draws[key].items()}
    whole=point['original_T360']
    for row in [whole['mean'],*whole['per_seed'].values()]:
        for arm in ['full','targeted','random']:
            row[arm+'_T']=row.pop(arm+'_L')
        row['delta_T']=row.pop('delta_L')
        row['gamma_T']=row.pop('gamma')
    result={'analysis_type':'post-hoc subgroup sensitivity of existing frozen intervention',
            'models':'original full, 17-image targeted deletion, and class-matched 17-image random-deletion models',
            'not_a_retraining_of_only_nine_copies':True,
            'reference_group':'original unflagged U343 throughout; uncertain eight never reassigned to clean control',
            'selection_rule':'pre-existing orthogonal SIFT corroboration, 8 inliers and 0.5 inlier fraction; nine also have AI-assisted visual correspondence',
            'endpoint':'mAP50','bootstrap_unit':'test image within selected L and original U; fixed three paired seed models',
            'bootstrap_replicates':nboot,'bootstrap_seed':11112027,
            'groups':{key:[reference_names[i] for i in indices] for key,indices in groups.items()},
            'results':point}
    ref=json.loads((OUT/'random_deletion_results.json').read_text())['metrics']['map50']['mean']
    assert abs(point['all_flagged_17']['mean']['delta_L']-ref['targeted_effect'])<1e-10
    assert abs(point['all_flagged_17']['mean']['gamma']-ref['specificity_contrast'])<1e-10
    original=json.loads((DATA/'source_data/tables_and_text/l1c_results.json').read_text())['neu_det']['map50']['delta_T']
    assert abs(point['original_T360']['mean']['delta_T']-original)<1e-10
    (OUT/'instance_intervention_sensitivity.json').write_text(json.dumps(result,indent=2)+chr(10),encoding='utf-8')
    print(json.dumps({key:{'mean':d['mean'],'ci95':d.get('image_bootstrap_ci95')} for key,d in point.items()},indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bootstrap',type=int,default=1000)
    main(p.parse_args().bootstrap)

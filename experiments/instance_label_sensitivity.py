"""Deterministic non-source-label sensitivity; original arrays are never modified."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hashdv.calibration import fit, metrics
DATA = ROOT / 'data'
ADDITIONAL = DATA / 'additional_analyses'
DATASETS = {
    'neu_det': ('neu_det_copy_pairs.csv', 'test_image', 360),
    'gc10_det': ('gc10_det_copy_pairs.csv', 'test_image', 460),
    'pcb_det': ('dspcbsd_official_copy_pairs.csv', 'validation_image', 2051),
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class WeightedROC:
    """Reuse a stable score order for a paired source-cluster bootstrap."""
    def __init__(self, scores, labels, group):
        order = np.argsort(-scores, kind='mergesort')
        s = scores[order]
        self.y = labels[order]
        self.group = group[order]
        self.ends = np.r_[np.flatnonzero(np.diff(s)), len(s)-1]

    def tpr(self, counts, rate=.01):
        w = counts[self.group]
        tp = np.cumsum(w * self.y)[self.ends]
        fp = np.cumsum(w * (1-self.y))[self.ends]
        if not tp[-1] or not fp[-1]:
            raise ValueError('bootstrap draw must contain both labels')
        return float(np.interp(rate, np.r_[0., fp/fp[-1]], np.r_[0., tp/tp[-1]]))


def query_hit_rate(z, score, tau, mask, role):
    candidates = mask & (z['label'] == 0) & (z['role'] == role)
    hit = np.zeros(int(z['query'].max())+1, bool)
    np.logical_or.at(hit, z['query'][candidates], score[candidates] >= tau)
    queries = np.unique(z['query'][z['role'] == role])
    return {'n_queries': len(queries), 'n_hit': int(hit[queries].sum()),
            'rate': float(hit[queries].mean())}


def main(n_bootstrap):
    results = {'analysis_type': 'post-review sensitivity, not independent truth',
               'random_seed': 11112026, 'bootstrap_replicates': n_bootstrap,
               'bootstrap_unit': 'original source image, paired transformations and candidate rows',
               'exclusion_rule': 'negative source-candidate original pair is in frozen flagged-pair CSV',
               'original_arrays_and_audit_sets_modified': False, 'datasets': {}}
    draws = {mode: [] for mode in ['original','exclude_flagged_pair_negatives']}
    conflicts = []
    for ds, (csv_name, query_col, n_eval) in DATASETS.items():
        archive = ADDITIONAL / f'{ds}_comparison_pairs.npz'
        before = digest(archive)
        with np.load(archive, allow_pickle=False) as f:
            z = {k:f[k] for k in f.files}
        meta = read(ADDITIONAL / f'{ds}_pair_index.json')
        with (DATA/'leak_lists'/csv_name).open(encoding='utf-8') as f:
            pairs = {(r[query_col],r['training_image']) for r in csv.DictReader(f)}
        exclude = np.zeros(len(z['label']),bool)
        for row in np.flatnonzero(z['label']==0):
            q = meta['queries'][int(z['query'][row])]
            source = meta['gallery'][q['source']]
            candidate = meta['gallery'][int(z['cand'][row])]
            if (source,candidate) in pairs:
                exclude[row] = True
                conflicts.append({'dataset':ds,'row':int(row),'role':str(z['role'][row]),
                                  'query':int(z['query'][row]),'source_image':source,
                                  'candidate_image':candidate,'copy_file':q['copy_file']})
        fm = z['role']=='fit'
        original_fit = fit(z['H'][fm],z['D_B1'][fm],z['label'][fm])
        sensitivity_fit = fit(z['H'][fm & ~exclude],z['D_B1'][fm & ~exclude],z['label'][fm & ~exclude])
        alpha = original_fit['alpha']
        score = alpha*z['H']+(1-alpha)*z['D_B1']
        refit_score = sensitivity_fit['alpha']*z['H']+(1-sensitivity_fit['alpha'])*z['D_B1']
        methods = {'HashDV':score,'Hash':z['H'],'Dense':z['D_B1'], 'Global':z['G'],
                   'SSCD_cosine':z['sscd_square320'],'SIFT':z['sift'],
                   'matched_HashDV':z['matched_HashDV'],'matched_Dense':z['matched_Dense'],
                   'matched_SSCD':z['matched_SSCD_short288'],
                   'SSCD_full_background':z['post_short288_raw_score_norm']}
        record = {'original_npz_sha256':before,
                  'excluded_negative_rows':{r:int(np.sum(exclude & (z['role']==r))) for r in ['fit','test']},
                  'original_fit':original_fit,'sensitivity_fit':sensitivity_fit,'evaluations':{}}
        for mode, keep in [('original',np.ones(len(exclude),bool)),('exclude_flagged_pair_negatives',~exclude)]:
            mask = keep & (z['role']=='test')
            ev = {method:metrics(s[mask],z['label'][mask]) for method,s in methods.items()}
            ev['HashDV_refit_sensitivity'] = metrics(refit_score[mask],z['label'][mask])
            rates = {role:query_hit_rate(z,score,original_fit['threshold_tau'],keep,role) for role in ['fit','test']}
            refit_rates = {role:query_hit_rate(z,refit_score,sensitivity_fit['threshold_tau'],keep,role) for role in ['fit','test']}
            out = {'test_pairs':int(mask.sum()),'test_positive_pairs':int(z['label'][mask].sum()),
                'method_metrics':ev,'frozen_threshold_non_source_query_hits':rates,
                'refitted_threshold_non_source_query_hits':refit_rates,
                'synthetic_query_reference_count_on_real_split':rates['test']['rate']*n_eval}
            record['evaluations'][mode] = out
            sources = np.array([q['source'] for q in meta['queries']], int)
            _, group = np.unique(sources[z['query'][mask]], return_inverse=True)
            n_sources = int(group.max()+1)
            record['test_source_images'] = n_sources
            keys = ['HashDV','Hash','Dense','matched_HashDV','matched_Dense','matched_SSCD']
            rocs = {k:WeightedROC(methods[k][mask],z['label'][mask],group) for k in keys}
            local_rng = np.random.default_rng(11112026 + list(DATASETS).index(ds))
            sample = []
            for _ in range(n_bootstrap):
                counts = np.bincount(local_rng.integers(n_sources,size=n_sources),minlength=n_sources)
                t = {k:v.tpr(counts) for k,v in rocs.items()}
                sample.append([t['HashDV']-max(t['Hash'],t['Dense']),
                               t['matched_HashDV']-t['matched_Dense'],
                               t['matched_HashDV']-t['matched_SSCD']])
            draws[mode].append(np.array(sample))
            point = [ev['HashDV']['tpr1']-max(ev['Hash']['tpr1'],ev['Dense']['tpr1']),
                     ev['matched_HashDV']['tpr1']-ev['matched_Dense']['tpr1'],
                     ev['matched_HashDV']['tpr1']-ev['matched_SSCD']['tpr1']]
            out['source_bootstrap_gains'] = {key:{'point':float(point[k]),
                'ci95':np.quantile(np.array(sample)[:,k],[.025,.975]).tolist()}
                for k,key in enumerate(['base_vs_best_single','matched_vs_dense','matched_vs_SSCD'])}
        assert before == digest(archive)
        results['datasets'][ds] = record
        print(ds,'exclusions',record['excluded_negative_rows'],'fit',original_fit,sensitivity_fit,flush=True)
    results['macro_average'] = {}
    for mode in draws:
        mean_draw = np.mean(draws[mode],axis=0)
        means = {method:float(np.mean([d['evaluations'][mode]['method_metrics'][method]['tpr1']
                  for d in results['datasets'].values()])) for method in methods}
        gains = {key:{'point':float(np.mean([d['evaluations'][mode]['source_bootstrap_gains'][key]['point']
                for d in results['datasets'].values()])), 'ci95':np.quantile(mean_draw[:,k],[.025,.975]).tolist()}
                for k,key in enumerate(['base_vs_best_single','matched_vs_dense','matched_vs_SSCD'])}
        results['macro_average'][mode] = {'mean_tpr1':means,'source_bootstrap_gains':gains}
    (ADDITIONAL/'instance_label_sensitivity.json').write_text(json.dumps(results,indent=2)+chr(10),encoding='utf-8')
    fields = ['dataset','row','role','query','source_image','candidate_image','copy_file']
    with (ADDITIONAL/'instance_label_conflict_rows.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(conflicts)
    print(json.dumps(results['macro_average'],indent=2),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bootstrap',type=int,default=1000)
    args=ap.parse_args()
    if args.bootstrap < 1:
        ap.error('--bootstrap must be positive')
    main(args.bootstrap)

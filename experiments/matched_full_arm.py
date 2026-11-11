"""Steel removal responses with the execution-matched full arm.

The original steel full models were trained with zero workers, no cache and plots enabled, while the removal arms
used one worker, RAM caching and plots disabled. The full arms were retrained with the removal arms' settings (same
data, seeds, pretrained weights and hyperparameters); their predictions are data/matched_full_arm/<dataset>/
a_matched_<seed>.npz. This script recomputes every quantity that uses the full arm from saved predictions and the
released AP code: Table VI rows, seed-aware intervals, the neighbour control, the subset gap and, for NEU-DET, the
match-evidence subgroups. The original full-arm values are reproduced alongside as a check.

Usage: python experiments/matched_full_arm.py [--data data] [--datasets neu_det gc10_det] [--resamples 1000]
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from hashdv.verification.verify_control_ap import load_dataset, load_predictions, subset_ap  # noqa: E402

SEEDS = (2027, 3407, 8888)
METRICS = ('map50', 'map50_95')


def analyse(data, dataset, nboot):
    pred, target, L, U = load_dataset(data, dataset, arms=('a', 'b'))
    names = None
    for seed in SEEDS:
        p, t, n = load_predictions(data / 'matched_full_arm' / dataset / f'a_matched_{seed}.npz')
        assert all(np.array_equal(x, y) for x, y in zip(t, target))
        names = names or n
        pred[('am', seed)] = p
    T = np.r_[L, U]
    sub = lambda arm, seed, idx: subset_ap(pred[(arm, seed)], target, idx)  # noqa: E731

    def responses(full, li, ui, seeds=SEEDS):
        ti, vals = np.r_[li, ui], []
        for s in seeds:
            a = [sub(full, s, i) for i in (li, ui, ti)]
            b = [sub('b', s, i) for i in (li, ui, ti)]
            shared = a[1] - b[1]
            vals.append(np.stack([a[0] - b[0] - shared, a[2] - b[2] - shared]))  # [delta_L, delta_T] x metric
        return np.mean(vals, axis=0)

    def accuracy(full):
        return {metric: {f'{label}_{s}': float(np.mean([sub(arm, seed, idx)[k] for seed in SEEDS]))
                         for arm, label in ((full, 'full'), ('b', 'removed')) for s, idx in (('L', L), ('U', U), ('T', T))}
                for k, metric in enumerate(METRICS)}

    result = {'dataset': dataset, 'units': 'AP fraction (x100 for points)', 'n_L': len(L), 'n_U': len(U)}
    for full, label in (('am', 'matched_full'), ('a', 'original_full')):
        point = responses(full, L, U)
        result[label] = {'accuracy': accuracy(full),
                         'delta_L': dict(zip(METRICS, point[0].tolist())),
                         'delta_T': dict(zip(METRICS, point[1].tolist())),
                         'per_seed_map50': {str(s): {**dict(zip(('delta_L', 'delta_T'), responses(full, L, U, [s])[:, 0].tolist())),
                                                     **{f'{lab}_{n}': float(sub(arm, s, idx)[0]) for arm, lab in ((full, 'full'), ('b', 'removed'))
                                                        for n, idx in (('L', L), ('U', U))}}
                                            for s in SEEDS}}
    ref = json.loads((data / 'source_data/tables_and_text/l1c_results.json').read_text())[dataset]
    assert abs(result['original_full']['delta_L']['map50'] - ref['map50']['delta_L']) < 1e-6

    rng = np.random.default_rng(0)  # bootstrap seed of the original steel estimates
    draws = np.array([responses('am', rng.choice(L, len(L)), rng.choice(U, len(U))) for _ in range(nboot)])
    q = np.quantile(draws, [.025, .975], axis=0)
    result['matched_full']['image_bootstrap_ci95'] = {
        est: {metric: q[:, i, k].tolist() for k, metric in enumerate(METRICS)} for i, est in enumerate(('delta_L', 'delta_T'))}
    rng = np.random.default_rng(11112042)
    joint = np.array([responses('am', rng.choice(L, len(L)), rng.choice(U, len(U)),
                                rng.choice(SEEDS, 3, replace=True).tolist())[:, 0] for _ in range(nboot)])
    result['matched_full']['paired_seed_and_image_ci95_map50'] = dict(zip(
        ('delta_L', 'delta_T'), np.quantile(joint, [.025, .975], axis=0).T.tolist()))

    p50 = {key: [(tp[:, :1], c, k) for tp, c, k in v] for key, v in pred.items()}
    ap = lambda arm, s, idx: subset_ap(p50[(arm, s)], target, idx)[0]  # noqa: E731

    def dl(full, li, ui):
        return float(np.mean([(ap(full, s, li) - ap('b', s, li)) - (ap(full, s, ui) - ap('b', s, ui)) for s in SEEDS]))

    if dataset == 'neu_det':  # match-evidence subgroups, as experiments/instance_intervention_sensitivity.py
        geometry = json.loads((data / 'additional_analyses/real_pair_geometry.json').read_text())['neu_det']['pairs']
        flagged = {r['test'] for r in geometry}
        corroborated = {r['test'] for r in geometry if r['corroborated']}
        groups = {'corroborated_9': np.array([i for i, n in enumerate(names) if n in corroborated]),
                  'uncertain_8': np.array([i for i, n in enumerate(names) if n in flagged - corroborated])}
        sg = {key: {'delta_L_map50': dl('am', idx, U), 'original_full_delta_L_map50': dl('a', idx, U)}
              for key, idx in groups.items()}
        rng = np.random.default_rng(11112027)
        boot = {key: [] for key in groups}
        for _ in range(nboot):
            bu = rng.choice(U, len(U))
            for key, idx in groups.items():
                boot[key].append(dl('am', rng.choice(idx, len(idx)), bu))
        for key in groups:
            sg[key]['image_bootstrap_ci95'] = np.quantile(boot[key], [.025, .975]).tolist()
        result['match_evidence_subgroups'] = sg

    near_names = set(json.loads((data / 'matched_full_arm' / dataset / 'neighbour_groups.json').read_text())['U_nearR'])
    near = np.array([i for i in U if names[i] in near_names])
    far = np.array([i for i in U if names[i] not in near_names])
    neigh = {'n_U_nearR': len(near), 'n_U_far': len(far),
             'gap_removed': float(np.mean([ap('b', s, L) - ap('b', s, U) for s in SEEDS]))}
    for full, label in (('am', 'matched_full'), ('a', 'original_full')):
        drop = {name: float(np.mean([ap(full, s, idx) - ap('b', s, idx) for s in SEEDS]))
                for name, idx in (('L', L), ('U_nearR', near), ('U_far', far))}
        drop['delta_L_far_reference'] = dl(full, L, far)
        drop['gap_full'] = float(np.mean([ap(full, s, L) - ap(full, s, U) for s in SEEDS]))
        neigh[label] = drop
    result['neighbour_and_gap'] = neigh
    out = data / 'additional_analyses' / f'{dataset}_matched_full_arm.json'
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT / 'data')
    p.add_argument('--datasets', nargs='+', default=['neu_det', 'gc10_det'])
    p.add_argument('--resamples', type=int, default=1000)
    a = p.parse_args()
    for d in a.datasets:
        print(json.dumps(analyse(a.data, d, a.resamples), indent=1), flush=True)

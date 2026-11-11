"""Rebuild Table VI and the NEU random-deletion control from saved matched predictions.

Verifies AP aggregation, not detector inference or box matching. The legacy copy
inside data/verification remains unchanged historical evidence.
"""
import argparse
import csv
import json
from pathlib import Path
import re
import numpy as np

SEEDS = (2027, 3407, 8888)
METRICS = ("map50", "map")
# Far finer than manuscript precision; confidence ties can move AP by 1e-8.
SCORE_ATOL = 1e-6


def unpack(p):
    po, go = p['prediction_offsets'], p['target_offsets']
    pred = [(p['tp'][po[i]:po[i+1]], p['confidence'][po[i]:po[i+1]],
             p['prediction_class'][po[i]:po[i+1]]) for i in range(len(po)-1)]
    target = [p['target_class'][go[i]:go[i+1]] for i in range(len(go)-1)]
    return pred, target


def load_predictions(path):
    with np.load(path, allow_pickle=False) as z:
        p = {key: z[key] for key in z.files}
    n = len(p['image_names'])
    assert len(set(p['image_names'].tolist())) == n, (path, 'duplicate image names')
    for key, count in [('prediction_offsets', len(p['confidence'])),
                       ('target_offsets', len(p['target_class']))]:
        offsets = p[key]
        assert offsets.shape == (n+1,) and offsets.dtype.kind in 'iu', (path, key)
        assert offsets[0] == 0 and offsets[-1] == count, (path, key)
        assert np.all(np.diff(offsets) >= 0), (path, key)
    assert p['tp'].shape == (len(p['confidence']), 10), (path, 'TP layout')
    assert len(p['prediction_class']) == len(p['confidence']), path
    assert np.isfinite(p['confidence']).all(), path
    assert re.fullmatch(r'[0-9a-f]{64}', str(p['weight_sha256'].item())), (path, 'weight hash')
    pred, target = unpack(p)
    return pred, target, p['image_names'].tolist()


def subset_ap(pred, target, indices):
    if not len(indices):
        raise ValueError('Cannot compute a reported subset AP from an empty image set')
    hit = np.concatenate([pred[i][0] for i in indices])
    confidence = np.concatenate([pred[i][1] for i in indices])
    cls = np.concatenate([pred[i][2] for i in indices])
    truth = np.concatenate([target[i] for i in indices])
    ordering = np.argsort(-confidence)
    hit, cls = hit[ordering], cls[ordering]
    classes, counts = np.unique(truth, return_counts=True)
    areas = np.zeros((len(classes), hit.shape[1]), float)
    grid = np.linspace(0., 1., 101)
    for row, (c, total) in enumerate(zip(classes, counts)):
        h = hit[cls == c]
        if not len(h):
            continue
        successes = h.cumsum(axis=0)
        failures = (1-h).cumsum(axis=0)
        recall = successes / (total+1e-16)
        precision = successes / (successes+failures)
        for col in range(hit.shape[1]):
            x = np.r_[0., recall[:, col], recall[-1, col], 1.]
            y = np.r_[1., precision[:, col], 0., 0.]
            envelope = np.maximum.accumulate(y[::-1])[::-1]
            areas[row, col] = np.trapezoid(np.interp(grid, x, envelope), grid)
    return np.array([areas[:, 0].mean(), areas.mean()]) if len(classes) else np.zeros(2)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check_close(actual, expected, pointer, atol=SCORE_ATOL):
    if not np.allclose(actual, expected, rtol=0, atol=atol):
        raise AssertionError((pointer, 'recomputed', np.asarray(actual).tolist(), 'reference', expected))


def load_dataset(root, dataset, arms=('a', 'b')):
    directory = (root / 'random_deletion/saved_predictions' if dataset == 'neu_det'
                 else root / 'intervention_predictions' / dataset)
    missing = [str(directory / f'{arm}_{seed}.npz') for arm in arms for seed in SEEDS
               if not (directory / f'{arm}_{seed}.npz').is_file()]
    if missing:
        raise FileNotFoundError('Incomplete intervention predictions: ' + ', '.join(missing))
    predictions, targets, names = {}, None, None
    for arm in arms:
        for seed in SEEDS:
            pred, target, image_names = load_predictions(directory / f'{arm}_{seed}.npz')
            if targets is None:
                targets, names = target, image_names
            else:
                assert image_names == names, (dataset, arm, seed, 'image alignment')
                assert all(np.array_equal(a, b) for a, b in zip(targets, target)), (dataset, arm, seed, 'targets')
            predictions[(arm, seed)] = pred
    filename, query_key = (('dspcbsd_official_copy_pairs.csv', 'validation_image') if dataset == 'pcb_det'
                           else (dataset+'_copy_pairs.csv', 'test_image'))
    with (root / 'leak_lists' / filename).open(encoding='utf-8') as handle:
        flagged = {row[query_key] for row in csv.DictReader(handle)}
    assert flagged <= set(names), (dataset, 'flagged images missing from predictions')
    L = np.array([i for i, name in enumerate(names) if name in flagged], dtype=np.int64)
    U = np.array([i for i, name in enumerate(names) if name not in flagged], dtype=np.int64)
    return predictions, targets, L, U


def estimate(predictions, target, L, U):
    # Original estimator concatenates L then U; retain its confidence-tie ordering.
    subsets = {'L': L, 'U': U, 'T': np.r_[L, U]}
    per_seed = {}
    for seed in SEEDS:
        values = {arm+'_'+name: subset_ap(predictions[(arm, seed)], target, indices)
                  for arm in ('a', 'b') for name, indices in subsets.items()}
        shared = values['a_U']-values['b_U']
        values['delta_L'] = values['a_L']-values['b_L']-shared
        values['delta_T'] = values['a_T']-values['b_T']-shared
        per_seed[str(seed)] = values
    mean = {key: np.mean([v[key] for v in per_seed.values()], axis=0)
            for key in per_seed[str(SEEDS[0])]}
    return mean, per_seed


def verify_main_interventions(root, bootstrap=False):
    steel = read_json(root / 'source_data/tables_and_text/l1c_results.json')
    steel_seeds = read_json(root / 'source_data/tables_and_text/l1c_extra.json')
    pcb = read_json(root / 'source_data/tables_and_text/l1d_results.json')
    references = {**steel, 'pcb_det': pcb}
    coverage = {}
    for dataset in ('neu_det', 'gc10_det', 'pcb_det'):
        reference = references[dataset]
        predictions, targets, L, U = load_dataset(root, dataset)
        assert (len(L), len(U)) == (reference['n_L'], reference['n_U']), dataset
        mean, per_seed = estimate(predictions, targets, L, U)
        seed_reference = pcb['per_seed'] if dataset == 'pcb_det' else steel_seeds[dataset]
        for col, metric in enumerate(METRICS):
            for key, actual in mean.items():
                check_close(actual[col], reference[metric][key], (dataset, metric, key))
            for seed, values in per_seed.items():
                for key, expected in seed_reference[seed][metric].items():
                    check_close(values[key][col], expected, (dataset, seed, metric, key))
        coverage[dataset] = {'images': {'L': len(L), 'U': len(U), 'T': len(L)+len(U)},
                             'seeds': list(SEEDS), 'endpoints': ['mAP50', 'mAP50-95'],
                             'delta_L_points': (100*mean['delta_L']).tolist(),
                             'delta_T_points': (100*mean['delta_T']).tolist()}
        print(dataset, 'L/U/T, two AP endpoints, three paired seeds and delta_L/delta_T verified', flush=True)
        if bootstrap:
            rng = np.random.default_rng(0)
            samples = {key: [] for key in ('delta_L', 'delta_T')}
            for iteration in range(1000):
                result, _ = estimate(predictions, targets, rng.choice(L, len(L)), rng.choice(U, len(U)))
                for key in samples:
                    samples[key].append(result[key])
                if iteration % 200 == 0:
                    print(dataset, 'main-table bootstrap', iteration, flush=True)
            for key, values in samples.items():
                interval = np.quantile(values, [.025, .975], axis=0)
                for col, metric in enumerate(METRICS):
                    check_close(interval[:, col], reference[metric][key+'_ci95'], (dataset, metric, key, 'CI'))
            print(dataset, '1,000-resample Table VI intervals verified', flush=True)
    return coverage


def verify_random_control(root, bootstrap=False):
    ref = read_json(root / 'additional_analyses/random_deletion_results.json')
    predictions, target, L, U = load_dataset(root, 'neu_det', ('a', 'b', 'random'))
    assert (len(L), len(U)) == (ref['n_leaked'], ref['n_other'])

    def contrast(li, ui, check=False):
        byseed = {}
        for seed in SEEDS:
            values = {(arm, name): subset_ap(predictions[(arm, seed)], target, indices)
                      for arm in ('a', 'b', 'random') for name, indices in [('L', li), ('U', ui)]}
            gap = {arm: values[(arm, 'L')]-values[(arm, 'U')] for arm in ('a', 'b', 'random')}
            byseed[seed] = {'targeted_effect': gap['a']-gap['b'],
                            'random_effect': gap['a']-gap['random'],
                            'specificity_contrast': gap['random']-gap['b']}
            if check:
                for col, metric in enumerate(('map50', 'map50_95')):
                    r = ref['metrics'][metric]['per_seed'][str(seed)]
                    for arm, label in [('a', 'full'), ('b', 'targeted'), ('random', 'random')]:
                        for name in ('L', 'U'):
                            check_close(values[(arm, name)][col], r[label+'_'+name], ('random control', seed, metric, arm, name), 1e-10)
        return {key: np.mean([byseed[seed][key] for seed in SEEDS], axis=0) for key in byseed[SEEDS[0]]}

    point = contrast(L, U, True)
    for key, value in point.items():
        for col, metric in enumerate(('map50', 'map50_95')):
            check_close(value[col], ref['metrics'][metric]['mean'][key], ('random control', metric, key), 1e-10)
    print('NEU class-matched random control, nine detector instances, L/U and all three contrasts verified', flush=True)
    if bootstrap:
        rng = np.random.default_rng(ref.get('bootstrap_rng_seed', 271027))
        samples = {key: [] for key in point}
        for iteration in range(1000):
            result = contrast(rng.choice(L, len(L)), rng.choice(U, len(U)))
            for key in samples:
                samples[key].append(result[key])
            if iteration % 200 == 0:
                print('NEU control bootstrap', iteration, flush=True)
        for key, values in samples.items():
            interval = np.quantile(values, [.025, .975], axis=0)
            for col, metric in enumerate(('map50', 'map50_95')):
                check_close(interval[:, col], ref['metrics'][metric]['ci95'][key], (metric, key, 'control CI'), 1e-10)
        print('All 1,000-resample NEU random-control intervals verified', flush=True)


def main(root, bootstrap=False):
    root = Path(root)
    coverage = verify_main_interventions(root, bootstrap=bootstrap)
    verify_random_control(root, bootstrap=bootstrap)
    print('Verified Table VI for all three datasets and the NEU random-deletion control.', flush=True)
    if not bootstrap:
        print('Confidence intervals not recomputed; use --bootstrap for all main-table and NEU control intervals.', flush=True)
    return coverage


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--bootstrap', action='store_true')
    args = parser.parse_args()
    main(args.root, args.bootstrap)

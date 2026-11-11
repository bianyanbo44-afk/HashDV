"""Synthetic non-source query exceedances and hash coverage of the frozen matched pairs.

The source-identity labels distinguish the generating source file from other candidate files. The diagnostic counts
synthetic queries with a non-source candidate at or above the frozen threshold. A non-source file can share a physical
region with its source, so these labels leave natural-instance identity unresolved.

Multiplying that synthetic rate by an evaluation-set size gives a hypothetical rescaling under the same rate.
Its interpretation requires matching query and gallery conditions. The released legacy fields
per_query_false_flag_rate and expected_chance_flags_test_rate retain their names for compatibility.
The natural audit counts remain operational outputs of the frozen rule.

The script also counts matched pairs satisfying the exact rotation-aware pHash rule and a 5-bit tolerance.

Usage: python experiments/chance_flags.py data [--output data/additional_analyses/chance_flags.json]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

BENCHMARKS = {
    "neu_det": ("NEU-DET", "neu_det_copy_pairs.csv", "test_image", 360),
    "gc10_det": ("GC10-DET", "gc10_det_copy_pairs.csv", "test_image", 460),
    "pcb_det": ("DsPCBSD+ (official split)", "dspcbsd_official_copy_pairs.csv", "validation_image", 2051),
}


def main(root: Path, output: Path | None) -> dict:
    params = json.loads((root / "leak_lists" / "audit_parameters.json").read_text(encoding="utf-8"))
    result = {}
    for key, (label, pairs_csv, query_col, n_eval) in BENCHMARKS.items():
        alpha, tau = params[label]["alpha"], params[label]["threshold_tau"]
        z = np.load(root / "additional_analyses" / f"{key}_comparison_pairs.npz", allow_pickle=False)
        fused = alpha * z["H"] + (1 - alpha) * z["D_B1"]
        rates = {}
        for role in ("fit", "test"):
            mask = (z["role"] == role) & (z["label"] == 0)
            hit = np.zeros(int(z["query"].max()) + 1, bool)
            np.logical_or.at(hit, z["query"][mask], fused[mask] >= tau)
            queries = np.unique(z["query"][z["role"] == role])
            rates[role] = float(hit[queries].mean())
        rows = list(csv.DictReader(open(root / "leak_lists" / pairs_csv, encoding="utf-8")))
        distance = [round(64 * (1 - float(r["hash_H"]))) for r in rows]
        flagged = {r[query_col] for r in rows}
        result[key] = {
            "evaluation_images": n_eval,
            "flagged_images": len(flagged),
            "per_query_false_flag_rate": rates,
            "expected_chance_flags_test_rate": rates["test"] * n_eval,
            "flagged_over_expected": len(flagged) / (rates["test"] * n_eval),
            "flagged_pairs": len(rows),
            "pairs_exact_phash_match": sum(d == 0 for d in distance),
            "pairs_phash_within_5_bits": sum(d <= 5 for d in distance),
        }
    text = json.dumps(result, indent=1)
    if output:
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    main(args.root, args.output)

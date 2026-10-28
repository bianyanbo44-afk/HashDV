"""Regenerate the exact confirmation copies from a benchmark image tree and released manifests."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from hashdv.features import read_rgb
from hashdv.intervention import index_files
from hashdv.synthetic import make_copy, seed_of


def main():
    import cv2
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["neu_det", "gc10_det", "pcb_det"], required=True)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--recipes", type=Path, default=Path("configs/recipes"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError("Use a new output directory")
    files = index_files(a.image_root, {".jpg", ".jpeg", ".png", ".bmp"})
    pools = {role: json.loads((a.recipes / f"{name}.json").read_text())["operator_sets"]
             for role, name in [("fit", "calibration_seen"), ("test", "test_unseen")]}
    with (a.data / "synthetic_copies" / f"{a.dataset}_copies.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if a.limit:
        rows = rows[:a.limit]
    a.output.mkdir(parents=True)
    for row in rows:
        name, role, copy = row["source_image"], row["role"], int(row["copy_index"])
        rng = np.random.default_rng(seed_of("L1b", a.dataset, role, name, copy))
        operators = pools[role][int(rng.integers(0, len(pools[role])))]
        severity = int(rng.integers(1, 6))
        image, meta = make_copy(read_rgb(files[name]), rng, operators, severity)
        expected = (float(row["crop_area_fraction"]), int(row["crop_x0"]), int(row["crop_y0"]),
                    float(row["rotation_deg"]), row["degradations"], int(row["severity"]))
        got = (meta["u"], meta["x0"], meta["y0"], meta["angle"], "+".join(meta["ops"]), meta["sev"])
        if got != expected:
            raise ValueError(f"Manifest mismatch for {row['copy_file']}: {got} != {expected}")
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("PNG encoding failed")
        encoded.tofile(str(a.output / row["copy_file"]))
    print(f"Generated and checked {len(rows)} confirmation copies")


if __name__ == "__main__":
    main()

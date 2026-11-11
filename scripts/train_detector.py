"""Run one paired YOLO11n arm with the manuscript's benchmark-specific recipe."""
import argparse
import hashlib
import json
from pathlib import Path
from hashdv.intervention import BENCHMARKS, training_recipe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    p.add_argument("--arm", choices=["full", "targeted", "random"], required=True)
    p.add_argument("--seed", type=int, choices=[2027, 3407, 8888], required=True)
    p.add_argument("--data-yaml", type=Path, required=True)
    p.add_argument("--weights", default="yolo11n.pt")
    p.add_argument("--device", default="0")
    p.add_argument("--project", type=Path, default=Path("runs"))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--version", choices=["current", "original"], default="current",
                   help="Use the submitted execution-matched recipe or the original recorded recipe")
    a = p.parse_args()
    recipe = training_recipe(a.benchmark, a.arm, a.seed, version=a.version)
    name = f"{a.benchmark}_{a.arm}_s{a.seed}"
    recipe.update(data=str(a.data_yaml.resolve()), device=a.device, project=str(a.project.resolve()), name=name)
    if a.dry_run:
        print(json.dumps(recipe, indent=2))
        return
    if (a.project / name).exists():
        raise FileExistsError("Use a new run directory to keep trained results separate")
    from ultralytics import YOLO
    YOLO(a.weights).train(**recipe)
    output = a.project / name
    checkpoint = output / "weights" / ("last.pt" if a.benchmark == "pcb_det" else "best.pt")
    record = {"version": a.version, "recipe": recipe, "checkpoint": checkpoint.name,
              "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (output / "done.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

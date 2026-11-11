"""Materialize the released split and deletion manifests without altering source data."""
import csv
import json
from pathlib import Path
import shutil

BENCHMARKS = {
    "neu_det": ("neu_det", ["crazing", "inclusion", "patches", "pitted_surface", "rolled_in_scale", "scratches"]),
    "gc10_det": ("gc10_det", ["punching_hole", "welding_line", "crescent_gap", "water_spot", "oil_spot", "silk_spot",
                              "inclusion", "rolled_pit", "crease", "waist_folding"]),
    "pcb_det": ("dspcbsd_official", ["SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO"]),
}


def manifest(data, benchmark, arm="targeted", seed=2027):
    data = Path(data)
    prefix, names = BENCHMARKS[benchmark]
    with (data / "splits" / (prefix + "_split.csv")).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if arm == "targeted":
        with (data / "splits" / (prefix + "_leakfree_train.csv")).open(encoding="utf-8", newline="") as stream:
            retained = {r["image"] for r in csv.DictReader(stream)}
        removed = {r["image"] for r in rows if r["split"] == "train" and r["image"] not in retained}
    elif arm == "random":
        if benchmark != "neu_det":
            raise ValueError("The released class-matched random deletion is defined for NEU-DET")
        record = json.loads((data / "random_deletion" / str(seed) / "manifest.json").read_text())
        removed = set(record["removed_images"])
    elif arm == "full":
        removed = set()
    else:
        raise ValueError("Arm must be full, targeted or random")
    selected = [r for r in rows if r["split"] != "train" or r["image"] not in removed]
    return selected, sorted(removed), names


def index_files(folder, suffixes):
    result = {}
    for path in Path(folder).rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            if path.name in result:
                raise ValueError(f"Duplicate filename in source tree: {path.name}")
            result[path.name] = path
    return result


def prepare(data, benchmark, arm, seed, images, labels, output):
    import yaml
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    rows, removed, names = manifest(data, benchmark, arm, seed)
    image_index = index_files(images, {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    label_index = index_files(labels, {".txt"})
    for row in rows:
        if row["image"] not in image_index or Path(row["image"]).with_suffix(".txt").name not in label_index:
            raise FileNotFoundError(f"Missing image or YOLO label: {row['image']}")
    for row in rows:
        filename, split = row["image"], row["split"]
        for kind, source in [("images", image_index[filename]),
                             ("labels", label_index[Path(filename).with_suffix(".txt").name])]:
            target = output / kind / split / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    test = "test" if benchmark != "pcb_det" else "validation"
    document = {"path": output.as_posix(), "train": "images/train", "val": "images/validation",
                "test": "images/" + test, "names": dict(enumerate(names))}
    (output / "data.yaml").write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    (output / "intervention.json").write_text(json.dumps({"benchmark": benchmark, "arm": arm, "seed": seed,
                                            "removed_images": removed, "n_train": sum(r["split"] == "train" for r in rows)},
                                           indent=2), encoding="utf-8")
    return output / "data.yaml"


def training_recipe(benchmark, arm, seed, *, version="current"):
    if benchmark not in BENCHMARKS or arm not in {"full", "targeted", "random"}:
        raise ValueError("Unknown benchmark or arm")
    if arm == "random" and benchmark != "neu_det":
        raise ValueError("Random-deletion control is NEU-DET only")
    if version not in {"current", "original"}:
        raise ValueError("Version must be current or original")
    pcb = benchmark == "pcb_det"
    original_full = version == "original" and not pcb and arm == "full"
    return {"epochs": 50 if pcb else 150, "imgsz": 640, "batch": 16 if pcb else 8,
            "workers": 0 if original_full else 1, "patience": 100 if pcb else 30,
            "seed": seed, "deterministic": True, "amp": True, "plots": original_full,
            "cache": False if pcb or original_full else "ram", "val": not pcb,
            "close_mosaic": 10, "optimizer": "auto"}

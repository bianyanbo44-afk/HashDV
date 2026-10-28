"""Create a private full/copy-removed/random-control dataset from the released manifests."""
import argparse
from pathlib import Path
from hashdv.intervention import BENCHMARKS, prepare


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    p.add_argument("--arm", choices=["full", "targeted", "random"], required=True)
    p.add_argument("--seed", type=int, choices=[2027, 3407, 8888], default=2027)
    p.add_argument("--image-root", type=Path, required=True)
    p.add_argument("--label-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(prepare(a.data, a.benchmark, a.arm, a.seed, a.image_root, a.label_root, a.output))


if __name__ == "__main__":
    main()

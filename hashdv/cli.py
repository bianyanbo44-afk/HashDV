"""Command-line entry points; model packages are loaded only for image inference."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="HashDV instance verification and leakage measurement")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("calibrate", help="Fit fusion and audit threshold on saved fit pairs")
    p.add_argument("pairs", type=Path)
    p.add_argument("--dense-key", default="D_B1")
    p.add_argument("--output", type=Path)
    p = sub.add_parser("audit", help="Find train/evaluation image matches")
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--test-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--device")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--cache-images", type=int, default=128)
    p = sub.add_parser("reproduce", help="Verify released scores, subset AP and deletion contrasts on CPU")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--bootstrap", action="store_true", help="Also recompute all 1,000-resample Table VI and NEU random-control intervals")
    args = parser.parse_args()
    if args.command == "calibrate":
        from .calibration import fit_pairs
        result = fit_pairs(args.pairs, args.dense_key)
        text = json.dumps(result, indent=2)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
    elif args.command == "audit":
        from .audit import audit
        config = json.loads(args.config.read_text(encoding="utf-8"))
        result = audit(args.train_dir, args.test_dir, args.output, config, args.checkpoint, args.device,
                       args.batch, args.cache_images)
        print(json.dumps({k: v for k, v in result.items() if k not in {"L", "U", "R"}}, indent=2))
    else:
        from .verification.verify_saved_results import main as scores
        from .verification.verify_control_ap import main as control
        scores(args.data)
        control(args.data, bootstrap=args.bootstrap)

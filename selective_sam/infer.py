"""Single-box inference with the published baseline-hint selector."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import yaml

from .corruption import corrupt_depth
from .data import decode_depth
from .infer_baseline import SelectivePredictorBaseline
from .samples import read_manifest


def assigned_manifest_damage(depth, valid, row):
    """Replay the one damaged depth variant assigned before model evaluation."""
    variants = [variant for variant in row["depth_variants"]
                if variant["condition"] != "clean"]
    if len(variants) != 1:
        raise ValueError("Expected exactly one assigned damaged variant")
    variant = variants[0]
    damaged_depth, damaged_valid = corrupt_depth(
        depth, valid, variant["condition"], variant["strength"], variant["seed"])
    return damaged_depth, damaged_valid, variant["condition"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final.yaml")
    parser.add_argument("--gate-config", default="configs/gate_baseline.yaml")
    parser.add_argument("--split", choices=("adapter_train", "gate_train", "val", "test"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--damaged", action="store_true",
                        help="With --split, replay that image's assigned damaged depth variant")
    parser.add_argument("--rgb")
    parser.add_argument("--depth")
    parser.add_argument("--box", nargs=4, type=float)
    parser.add_argument("--mode", choices=("baseline", "fusion", "selective"), default="selective")
    parser.add_argument("--output", default="results/single_prediction.png")
    args = parser.parse_args()
    if args.damaged and not args.split:
        parser.error("--damaged requires --split")
    config = yaml.safe_load(Path(args.config).read_text())
    if args.split:
        row = read_manifest(args.split)[args.index]
        rgb_path = Path(config["data"]["root"]) / row["rgb"]
        depth_path = Path(config["data"]["root"]) / row["depth"]
        box = row["targets"][args.target_index]["box"]
    else:
        if not args.rgb or not args.box:
            parser.error("Use --split or provide --rgb and --box")
        rgb_path, depth_path, box = Path(args.rgb), Path(args.depth) if args.depth else None, args.box
    with Image.open(rgb_path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    depth, valid = decode_depth(depth_path) if depth_path else (None, None)
    input_condition = "clean" if args.split else "provided_input"
    if args.damaged:
        depth, valid, input_condition = assigned_manifest_damage(depth, valid, row)
    predictor = SelectivePredictorBaseline(
        config_path=args.config, gate_config_path=args.gate_config)
    result = predictor.predict(rgb, depth, valid, box, args.mode)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result["mask"] * 255).save(args.output)
    print(json.dumps({"output": args.output, "mode": args.mode,
                      "input_condition": input_condition,
                      "adapter_used": result["adapter_used"], "gate_score": result["gate_score"],
                      "foreground_pixels": int(result["mask"].sum())}))


if __name__ == "__main__":
    main()

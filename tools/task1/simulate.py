"""Apply a Task 1 model package's runtime JSON and launch official simulation."""

import argparse
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import sys

import yaml

REPO = Path(__file__).resolve().parents[2]
CAMERAS = ("observation.images.head_cam_h", "observation.images.wrist_cam_r")
MODEL_FILES = ("model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json")


def prepare(phase, bundle, runtime_path, output_dir, episodes, max_steps):
    """Create a run copy of model settings and link its immutable weights."""
    output = bundle.resolve()
    if (output / "output").is_dir():
        output /= "output"
    model = output / "pretrained_model"
    config = json.loads((model / "config.json").read_text(encoding="utf-8"))
    if (
        config["type"] != "act"
        or config["input_features"]["observation.state"]["shape"] != [18]
        or config["output_features"]["action"]["shape"] != [18]
        or any(config["input_features"].get(key, {}).get("shape") != [3, 480, 640] for key in CAMERAS)
    ):
        raise ValueError("Expected the Task 1 ACT package with 18 state and action features and two RGB cameras")

    runtime = json.loads((runtime_path or output / "deploy_act_runtime.json").read_text(encoding="utf-8"))
    steps = runtime["n_action_steps"]
    ensemble = runtime["temporal_ensemble_coeff"]
    hz = runtime["action_hz"]
    if type(steps) is not int or not 1 <= steps <= config["chunk_size"]:
        raise ValueError("n_action_steps must be an integer within the checkpoint's chunk size")
    if ensemble is not None and (type(ensemble) not in (float, int) or not math.isfinite(ensemble) or steps != 1):
        raise ValueError("Temporal ensembling requires a finite coefficient and n_action_steps=1")
    if type(hz) not in (float, int) or not math.isfinite(hz) or hz <= 0:
        raise ValueError("action_hz must be a positive finite number")

    files = set(MODEL_FILES)
    for name in MODEL_FILES[1:]:
        processor = json.loads((model / name).read_text(encoding="utf-8"))
        for step in processor["steps"]:
            if "state_file" in step:
                filename = step["state_file"]
                if Path(filename).name != filename or Path(filename).suffix != ".safetensors":
                    raise ValueError(f"Invalid processor state filename in {name}")
                files.add(filename)
    if (model / "train_config.json").is_file():
        train = json.loads((model / "train_config.json").read_text(encoding="utf-8"))
        expected = 438600 if phase == "act1" else 259000
        if train["steps"] != expected:
            raise ValueError(f"{phase} expects the {expected}-step checkpoint")
    for name in files:
        if not (model / name).is_file():
            raise FileNotFoundError(model / name)

    run_name = f"{phase}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    run_model = output_dir.resolve() / run_name / "pretrained_model"
    run_model.mkdir(parents=True)
    for name in sorted(files):
        (run_model / name).symlink_to((model / name).resolve())
    config.update(n_action_steps=steps, temporal_ensemble_coeff=ensemble, pretrained_backbone_weights=None)
    (run_model / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    profile = REPO / "configs/deploy/task1/sim.yaml"
    deploy = yaml.safe_load(profile.read_text(encoding="utf-8"))
    deploy["env"]["ros_rate"] = hz
    deploy["inference"].update(
        pretrained_path=str(run_model), eval_episodes=episodes, max_episode_steps=max_steps,
    )
    config_path = run_model.parent / "deploy.yaml"
    config_path.write_text(yaml.safe_dump(deploy, sort_keys=False), encoding="utf-8")
    return config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("act1", "act2"))
    parser.add_argument("--bundle", required=True, type=Path, help="Extracted model directory containing output/")
    parser.add_argument("--runtime", type=Path, help="Override the bundle's deploy_act_runtime.json")
    parser.add_argument("--run-dir", type=Path, default=REPO / "outputs/task1_sim")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=470)
    parser.add_argument("--prepare-only", action="store_true", help="Write and validate the run config without connecting to ROS")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("--episodes and --max-steps must be positive")

    config_path = prepare(args.phase, args.bundle, args.runtime, args.run_dir, args.episodes, args.max_steps)
    from kuavo_deploy.config import load_kuavo_config
    config = load_kuavo_config(str(config_path), task="auto_test")
    print(f"Task 1 {args.phase} profile: {config_path}", flush=True)
    if args.prepare_only:
        return

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    if sys.version_info >= (3, 12):
        from rosgraph.roslogging import RospyLogger

        def find_caller(self, stack_info=False, stacklevel=1):
            return logging.Logger.findCaller(self, stack_info, stacklevel + 1)

        RospyLogger.findCaller = find_caller

    from tools.task1.recorder import install_async_recorder
    install_async_recorder()
    from kuavo_deploy.src.scripts.script_auto_test import ArmMove
    ArmMove(config).auto_test()


if __name__ == "__main__":
    main()

"""Train an ACT policy against a prepared task-one LeRobot v3 dataset."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
LEROBOT_SRC = REPO / "third_party/lerobot/src"
PHASES = {"act1": (52623, 438600), "act2": (31079, 259000)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--dataset", required=True, type=Path, help="Prepared 201-episode dataset directory")
    parser.add_argument("--output", required=True, type=Path, help="New checkpoint output directory")
    parser.add_argument("--check", action="store_true", help="Validate the dataset metadata and config without training")
    args = parser.parse_args()

    if not (LEROBOT_SRC / "lerobot/scripts/lerobot_train.py").is_file():
        parser.exit(2, "Run git submodule update --init third_party/lerobot from the repository.\n")
    dataset, output = args.dataset.resolve(), args.output.resolve()
    info = json.loads((dataset / "meta/info.json").read_text(encoding="utf-8"))
    frames, steps = PHASES[args.phase]
    if (info["total_episodes"], info["total_frames"]) != (201, frames):
        parser.error(f"{args.phase} expects 201 episodes and {frames} frames")
    if any(info["features"][key]["shape"] != [18] for key in ("observation.state", "action")):
        parser.error("Task 1 requires 18 state features and 18 action features")
    if output.exists():
        parser.error(f"Choose a new output directory: {output}")

    path = REPO / "configs" / "train" / "task1" / f"{args.phase}.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["dataset"]["root"] = str(dataset)
    config["output_dir"] = str(output)
    if config["steps"] != steps:
        parser.error(f"{args.phase} configuration must record {steps} steps")

    with tempfile.TemporaryDirectory(prefix="openlet-task1-train-") as temp:
        resolved = Path(temp) / "training.json"
        resolved.write_text(json.dumps(config, indent=2), encoding="utf-8")
        if args.check:
            sys.path.insert(0, str(LEROBOT_SRC))
            import draccus
            from lerobot.configs.train import TrainPipelineConfig
            from lerobot.policies.act.configuration_act import ACTConfig

            parsed = draccus.parse(TrainPipelineConfig, config_path=str(resolved), args=[])
            parsed.validate()
            if not isinstance(parsed.policy, ACTConfig):
                parser.error("Task 1 requires an ACT policy")
            print(f"{args.phase}: {frames} frames, {parsed.steps} steps, batch {parsed.batch_size}", flush=True)
            return
        env = os.environ.copy()
        python_paths = [str(REPO), str(LEROBOT_SRC)]
        if env.get("PYTHONPATH"):
            python_paths.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(python_paths)
        env.setdefault("WANDB_MODE", "disabled")
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("MKL_NUM_THREADS", "4")
        subprocess.run(
            [sys.executable, "-u", "-m", "lerobot.scripts.lerobot_train", f"--config_path={resolved}"],
            cwd=REPO, env=env, check=True,
        )


if __name__ == "__main__":
    main()

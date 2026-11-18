#!/usr/bin/env python3
"""Task2手部每侧11维压缩为1个开关；统计与元数据读写复用官方函数。"""
import os
from pathlib import Path
import shutil
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LeTools-Learning/third_party/lerobot/src"))

from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_tools import _write_parquet
from lerobot.datasets.io_utils import load_info, load_stats, write_info, write_stats

SOURCE = ROOT / "dataset-task2-filtered200"
TARGET = ROOT / "dataset-task2-filtered200-binary-scalar"
THRESHOLD = 1.55 / 1.57  # 采用新版公式1.0+1.1*0.5=1.55作为关节角阈值，其中0.5是逻辑闭合度
FEATURES = ("observation.state", "action")

def main():
    TARGET.mkdir(parents=True, exist_ok=False)
    info = load_info(SOURCE)
    stats = load_stats(SOURCE)
    retained, hands = {}, {}
    for key in FEATURES:
        feature = info["features"][key]
        names = feature["names"]
        old_names = next(iter(names.values())) if isinstance(names, dict) else names
        # 保留非手部维度，以及左右手的第1关节。
        retained[key] = [i for i, name in enumerate(old_names)
                         if not name.startswith(("left_sg100_", "right_sg100_"))
                         or name in ("left_sg100_1", "right_sg100_1")]
        new_names = [old_names[i] for i in retained[key]]
        hands[key] = [new_names.index(f"{side}_sg100_1") for side in ("left", "right")]
        for side, index in zip(("left", "right"), hands[key]):
            new_names[index] = f"{side}_hand_closed"
        feature["shape"] = (len(new_names),)
        feature["names"] = {next(iter(names)): new_names} if isinstance(names, dict) else new_names

    shutil.copytree(SOURCE / "meta", TARGET / "meta")
    shutil.copytree(SOURCE / "videos", TARGET / "videos", copy_function=os.link)
    write_info(info, TARGET)
    metadata = LeRobotDatasetMetadata("openlet/task2_binary", root=TARGET)
    episode_stats = {}
    for path in sorted((SOURCE / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        frame = table.to_pandas()
        ids = table["episode_index"].to_numpy()
        arrays = {}
        for key in FEATURES:
            values = np.stack(table[key].combine_chunks().to_numpy(zero_copy_only=False))
            values = values[:, retained[key]].copy()
            values[:, hands[key]] = values[:, hands[key]] > np.float32(THRESHOLD)
            arrays[key] = values
            frame[key] = list(values)
        for episode in np.unique(ids):
            episode_stats[int(episode)] = compute_episode_stats(
                {key: values[ids == episode] for key, values in arrays.items()},
                {key: info["features"][key] for key in FEATURES},
            )
        output = TARGET / path.relative_to(SOURCE)
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_parquet(frame, output, metadata)

    # 同步每条轨迹的统计；图像与其他字段沿用原数据。
    for path in sorted((SOURCE / "meta/episodes").rglob("*.parquet")):
        table = pq.read_table(path)
        rows = table.to_pylist()
        for row in rows:
            for key, feature_stats in episode_stats[row["episode_index"]].items():
                for name, value in feature_stats.items():
                    row[f"stats/{key}/{name}"] = value.tolist()
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema),
                       TARGET / path.relative_to(SOURCE), compression="snappy")
    stats.update(aggregate_stats(list(episode_stats.values())))
    write_stats(stats, TARGET)
    print(f"完成：{len(episode_stats)}条轨迹二值化处理，处理后状态16维/动作19维，输出到 {TARGET}")

if __name__ == "__main__":
    main()
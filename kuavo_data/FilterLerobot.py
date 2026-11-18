#!/usr/bin/env python3
"""按原始轨迹编号筛选LeRobot数据；复用官方重编号和元数据处理。"""
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "LeTools-Learning/third_party/lerobot/src"))

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.datasets.dataset_tools import (
    _copy_and_reindex_data,
    _copy_and_reindex_episodes_metadata,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SOURCE = ROOT / "dataset-task2-original"
TARGET = ROOT / "dataset-task2-filtered200"
# 填写要保留的原始轨迹编号，从0开始；例如 [0, 3, 8]。
SELECTED_EPISODES = [
    0, 2, 3, 6, 7, 8, 9, 13, 14, 17, 19, 20,
    21, 26, 27, 31, 32, 33, 35, 37, 38, 40, 42, 44,
    45, 47, 49, 52, 55, 56, 59, 62, 66, 67, 68, 74,
    77, 80, 81, 82, 85, 86, 87, 92, 95, 96, 99, 100,
    101, 103, 104, 105, 106, 112, 115, 116, 117, 118, 119, 120,
    121, 122, 123, 124, 127, 130, 138, 139, 140, 142, 143, 144,
    145, 151, 155, 156, 157, 158, 162, 177, 178, 179, 180, 182,
    183, 185, 186, 190, 191, 192, 193, 196, 199, 200, 201, 204,
    205, 208, 209, 214, 215, 216, 217, 218, 221, 222, 227, 237,
    239, 240, 243, 244,
    251, 252, 253, 255, 256, 258, 259, 260, 261,
    262, 263, 264, 265, 266, 268, 269, 271, 272, 276,
    280, 281, 282, 284, 285, 286, 288, 289, 290, 292,
    293, 294, 295, 296, 297, 298, 299, 300, 302, 303,
    304, 305, 307, 308, 309, 310, 311, 312, 314, 316,
    317, 318, 320, 322, 323, 324, 326, 327, 329, 330,
    331, 333, 334, 335, 336, 337, 338, 340, 341, 342,
    343, 344, 346, 347, 348, 349, 350, 351, 353, 354,
    356, 358, 359, 360, 361, 363, 364, 365, 366,
    367, 368, 369, 370, 372, 373, 374, 376, 377, 378,
    379, 380, 382, 384, 385, 386, 387, 388, 389, 390,
    391, 392, 393, 395, 396, 397, 400, 402, 403,
    404, 405, 408, 409, 410, 411, 412, 413, 414, 415,
    416, 417, 419, 420, 421, 422, 423, 424, 426, 427,
    428, 430, 431, 432, 433, 434, 437, 439, 441, 442,
    443, 444, 447, 448, 449, 452, 453, 454, 455, 457,
    458, 462, 463, 464, 466, 467, 468, 469, 470, 471,
    472, 473, 474, 476, 477, 478, 479, 481, 482, 483,
    484, 485, 486, 487, 488, 489, 490, 491, 492, 495,
    496, 497, 499,
][0:200]  # 仅保留前200条轨迹

def main():
    if TARGET.exists():
        raise FileExistsError(f"目标目录已存在，请修改TARGET：{TARGET}")
    source = LeRobotDataset("openlet/task2", root=SOURCE, video_backend="pyav")
    destination = LeRobotDatasetMetadata.create(
        repo_id="openlet/task2_filtered", root=TARGET,
        fps=source.meta.fps, robot_type=source.meta.robot_type,
        features=source.meta.features, use_videos=True,
        chunks_size=source.meta.chunks_size,
    )
    mapping = {old: new for new, old in enumerate(sorted(set(SELECTED_EPISODES)))}
    shutil.copytree(SOURCE / "videos", TARGET / "videos", copy_function=os.link)
    videos = {
        new: {key: value for key, value in source.meta.episodes[old].items()
              if key.startswith("videos/")}
        for old, new in mapping.items()
    }
    data = _copy_and_reindex_data(source, destination, mapping)
    _copy_and_reindex_episodes_metadata(source, destination, mapping, data, videos)
    print(f"完成：筛选出{len(mapping)}条轨迹，输出到 {TARGET}")


if __name__ == "__main__":
    main()

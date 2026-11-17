"""Task3 deployment boundary: postprocessed physical actions in, ROS-order actions out."""
import json
from pathlib import Path
import numpy as np


class ActionAdapter:
    def __init__(self, poses_path, stats_path=None):
        self.spec = json.loads(Path(poses_path).read_text(encoding="utf-8"))
        self.poses = np.asarray([self.spec["gestures"][str(i)]["pose_rad"] for i in (0, 1)])
        self.stats = None if stats_path is None else json.loads(Path(stats_path).read_text())["action"]
        if self.poses.shape != (2, 11) or self.spec["unit"] != "rad":
            raise ValueError("Invalid gesture table")

    def restore(self, physical_action):
        """Call AFTER LeRobot postprocessor/temporal fusion; never unnormalize twice."""
        a = np.asarray(physical_action)
        if a.ndim == 0 or a.shape[-1] != 11 or not np.isfinite(a).all():
            raise ValueError("Expected finite (..., 11) physical actions")
        label = (a[..., 7] >= 0.5).astype(np.int64)
        return np.concatenate([a[..., :7], self.poses[label], a[..., 8:]], axis=-1)

    def restore_normalized(self, normalized_action):
        """Only for raw MEAN_STD outputs without LeRobot action postprocessing."""
        if self.stats is None:
            raise ValueError("Provide training stats for raw normalized outputs")
        a = np.asarray(normalized_action)
        if a.ndim == 0 or a.shape[-1] != 11 or not np.isfinite(a).all():
            raise ValueError("Expected finite (..., 11) normalized actions")
        return self.restore(a * np.asarray(self.stats["std"]) + np.asarray(self.stats["mean"]))

    def restore_tensor(self, physical_action):
        """Preserve torch device/dtype; caller handles final robot transport."""
        import torch
        a = physical_action
        if not isinstance(a, torch.Tensor) or a.ndim == 0 or a.shape[-1] != 11:
            raise ValueError("Expected torch (..., 11) physical actions")
        if not a.is_floating_point() or not torch.isfinite(a).all():
            raise ValueError("Expected finite floating-point actions")
        poses = torch.as_tensor(self.poses, device=a.device, dtype=a.dtype)
        label = (a[..., 7] >= 0.5).long()
        return torch.cat((a[..., :7], poses[label], a[..., 8:]), dim=-1)

    def restore_normalized_tensor(self, normalized_action):
        import torch
        a = normalized_action
        if self.stats is None:
            raise ValueError("Provide training stats for raw normalized outputs")
        if not isinstance(a, torch.Tensor) or a.ndim == 0 or a.shape[-1] != 11:
            raise ValueError("Expected torch (..., 11) normalized actions")
        mean = torch.as_tensor(self.stats["mean"], device=a.device, dtype=a.dtype)
        std = torch.as_tensor(self.stats["std"], device=a.device, dtype=a.dtype)
        return self.restore_tensor(a * std + mean)

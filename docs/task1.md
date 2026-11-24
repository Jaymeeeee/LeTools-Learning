# OpenLET 任务一

本分支在 Jaymeeeee 的 LeTools-Learning fork 中维护任务一的 ACT 训练和仿真。ACT1 独立学习抓取、放置和松手，ACT2 独立学习拨杆滑箱。

| 阶段 | 轨迹 | 帧数 | 训练轮数 | 训练步数 | 模型包 |
| --- | ---: | ---: | ---: | ---: | --- |
| ACT1 | 201 | 52,623 | 200 | 438,600 | `ACT1_438600_output_minimal.zip` |
| ACT2 | 201 | 31,079 | 200 | 259,000 | `ACT2_259000_output_minimal.zip` |

## 训练

在已配置依赖的 Python 环境中执行。数据路径指向已整理好的 LeRobot v3 数据集，使用训练时像素无损优化后的版本可复现解码性能。两个配置都保留批大小 24、学习率 1e-5、在线颜色增强和长度为 100 的动作块。训练版本使用仓库锁定的 LeRobot 子模块。

```bash
git clone -b task1 https://github.com/Jaymeeeee/LeTools-Learning.git
cd LeTools-Learning
git submodule update --init third_party/lerobot

python -m tools.task1.train act1 --dataset /path/to/act1_dataset --output outputs/train/task1_act1
python -m tools.task1.train act2 --dataset /path/to/act2_dataset --output outputs/train/task1_act2
```

命令后加 `--check` 可在训练前检查元数据和配置。实际训练会初始化 ImageNet 主干，离线训练时应预先把 ResNet18 权重放入 Torch 缓存。最终检查点会保存到各阶段输出目录。

## 仿真

分别解压两个 ZIP。每个包内的 `output/pretrained_model/` 包含权重与该阶段的处理器，新模型权重及数据集通过另外的文件交付。

在官方任务一仿真端启动 model 模式。部署端使用已经配好的 GPU 和 ROS 环境，从本仓库根目录运行以下命令。若使用两个容器，部署端要能访问仿真 ROS Master，并能看到解压后的模型和本仓库。

```bash
export PYTHONPATH="$PWD:$PWD/third_party/lerobot/src${PYTHONPATH:+:$PYTHONPATH}"
python -m tools.task1.simulate act1 --bundle /path/to/extracted_act1
python -m tools.task1.simulate act2 --bundle /path/to/extracted_act2
```

ACT1 使用任务一原始起点。ACT2 单独测试时需要仿真端在每次复位后进入拨杆前的状态，包括物块已经入盒、拨杆尚未拉动和合适的右手姿态。官方常规任务一复位会进入抓取起点，因此应先准备阶段二复位场景。两个命令分别加载独立模型，阶段切换由评测侧控制。

入口会读取解压包中 `output/deploy_act_runtime.json`。默认是 100 步、10 Hz、关闭时序集成，开启集成时将 `n_action_steps` 设为 1。也可以用 `--runtime` 指定另一个 JSON。部署配置为右臂 7 维和 SG100 右手 11 维，底盘控制关闭，使用头部和右腕双视角。头部初始化采用已测 ACT1 参数 `[0.0, 0.209]`，更改相机俯仰角可以调整 `configs/deploy/task1/sim.yaml`。

每次运行的实际部署参数保存在 `outputs/task1_sim/`，仿真视频和评测日志由官方入口写入 `outputs/eval/`。用 `--prepare-only` 可以生成并检查实际配置，随后再运行仿真。

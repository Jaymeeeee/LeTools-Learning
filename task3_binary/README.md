# Task3 二值化代码与排查入口

这是当前比赛实验的代码交接分支，方便复现并查找抓取偏差。不是已验证成功率的算法发布。权重和数据不在Git中。

## 先看哪些文件

- `convert_dataset.py`：原始21维动作变为11维。保留右臂7维、手势标签1维、底盘3维；状态仍为18维。
- `poses.json`：训练时固定的两种手形及标签规则。`poses_scaled157.json`：部署发送使用的手形，已乘1.57，不要再乘。
- `action_adapter.py`：模型输出先反归一化，再按0.5阈值还原为21维，之后发送。
- `selection.json`：实际160条训练、40条验证名单；验证集没有参与统计拟合。
- `stats.json`、`train_config.json`：本轮训练统计与参数。代码配置路径改为相对路径；换机器需设置数据和输出位置。
- `kuavo_deploy/utils/inference_overrides.py`：步数设置、手部反馈除1.57、右臂第6轴+0.20rad。无根目录deployment_tuning.json时不启用。
- `reference/`：实际迁移恢复脚本原件，保留历史绝对路径和当时中间结论，仅供追溯，不作为运行入口。40条验证名单后来已经找回，以selection.json为准。

## 数据处理

只适用于本轮旧版task3_loop_feed（490bba041cfac362a74ef06febbc5e78d126c9bd）的LeRobot v3数据结构，不用于任意新版数据。

```bash
python task3_binary/convert_dataset.py --source /data/task3_source_490bba0 --out /data/task3_binary_490bba0
```

输出目录必须不存在。只删除开头action[11]<0.8的初始化段，后续低值帧保留；action[11]<1.027070015668869标为1，否则为0。视频不重编码，通过时间戳偏移对齐，并建立到源视频的符号链接，因此源视频必须保留。需要numpy、pandas、pyarrow。

500条共179034帧；选定200条共71513帧，其中训练57212、验证14301。统计文件来自当时冻结产物；转换会独立核对训练集state/action统计，图像统计沿用冻结值。不要将这些统计直接套在新数据上。

训练：ACT、ResNet18，batch16、workers8、357600优化步，约100等效epoch。具体参数见train_config.json。可通过prepare_configs.py生成其他batch/workers候选；需在训练环境安装LeRobot后使用。

## 推理

在仓库根目录操作，使用已有的匹配ROS、LeRobot和CUDA环境。先安装官方仓库依赖，并放好小包内的pretrained_model。

```bash
cp task3_binary/deploy.example.yaml configs/deploy/deploy.yaml
cp task3_binary/deployment_tuning.example.json deployment_tuning.json
# 修改deploy.yaml中的权重位置、评测轮数等；先做一轮测试
python task3_binary/set_inference.py --steps 50 --hz 10
python -u kuavo_deploy/eval.py
```

示例为50步、10Hz、手部反馈除1.57及腕部+0.20rad；手形表已乘1.57。部署JSON仅在下一次加载策略时生效。停止旧推理、按官方流程重置仿真后再启动新推理；日志queue_maxlen应与步数一致。

根目录deployment_tuning.json是本地运行配置，不提交；要停用此模块可移开该文件。不要对连续动作基线启用二值化配置。

+0.20rad是单姿态跟踪测试得到的实验补偿，不是全姿态标定。根目录没有该JSON时，普通策略加载保持原行为。本分支未包含本地观看专用的固定seed、跳过末轮reset等改动。

## 当前问题和检查

多次单场景运行仍有入环偏侧、空抓后转移、抬手高度不足；没有证明某组步数抓取率更高，也没有保证非0分。初查两相机名称映射、RGB转换和输入尺寸没有发现错误，尚未完成相机内外参、同姿态画面对照以及全轨迹腕部标定。

```bash
# 在匹配的推理环境中：只检查边界运算，不发送机器人动作
python task3_binary/test_contract.py
```

检查涵盖11→21维还原、标签阈值、两份手形1.57关系、反馈换算、50步队列、腕部只改第6轴和输入不被原地修改。原数据完整转换和新机器闭环测试需在相应环境另做；本分支不是重新训练的模型。

## 分支来源与验证记录

分支基于我们原克隆的6d20026，保存本轮实际部署改动。师兄仓库main的8647bc5含另一套SG100修改；本分支未与其合并，避免改变实验含义。请按文件比较，不能直接覆盖师兄main。

边界测试已在L40现有Python/ROS依赖环境运行，3项通过。测试不驱动机器人；尚未在全新机器验收闭环或成功率。

参数化转换入口已在L40源数据上完整运行：500条、179034帧、训练57212帧、验证14301帧；重算的训练state/action均值、标准差和极值与冻结统计一致。未启动训练。

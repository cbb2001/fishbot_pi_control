# 架上模拟 t4 分段预测与间歇静止校正

已完成实机验证：t4=10秒，180秒内恰好18次校正，段内没有静止更新。[中文结果表](logs/imu_static_20261031/PERIODIC_RESULTS_zh.txt)与[对比曲线](logs/imu_static_20261031/periodic_analysis_c/periodic_comparison.png)已保存。零速加局部偏置方案在本轮的单段位移P95约16.3cm，全程位置偏差最大约19.5cm；历史数据的对应结果更差，详见结果表。

之后完成了 [连续40分钟实机采集与评估](logs/imu_static_20261031/run_40min_20261031_1435/RESULTS_zh.txt)：480007帧，239次间歇校正，无缺帧或CRC错误。长时统计和短时统计应分开查看。

本实验在树莓派上读取真实 IMU，鱼体全程固定在架子上，不初始化 PWM、不发任何舵机命令。它模拟导航的估计时序，不执行实际下潜、上浮或读入真实 UWB 位置。

当前 `robot.yaml` 为 `t1=5 s`、`t2=5 s`、`t4=10 s`，加速度两级低通各为 `0.2 s`。实验先等待 5 秒，再用接下来 5 秒做启动零偏标定。之后每经过 t4 秒，只在边界施加一次校正。在两个边界之间，虽然鱼实际上静止，算法也不能获得任何零速或静止偏置更新。

边界一次性使用此前 2 秒样本的统计量。若窗口不安静，这次更新被跳过，不会在下一段中补更新。各 t4 对照重新从同一启动窗口初始化，不共享未来估计；历史训练文件仅提供固定噪声初值，不提供本次加速度偏置。

## 五个独立对照

| 日志名称 | t4 边界操作 | 绝对位置直接归零 |
|---|---|---|
| `free_vendor` | 不校正 | 否 |
| `periodic_velocity` | 速度置零，保留原偏置和累计位置 | 否 |
| `periodic_velocity_bias` | 速度置零，使用该边界窗口重新估计有效体坐标偏置 | 否 |
| `periodic_surface_anchor` | 模拟获得原点位置，位置与速度置零，保留原偏置 | 是，显式的理想位置观测 |
| `periodic_eskf_zupt` | 15维 ESKF 仅进行一次零速/零角速观测更新 | 否；卡尔曼交叉协方差仍可修正位置估计 |

前四个对照使用设备输出的当前四元数去重力、两级低通和梯形积分，其水平传播与导航 `InertialEstimator` 一致，已有数值对照测试。上浮校正后清空两级滤波器，沿用实际导航的策略。ESKF 使用去偏陀螺传播姿态；其偏置与姿态只在周期边界通过观测更新，段内仅预测。

这里的三维积分用于诊断；原导航的垂向状态由压力约束，不能用本实验的 Z 双积分代替。世界航向未额外对齐泳池坐标，但水平向量范数不受该固定航向旋转影响。

## 运行

```bash
cd /home/fish/fishbot_pi_control
.venv/bin/python scripts/capture_imu_static_20261031.py \
  --duration 190 --tid-modulus 60000 --periodic-t4 10 \
  --bench-static-confirmed \
  --output logs/imu_static_20261031/session_periodic_c
```

190 秒包括 10 秒启动及 180 秒分段实验，预期 18 个 10 秒边界。`periodic_live.jsonl` 保存树莓派上实时计算的初始化、每次校正前后状态和 10 Hz 状态曲线。`result.json` 单独记录边界数、实际接受次数和 ESKF 更新次数。

离线对比 5、10、20、30 秒，并核对 t4=10 的实机逐边界状态：

```bash
python scripts/analyze_imu_periodic_static_20261031.py \
  logs/imu_static_20261031/session_periodic_c \
  --t4 5 10 20 30 --output logs/imu_static_20261031/periodic_analysis_c
```

输出包含 `report.json`、`report.txt`、`cycles.csv`、`events_t4_*.json`、主间隔的状态曲线 JSON 和 `periodic_comparison.png`。历史第二轮数据也按同一规则重放，结果在 `periodic_replay_b`；其启动后有效预测时间约 170 秒。

## 如何看结果

重点是校正前的水平速度、每段新增位移，以及整个实验中累计位置的最大偏差。不能只报告最后一次校正之后的位置，否则位置归零组必然显得“无误差”。统计表给出完整段的平均、P95、最大误差；不足 t4 的末尾部分不算完整段，但仍计入全程最大位置偏差。

`periodic_surface_anchor` 使用了额外已知位置，代表理想化的上浮定位，不能与仅靠静止零速更新宣称同样的信息条件。静止只说明真实速度为零，并不能一般地测得当前位置。

**本实验边界前的 2 秒确实静止，因此可以作为校正窗口。真实游动时，不能把下潜段末尾的运动数据当作静止样本：应在上浮后独立确认停稳，再采集静止窗口，或使用真实 UWB/视觉观测。实际校正间隔还要加上上浮与取数耗时。** 本次没有更改原导航流程或自动启用架上静止假设。

测试覆盖：段内不更新且每边界只更新一次；零速校正保留累计位置、位置锚定单独清零；偏置只在边界利用过去窗口变化；扰动边界被跳过；与原导航水平积分一致。

```bash
python -m unittest discover -s tests -p test_imu_periodic_static_20261031.py -v
```

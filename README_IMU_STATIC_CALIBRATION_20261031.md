# 2026-10-31 架上静止 IMU 标定与积分漂移

**本次实测未得到可直接用于运动的可靠固定偏置。** 两轮共 156010 帧；验证段选出的均值偏置在最终留出及独立复测中均使水平位移漂移增大。[实机数值与结论](logs/imu_static_20261031/RESULTS_zh.txt)、[第一轮曲线](logs/imu_static_20261031/analysis_a_verified_clock/drift_comparison.png)、[独立复测曲线](logs/imu_static_20261031/analysis_b/drift_comparison.png) 已保存。`config/imu_static_candidate_20261031.json` 是实验候选，不是已通过运动验证的标定文件。

后续按用户要求另做了“每隔 t4 只校正一次”的实机实验，见 [间歇静止校正说明](README_IMU_PERIODIC_STATIC_20261031.md)。不要把前面的持续零速更新结果与间歇校正结果混用。

本实验仅打开 IMU 串口 `/dev/ttyAMA4`，复用导航框架的 `IMUWorker`、Yesense 解码器及 TID 时间戳重建。采集入口不导入导航任务、舵机执行器或 PWM，并用 Python 审计钩子拒绝打开其他硬件设备。没有修改 `robot.yaml` 或现有导航行为。

## 复现实机采集

鱼体需由操作者确认静止固定。运行以下程序本身不会给舵机发命令。输出目录必须不存在。

```bash
cd /home/fish/fishbot_pi_control
.venv/bin/python scripts/capture_imu_static_20261031.py \
  --duration 600 --tid-modulus 60000 --output logs/imu_static_20261031/session_a
```

`raw_imu.jsonl` 保留原始串口帧、完整加速度、角速度、四元数、欧拉角、帧号、采样重建时间戳及主机接收时间；`metadata.json` 保留配置和源码摘要；`result.json` 保留实际打开的设备、退出原因及 `executor_started=false`。当前 67 字节报文只有加速度、角速度、欧拉角和四元数，不含温度，`temperature_c=0` 是旧解码器的缺省值，不能用于温漂拟合。

**实测发现帧号回绕差异**：本机固件的 TID 为 `1..60000`，第一轮两次出现 `60000→1`。旧采集器按 65536 取模，累计误报 11072 个缺帧，并把回绕附近时间戳压缩。本次给 `IMUWorker` 和 `SensorManager` 增加可配置的 `tid_modulus`；默认保留 65536 以兼容其他设备。第二轮明确设置 60000。以后使用本台 IMU 的导航会话可在 `sensors.imu` 下设置 `tid_modulus: 60000`；本次未改动原 `robot.yaml`。

离线积分先逐帧核对连续性、识别实际 `60000→1` 回绕，再采用帧序号 / 配置 200 Hz 的时基，并与主机接收总时长交叉核对。原始重建时间戳仍完整保留。这不能把真正丢失的帧变成有效数据；发现其他缺帧、反序或主机时长不一致会拒绝分析。

## 标定、选择与验证

```bash
python scripts/analyze_imu_static_20261031.py \
  logs/imu_static_20261031/session_a \
  --output logs/imu_static_20261031/analysis_a
```

600 秒采集按时间分成：前 60 秒舍弃；60–240 秒训练；240–360 秒选择；360–600 秒最终留出测试。训练比较启动 5 秒均值、180 秒均值、1 秒分块均值的中位数、训练末 60 秒均值，以及静止 ESKF 学到的加速度偏置；另比较 180 秒均值加两级 0.2 秒低通。所有参数在最终测试前冻结。选择依据是验证段独立 20/60 秒窗口末端水平位移误差的 RMS，原始基线也参与选择，避免强制采用更差的校准。

第二次独立采集使用已有参数，绝不重新拟合：

```bash
.venv/bin/python scripts/capture_imu_static_20261031.py \
  --duration 180 --tid-modulus 60000 \
  --calibration config/imu_static_candidate_20261031.json \
  --output logs/imu_static_20261031/session_b
```

该模式在树莓派上实时输出去偏加速度和角速度到日志，仍不控制任何执行器；后处理核对实时与离线校正是否一致。

```bash
python scripts/analyze_imu_static_20261031.py \
  logs/imu_static_20261031/session_b \
  --frozen-params logs/imu_static_20261031/analysis_a/calibration_candidate.json \
  --output logs/imu_static_20261031/analysis_b
```

各积分窗口独立设置初始位置、速度为零，积分后不减均值、不拟合终点、不用未来数据回拉、不做速度衰减。`report.json`、`window_metrics.csv` 给出水平与三维速度、位移；`holdout_timeseries.csv` 为 10 Hz 展示数据，计算使用全部约 200 Hz 样本。`sensor_diagnostics.png` 和 `drift_comparison.png` 为图形结果。

这里的“原始积分”已用当前真实四元数去除重力，只是不校准偏置；直接把包含 9.8 m/s² 重力的原始加速度积分没有比较意义。

## EKF 的含义

`control/runtime/imu_static_calibration_20261031.py` 实现 15 维误差状态扩展卡尔曼滤波器（ESKF）：名义状态为位置、速度、四元数、加速度偏置、陀螺偏置，误差状态为 `[δp, δv, δθ, δba, δbg]`。它以原始陀螺积分姿态，重力补偿后传播速度与位置。

对照有三类，不能混为一谈：

- 冻结偏置积分：测试中偏置不变，用传感器输出的当前四元数去重力。包括 ESKF 训练后只导出加速度偏置的候选。
- `eskf_predict_only`：用冻结的初始偏置，原始陀螺传播姿态，无任何观测更新。
- `eskf_known_rest_zupt`：在操作者明确保证架上静止且 1 秒窗口安静时，约 10 Hz 注入零速及零角速观测，在线估计误差。位置没有直接强制归零。

因此最后一项使用了额外“真实速度为零”的信息，不能据此宣称纯 IMU 在游动中也能达到同样精度。安静的 IMU 无法区分静止和匀速平移；`externally_confirmed_static` 默认是 `False`。舵机关闭也不意味着水中鱼体速度为零。

噪声参数使用训练段 1 秒分块均值标准差、10 秒均值差分估计，并设置数值下限。它们是描述当前已被传感器处理过、存在时间相关的输出的工程初值，不是严格 Allan 方差标定。零速观测噪声 0.005 m/s 等超参数在测试前固定，不以最终测试误差调参。

## 后续运动如何使用

`calibration_candidate.json` 可由 `FrozenIMUCalibration.load(path)` 读取。对每帧 `correct(data)` 返回世界系线加速度及去偏陀螺：

```python
from control.runtime.imu_static_calibration_20261031 import FrozenIMUCalibration
cal = FrozenIMUCalibration.load('config/imu_static_candidate_20261031.json')
linear_acc_world, gyro_corrected = cal.correct(sample.data)
```

计算关系为 `a_world = R(q) @ (a_measured - b_body) - [0, 0, g]`，四元数为 `wxyz`、body-to-world。如用于现有 `InertialEstimator`，只能把文件中的 `accel_bias_body_mps2` 作为其 `bias` 的替代初值，不能同时再减一次启动标定的 `b_body`。若需要泳池坐标，还应沿用导航里的初始航向旋转。文件中的低通时间常数需要由调用端明确应用；`correct()` 只做偏置与重力补偿。

陀螺偏置只能校正原始陀螺积分，不能通过减角速度改变传感器已经融合输出的四元数。本次只增加可复用、显式启用的模块，不自动把候选参数接入正在使用的导航配置。

单一固定姿态只能得到“有效残差偏置”，不能区分加速度本征偏置、重力值误差、尺度因子、安装误差及姿态倾角误差。它也不能证明跨姿态、跨温度、重新上电或实际摆尾时可迁移。后续需要多姿态静态数据及带 UWB/视觉等外部真值的运动验证；运动期应由真实位置/速度观测帮助估计偏置，不能一直套用架上零速约束。

## 验证与参考

离线回归测试覆盖旋转坐标下的重力/偏置补偿、真实加速度、匀速不能误归零、静止 ESKF 收敛和协方差半正定、转向不产生平移、非法时间间隔拒绝。

```bash
python -m unittest discover -s tests -p test_imu_static_calibration_20261031.py -v
```

- [Solà: Quaternion kinematics for the error-state Kalman filter](https://arxiv.org/abs/1711.02508)：误差状态、右局部姿态误差及四元数更新。
- [OpenVINS: Zero Velocity Update](https://docs.openvins.com/update-zerovelocity.html)：零速约束、安静/恒速误判与外部运动信息。

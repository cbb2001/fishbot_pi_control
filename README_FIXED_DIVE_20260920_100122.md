# 固定策略下潜（20260920_100122）

入口：`scripts/run_fixed_dive_20260920_100122.py`。控制实现：`control/runtime/fixed_dive_20260920_100122.py`。
只新增文件，不修改 PPO 或其他旧代码；无需 Torch、模型权重或 PPO 训练。

## 运行

在项目根目录执行（依赖沿用项目的 PyYAML 和硬件驱动）：

```bash
# 模拟传感器和 PWM，检查完整流程，不访问硬件
python scripts/run_fixed_dive_20260920_100122.py --h0 0.3 --k 0.5 --dry-run --max-control-s 5

# 树莓派实机；持续控制直到 Ctrl+C
python scripts/run_fixed_dive_20260920_100122.py --h0 0.3 --k 0.5 --confirm MOVE
```

- `--h0`：目标深度，单位米，必须大于 0 且小于安全深度上限。
- `--k`：可调非负增益，默认 1；设为 0 时根部无行程。
- `--action-duration-s`（简写 `--t`）：每个 action2 的时长，单位秒，默认 0.6，必须为正有限数。一组两个动作使用相同的 t，总时长约为 2t（另有调度间隔）。例如 `--t 0.8`。
- `--initial-b2`：第一组第二动作的 b2，默认 +1，也可设为 -1。
- `--surface-pressure-pa`：人工给定本次水面初始 P0（Pa），默认空；自动标定使用 `mean(P_i-f(t_i))`。
- `--pressure-per-meter`：可选覆盖深度换算系数（Pa/m）；默认使用标定配置的 `rho*g`，淡水默认9806.65。显式传入750仍会覆盖，按新物理深度公式运行时请省略此参数。
- `--pressure-drift-config`：默认读取项目 `config/pressure_drift_10hz.json`。
- `--calibration-window-s`：启动标定窗口秒数，实机默认5；IMU和压力稳定性检查仍执行。
- `--config` / `--rl-config`：默认读取 `config/robot.yaml` / `config/rl_training_20260914.yaml`。
- `--pretrain-wait-s`：覆盖启动等待时间；默认沿用 PPO 配置的等待、预热和 IMU 标定。
- `--max-control-s`：从标定及舵机初始化完成后开始计时；到时等待当前组合完成再正常退出。不设则一直运行。
- `--keep-pwm`：正常退出回中后保留 PWM。故障/Ctrl+C 仍立即关闭 PWM；这与目标深度处的保持不同。

## 动作规则

传感器使用 `pressure_pa`（Pa），固定策略会话强制以10 Hz连续采集。默认在启动预热后采集稳定窗口，以 `P-f(t)` 的均值作为 P0，
同时沿用 IMU 标定。标定时应将感压口保持在你要定义的零深度位置；若在水下标定，得到的是相对该位置的深度。
默认真实标定窗口5秒，默认至少45个有效压力样本（给10 Hz窗口边界留余量），修正压力标准差不得超过100 Pa；显式配置可覆盖，分别使用
`startup.calibration_window_s`、`calibration.minimum_depth_samples`、`calibration.max_pressure_std_pa`。
稳定性/样本数未通过会继续等待，超时退出，不启动动作。

提供 `--surface-pressure-pa` 时直接使用人工压力，不用启动样本覆盖它；IMU 标定仍执行。
也可在传入的 `--rl-config` YAML 内新增独立配置（命令行值优先）：

```yaml
fixed_depth:
  surface_pressure_pa: null  # null=自动标定；有数值则使用人工参考，单位Pa
  pressure_per_meter: null  # null=使用rho*g；显式数字可覆盖，单位Pa/m
```

固定策略不再使用 `robot.yaml.depth_sensor.surface_pressure_mbar` 作为控制零点，该旧字段只供底层驱动使用。
控制决策和超深保护统一使用漂移修正深度；`raw_depth.jsonl` 的驱动自带 `depth_m` 仍是旧公式，
分析应使用原始压力、本次 `calibration.json` 及随会话保存的 `pressure_drift.json`。离线分析脚本已支持自动识别。
查表时间从连续采样第一条有效压力记录的单调时间戳开始；预热、标定、动作及保持阶段均连续计时，
不会在电机启动时归零。按每条样本的采样时间线性插值。超出0～7200秒范围会故障退出，不循环或截断索引。

```text
P0 = mean(P_i - f(t_i))  # 人工P0存在时直接采用人工值
h = (pressure_pa - P0 - f(t)) / pressure_per_meter  # 默认分母rho*g
delta = k * (h - h0) / h0 * (max_angle4 - center_angle4)
theta1_absolute = center_angle4 + delta
theta2_absolute = center_angle4 - delta
```

例如自动标定5秒、使用默认rho*g、t=0.4秒：

```bash
python scripts/run_fixed_dive_20260920_100122.py --h0 0.2 --k 1 --t 0.4 --calibration-window-s 5 --confirm MOVE --max-control-s 60
# 人工参考示例：在上面命令中添加 --surface-pressure-pa 104100
```

在每组开始时用最新深度计算一组角度，并固定到该组结束：先 `b1=0,b2=0`，
再 `b1=1,b2=±1`，两动作均采用 `--action-duration-s` 指定的 t（默认0.6秒）。`delta` 严格保留上述符号；浅于目标且 k>0 时为负。
原 PPO `FinAction.theta` 是相对中位偏移，因此底层传入的是 `delta` 和 `-delta`，
日志同时记录绝对角度；原五次平滑轨迹、鳍尖耦合与左右镜像关系不变。

限幅会缩小 delta 的幅值，同时满足 4/6 号根部和 5/7 号鳍尖的机械范围，保持两目标关于中位对称；
每次提交前还会检查轨迹起点、半程峰值和终点。由于镜像及鳍尖限制，实际幅值可能小于 4 号舵机单独允许的幅值。

组合途中穿过 h0 **不取消动作**：仍完成本组两个动作，再读取最新深度。
组合完成后，h<h0 继续执行下潜组合，h>h0 执行上浮组合，h==h0 保持组合结束时已下发的姿态。
b1=0、b1=1 的顺序和 b1=1 时 b2 取反规则在下潜、上浮两种方向中均保持一致。
b2 在每个实际完成的 b1=1 动作后取反，跨保持阶段不清零。

压力采集、PWM 执行、安全监测分别由线程运行；主循环默认每 10ms 检查，
深度固定 10Hz，PWM 默认 50Hz。“立即”表示下次采样/调度机会响应，并非硬实时保证。
只有目标深度判断等待整组完成；传感器故障、安全深度越界、低电压、过流或 Ctrl+C 仍可立即中断。
仅控制 4、5、6、7 号舵机，不操作尾部三舵机。

## 日志与验证

### 可选 PI 控制（2026-10-09）

`--k`（也可写`--kp`）是比例增益；新增 `--ki` 为积分增益，单位1/秒。
`--ki 0` 是默认值，保留原P控制；传入正值启用PI。积分使用10 Hz修正深度的实际采样间隔，
预热、P0/IMU标定及舵机初始化期间不积分。令 `e=(h0-h)/h0`、`span=max_angle4-center_angle4`：

```text
I_next = I + ki * e * dt                 # 受下面的抗饱和规则限制
delta_raw = -(k * e + I) * span           # h<h0 为负（下潜），h>h0 为正（上浮）
theta1 = center_angle4 + delta
theta2 = center_angle4 - delta
```

每组开始时冻结PI输出，两步完成后才采用下一组输出。当前输出已达到根部/鳍尖可用幅度时，
停止继续增加同方向积分；误差反向时允许退积分。在下一组开始前，若积分仍指向旧方向，
清零旧积分（`direction_reset`），保证浅于目标时下潜、深于目标时上浮。这是带方向切换重置的PI。
仅当整组完成且 h==h0 时进入保持并清零积分；之后偏离目标时重新积分。
两种b2方向中较小的允许幅度作为积分上限依据，保留原动作限幅。
重复或乱序样本不积分；样本间隔超过 `state.sensor_age.depth.fault_ms`（默认300ms）时跳过
该间隔的积分并记录 `gap_skipped`，避免把缺失数据期间假定成持续误差。

下面仅为启用参数的示例，`ki=0.1` 尚未通过实机整定：

```bash
python scripts/run_fixed_dive_20260920_100122.py --h0 0.2 --kp 1 --ki 0.1 --t 0.2 --calibration-window-s 5 --confirm MOVE --max-control-s 60
```

`commands.jsonl` 的 `pair` 增加 `p_term_deg`、`i_term_deg`、`integral_status`；前两项均采用原delta符号，
每组下潜时为负、上浮时为正，两者之和等于 `delta_raw_deg`。`depth_control.jsonl` 记录每个新深度样本时的P/I值，
本组实际下发的固定值仍以 `commands.jsonl` 为准。元数据与状态文件记录ki。

这是支持下潜和上浮的双向 PI 控制；h>h0 时主动执行上浮 action2 组合，不再保持上一时刻的下潜姿态。
若组合的 `saturated` 长期为true，说明输出已达到上限，增加ki不能获得更大拍幅。
`angle_cap_deg` 记录实际允许的幅度，`limited` 仍仅表示发生了裁剪（恰好等于上限时可为false）。压力测量偏差也无法由积分修正。

每次运行单独创建 `logs/fixed_dive_20260920_100122/<时间>_fixed_dive/`，包含：

- `commands.jsonl`：相对/绝对目标角、增益限幅结果、组号和 b2。
- `events.jsonl`：动作完成及进入保持的事件。
- `depth_control.jsonl`、`raw_*.jsonl`：深度判断和原始传感器采样。
- `depth_control.jsonl` 还保存原始压力、去漂移压力、漂移量、漂移计时及随时间变化的水面参考压力。
- `pressure_drift.json`：本次使用的模型完整快照，保证离线重算可复现。
- `metadata.yaml`、`calibration.json`：运行配置与标定依据。
- `fixed_dive_status.json`、`result.json`：运行状态、完成组数、故障和清理结果。

```bash
python -m pytest tests/test_fixed_dive_20260920_100122.py -q
```

模拟传感器提供预设深度轨迹，不是机器鱼动力学仿真；模拟通过仅验证程序流程，不能证明实机下潜效果。

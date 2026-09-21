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
- `--surface-pressure-pa`：人工给定参考压力（Pa），默认空；不提供时，启动阶段自动采集稳定压力均值作为本次零点。
- `--pressure-per-meter`：压力-深度系数（Pa/m），默认750，必须为正有限数。
- `--config` / `--rl-config`：默认读取 `config/robot.yaml` / `config/rl_training_20260914.yaml`。
- `--pretrain-wait-s`：覆盖启动等待时间；默认沿用 PPO 配置的等待、预热和 IMU 标定。
- `--max-control-s`：从标定及舵机初始化完成后开始计时；到时等待当前组合完成再正常退出。不设则一直运行。
- `--keep-pwm`：正常退出回中后保留 PWM。故障/Ctrl+C 仍立即关闭 PWM；这与目标深度处的保持不同。

## 动作规则

传感器使用 `pressure_pa`（Pa）。默认在启动预热后采集稳定窗口，以压力均值作为参考，
同时沿用 IMU 标定。标定时应将感压口保持在你要定义的零深度位置；若在水下标定，得到的是相对该位置的深度。
默认真实标定窗口5秒，至少50个有效压力样本，标准差不得超过100 Pa；这些值分别沿用 RL 配置的
`startup.calibration_window_s`、`calibration.minimum_depth_samples`、`calibration.max_pressure_std_pa`。
稳定性/样本数未通过会继续等待，超时退出，不启动动作。

提供 `--surface-pressure-pa` 时直接使用人工压力，不用启动样本覆盖它；IMU 标定仍执行。
也可在传入的 `--rl-config` YAML 内新增独立配置（命令行值优先）：

```yaml
fixed_depth:
  surface_pressure_pa: null  # null=自动标定；有数值则使用人工参考，单位Pa
  pressure_per_meter: 750.0  # Pa/m
```

固定策略不再使用 `robot.yaml.depth_sensor.surface_pressure_mbar` 作为控制零点，该旧字段只供底层驱动使用。
控制决策和超深保护统一使用经验公式；`raw_depth.jsonl` 的驱动自带 `depth_m` 仍可能是旧公式，
分析应使用原始压力及本次 `calibration.json`。新的离线分析脚本已支持自动识别。

```text
h = (pressure_pa - p_surface_pa) / pressure_per_meter
delta = k * (h - h0) / h0 * (max_angle4 - center_angle4)
theta1_absolute = center_angle4 + delta
theta2_absolute = center_angle4 - delta
```

例如自动标定、系数750、t=0.4秒：

```bash
python scripts/run_fixed_dive_20260920_100122.py --h0 0.2 --k 1 --t 0.4 --pressure-per-meter 750 --confirm MOVE --max-control-s 60
# 人工参考示例：在上面命令中添加 --surface-pressure-pa 104100
```

在每组开始时用最新深度计算一组角度，并固定到该组结束：先 `b1=0,b2=0`，
再 `b1=1,b2=±1`，两动作均采用 `--action-duration-s` 指定的 t（默认0.6秒）。`delta` 严格保留上述符号；浅于目标且 k>0 时为负。
原 PPO `FinAction.theta` 是相对中位偏移，因此底层传入的是 `delta` 和 `-delta`，
日志同时记录绝对角度；原五次平滑轨迹、鳍尖耦合与左右镜像关系不变。

限幅会缩小 delta 的幅值，同时满足 4/6 号根部和 5/7 号鳍尖的机械范围，保持两目标关于中位对称；
每次提交前还会检查轨迹起点、半程峰值和终点。由于镜像及鳍尖限制，实际幅值可能小于 4 号舵机单独允许的幅值。

组合途中达到或超过 h0 **不取消动作**：仍完成本组两个动作，再读取最新深度。
此时 h<h0 就立即开始下一组；h>=h0 则保持组合结束时已下发的姿态，不回中、不关闭 PWM。
保持期间持续监测深度，检测到 h<h0 即启动新组，不等待任何定时保持动作结束。
b2 在每个实际完成的 b1=1 动作后取反，跨保持阶段不清零。

压力采集、PWM 执行、安全监测分别由线程运行；主循环默认每 10ms 检查，
深度默认 20Hz，PWM 默认 50Hz。“立即”表示下次采样/调度机会响应，并非硬实时保证。
只有目标深度判断等待整组完成；传感器故障、安全深度越界、低电压、过流或 Ctrl+C 仍可立即中断。
仅控制 4、5、6、7 号舵机，不操作尾部三舵机。

## 日志与验证

每次运行单独创建 `logs/fixed_dive_20260920_100122/<时间>_fixed_dive/`，包含：

- `commands.jsonl`：相对/绝对目标角、增益限幅结果、组号和 b2。
- `events.jsonl`：动作完成及进入保持的事件。
- `depth_control.jsonl`、`raw_*.jsonl`：深度判断和原始传感器采样。
- `metadata.yaml`、`calibration.json`：运行配置与标定依据。
- `fixed_dive_status.json`、`result.json`：运行状态、完成组数、故障和清理结果。

```bash
python -m pytest tests/test_fixed_dive_20260920_100122.py -q
```

模拟传感器提供预设深度轨迹，不是机器鱼动力学仿真；模拟通过仅验证程序流程，不能证明实机下潜效果。

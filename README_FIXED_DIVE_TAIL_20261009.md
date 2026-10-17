# 侧鳍双向 PI 与尾鳍联合控制

入口：`scripts/run_fixed_dive_tail_20261009.py`。
实际控制循环：`control/runtime/fixed_dive_tail_20261009.py`。
侧鳍复用 `FixedDivePolicy`，与 `run_fixed_dive_20260920_100122.py` 使用同一套 action2、PI、限幅和 b2 交替规则。

## 完整调参表（按当前代码核对）

下面的“当前值”对应仓库现有配置或入口默认值；修改文件后以本次命令和 `metadata.yaml` 为准。所有数值应为有限数，单位为秒、米、度、Pa，另有说明的除外。

**本脚本 `--t` 是侧鳍单动作时长；水面脚本 `--t` 是任务总时长，两者不同。** 本脚本总时长使用 `--max-control-s`，省略时仍会在所有阶段总时间后结束，不是无限运行。

### 1. 命令行全部参数

| 参数（同一行的名称为别名） | 默认值 | 作用及限制 |
|---|---|---|
| `--stage H T` | 不指定 | 可重复任意次，按顺序设置目标深度H及阶段时间T；H≥0且小于安全深度上限，T>0。指定后覆盖以下三个旧四阶段参数 |
| `--h0-dive1` | 1 m | 默认四阶段的第1阶段深度；第2阶段固定0米 |
| `--h0-dive2` | 1.5 m | 默认四阶段的第3阶段深度；第4阶段固定0米；两项均要求非负且小于安全上限 |
| `--stage-duration-s` | 30 s | 默认四阶段每段时间，必须>0；默认合计120秒 |
| `--max-control-s` | 所有阶段时间之和 | >0；控制开始后计时，到时不再开启新组，等待当前侧鳍组合完成再退出。大于阶段总时长时继续使用最后阶段目标 |
| `--k` / `--kp` | 1 | 深度PI比例增益，≥0；下潜/上浮共用 |
| `--ki` | 0 | 深度PI积分增益，≥0，单位1/s；0关闭积分 |
| `--side-action-duration-s` / `--t` | 0.6 s | 每个侧鳍action2时长，>0；一组两次动作约为2倍该时间，另有调度开销 |
| `--initial-b2` | 1 | 首个第二动作的b2，只能为1或−1；每完成一组取反 |
| `--tail-action-duration-s` | 0.3 s | 尾鳍每次左摆或右摆时长，>0；一轮往复约0.6秒 |
| `--tail-left-deg` | −15° | 相对三个尾鳍各自中位的左摆偏移 |
| `--tail-right-deg` | +15° | 相对中位的右摆偏移；必须左值<右值，三个舵机的中位+偏移都须在物理限位内；当前共同可用偏移为[−30,+30]° |
| `--pretrain-wait-s` | 实机20 s；模拟0 s | 启动采集后的人工等待，≥0；覆盖RL配置中的同名startup项 |
| `--calibration-window-s` | 实机5 s；模拟0.5 s | 静止标定窗口，>0且小于标定超时；同时配合最少样本数调整 |
| `--surface-pressure-pa` | 自动标定 | 手动水面零点P0，单位Pa，>0；优先于`fixed_depth.surface_pressure_pa`。仍需完成IMU标定 |
| `--pressure-per-meter` | 当前9806.65 Pa/m | 每米对应压力，>0；优先于`fixed_depth.pressure_per_meter`，再回退到水密度×重力加速度。经验系数须按实际标定填写 |
| `--pressure-drift-config` | `config/pressure_drift_10hz.json` | 压力漂移JSON文件；优先于`fixed_depth.pressure_drift_config`，最后使用项目内默认文件 |
| `--config` | `config/robot.yaml` | 硬件、传感器、舵机、日志根目录及基础安全限值 |
| `--rl-config` | `config/rl_training_20260914.yaml` | 读取启动、标定、健康监测和运行设置；本脚本不训练PPO |
| `--dry-run` | 不启用 | 模拟传感器和舵机，不连接硬件；启用simulation覆盖 |
| `--confirm MOVE` | 未确认 | 实机运行必填；模拟不需要 |
| `--keep-pwm` | 不启用 | 请求正常退出后保留PWM；故障/紧急停车仍走停机处理。默认退出释放PWM |
| `-h` / `--help` | — | 显示帮助并退出 |

阶段、PI、动作幅值和动作时间由上述命令行设置，不能靠修改RL配置的`targets`、`training`或`fin_ppo`来改变。命令中的相对文件路径从运行时工作目录解析，建议在项目根目录启动。

### 2. RL配置中生效的参数

以下路径相对 `--rl-config` 指定的YAML。命令行对应项优先；模拟时先应用`simulation`，再应用显式命令行覆盖。

| YAML路径 | 当前值 | 作用及限制 |
|---|---|---|
| `startup.pretrain_wait_s` | 20 | 人工等待秒数，≥0 |
| `startup.sensor_warmup_s` | 5 | 人工等待后的额外传感器预热秒数，≥0 |
| `startup.calibration_window_s` | 5 | 稳定标定窗口秒数，>0 |
| `startup.calibration_timeout_s` | 30 | 标定等待/重试总预算秒数，必须大于窗口 |
| `calibration.minimum_imu_samples` | 200 | 标定窗口内最少IMU有效样本数，正整数 |
| `calibration.minimum_depth_samples` | 45 | 自动P0的最少压力样本数；10Hz、5秒窗口约50条；缩短窗口时需同步调整 |
| `calibration.max_pressure_std_pa` | 100 | 修正后压力窗口标准差上限，Pa；越小越严格 |
| `calibration.max_angle_std_deg` | 1 | IMU三轴角度窗口标准差上限，度；越小越严格 |
| `calibration.max_linear_acc_rms_mps2` | 未设置，默认0.5 | 静止去重力加速度残差RMS上限，m/s²，>0 |
| `calibration.max_acc_std_mps2` | 未设置，默认0.25 | 各轴加速度标准差上限，m/s²，>0 |
| `calibration.max_gyro_radps` | 未设置，默认0.15 | 标定期间各轴角速度绝对值上限，rad/s，>0 |
| `calibration.max_quat_euler_tilt_error_deg` | 未设置，默认5 | 四元数与欧拉角倾斜方向一致性误差上限，度，>0 |
| `calibration.quaternion_rotation` | 未设置，默认`body_to_world` | 四元数旋转约定，可选`body_to_world`或`world_to_body`，按IMU定义设置 |
| `calibration.gravity_world_z_sign` | 未设置，默认1 | 世界坐标重力方向符号，只能±1，按IMU坐标定义设置 |
| `calibration.water_density_kg_m3` | 1000 | 水密度，kg/m³；未显式设置每米压力系数时参与换算 |
| `calibration.gravity_mps2` | 9.80665 | 重力加速度，m/s²；与水密度相乘得默认Pa/m；两项应>0 |
| `fixed_depth.surface_pressure_pa` | 未设置 | 可新增，null表示自动P0；手动值须>0；CLI优先 |
| `fixed_depth.pressure_per_meter` | 未设置 | 可新增，单位Pa/m，>0；CLI优先 |
| `fixed_depth.pressure_drift_config` | 未设置 | 可新增，指定漂移JSON；注意本脚本不使用`pressure_drift.config_path` |
| `state.enforce_data_freshness` | false | true时数据年龄超限触发停车；false时沿用最近样本，首帧缺失、无效数据和硬件故障仍会报错 |
| `state.sensor_age.imu.fault_ms` | 100 | IMU数据最大年龄，ms |
| `state.sensor_age.depth.fault_ms` | 300 | 深度数据最大年龄，ms；也决定PI允许积分的最大采样间隔，即使关闭年龄停车仍影响积分 |
| `state.sensor_age.power.fault_ms` | 2000 | 电源数据最大年龄，ms；三项均须>0 |
| `safety.min_voltage_v` | 继承10.5 | 可在此覆盖robot中的最低电压，V，>0 |
| `safety.max_current_a` | 继承10 | 最大绝对电流，A，>0 |
| `safety.max_depth_m` | 继承5 | 最大允许深度，m，>0；所有阶段目标必须严格小于它 |
| `runtime.poll_interval_s` | 0.01 | 主循环轮询间隔，>0 |
| `runtime.safety_poll_interval_s` | 0.01 | 独立健康监测周期，>0 |
| `runtime.sensor_ready_timeout_s` | 5 | 首个样本等就绪等待预算，>0 |
| `runtime.shutdown_timeout_s` | 10 | 各线程清理等待上限，>0 |
| `runtime.live_status_interval_s` | 1 | 状态JSON最小发布间隔，>0 |
| `simulation.pretrain_wait_s` | 0 | 仅模拟覆盖人工等待 |
| `simulation.sensor_warmup_s` | 0.1 | 仅模拟覆盖预热 |
| `simulation.calibration_window_s` | 0.5 | 仅模拟覆盖标定窗口 |
| `simulation.calibration_timeout_s` | 5 | 仅模拟覆盖标定超时；显式传入`--calibration-window-s`时改用startup中的超时 |
| `simulation.minimum_imu_samples` | 30 | 仅模拟覆盖IMU样本要求 |
| `simulation.minimum_depth_samples` | 4 | 仅模拟覆盖压力样本要求 |

`runtime.action_timeout_margin_s`（当前1秒）会通过公共配置校验，但联合控制循环没有用它实现动作超时；`state.max_state_age_ms`（150ms）也会校验，但本脚本不构建RL状态。不要把这两项当成已实现的联合控制保护。`state.sensor_age.*.warn_ms`、PPO网络/奖励/训练参数、`runtime.update_timeout_s`、`console_status_interval_s`不改变本脚本控制行为。

### 3. robot.yaml硬件和采集参数

| 路径 | 当前值/范围 | 作用 |
|---|---|---|
| `runtime.log_dir` | `logs` | 日志根目录 |
| `sensors.imu.port` / `baudrate` | `/dev/ttyAMA4` / 460800 | IMU串口和波特率 |
| `sensors.imu.sample_rate_hz` | 200 | IMU采集配置频率，须与设备支持值匹配 |
| `sensors.imu.serial_timeout_s` | 未设置，驱动默认0.1 | 串口读超时秒数；与样本年龄阈值不同 |
| `sensors.imu.buffer_size` | 1000 | IMU环形缓存容量，正整数 |
| `sensors.depth.buffer_size` / `power.buffer_size` | 500 / 200 | 压力/电源缓存容量 |
| `sensors.power.rate_hz` | 2 | 电源采集频率，>0 |
| `sensors.depth.i2c_bus` / `address` | 1 / `0x76` | 压力传感器I²C总线和地址；优先于旧i2c/depth_sensor配置 |
| `sensors.power.i2c_bus` / `address` | 1 / `0x45` | 电源传感器I²C总线和地址；回退到`i2c.bus`、`i2c.ina219_address` |
| `sensors.power.linear_cal_reading_ma` | 未设置，默认1000 | 电流线性标定的仪表读数，mA |
| `sensors.power.linear_cal_actual_ma` | 未设置，默认1000 | 上述读数对应的真实电流，mA；两项应使用有效的非零标定值 |
| `sensors.imu/depth/power.enabled` | true | 本脚本需要三种数据；关闭后会因缺失数据而不能正常运行 |
| `sensors.imu/depth/power.timeout_ms` | 100 / 300 / 2000 | 通用健康年龄配置；RL的`fault_ms`优先，标定末样本年龄检查也读取它 |
| `servo.enable_pin` | 4 | 舵机电源使能GPIO，可设null表示不使用该引脚 |
| `servo.pca9685_channels` | 未设置，默认16 | ServoKit通道总数，按硬件填写 |
| `servo.pulse_min_us` / `pulse_max_us` | 500 / 2500 | 全局PWM脉宽端点；每路可用同名字段覆盖 |
| `servo.channels[].servo_id` / `channel` | 1～7 / 0～6 | 逻辑舵机与硬件通道映射 |
| `servo.channels[].center_angle` | 85/95/95/111/90/143/90 | 1～7号中位，度；控制偏移以此为基准 |
| `servo.channels[].min_angle` / `max_angle` | 各路独立配置 | 机械限位；中位应在限位内，决定可用幅值 |
| `servo.channels[].actuation_range` | 180 | 驱动角度量程，度，按舵机标定设置 |
| `discrete_action_coupling.left.root_reference_span_deg` | 106 | 侧鳍根部参考跨度 |
| `discrete_action_coupling.left.tip_reference_span_deg` | 89 | 尖端参考跨度；两跨度须>0，用于action2耦合，仍受物理限位约束 |
| `safety.min_voltage_v/max_current_a/max_depth_m` | 10.5/10/5 | RL配置未覆盖时使用的三个基础安全限值 |

当前公共action2标定读取`discrete_action_coupling.left`并用于两侧；缺失时回退到`discrete_absolute_actions_20260725.left_fin`。修改`right`分支不会单独改变右侧耦合。

深度采样在本脚本中强制为10Hz；UWB和相机强制关闭。执行器写入周期固定50Hz，实机初始化/正常退出回中各2秒（模拟为0），不会读取`servo.discrete_executor`中的同名值。`servo.frequency_hz`当前也未传给ServoKit构造器。压力控制不使用旧`depth_sensor.surface_pressure_mbar`作零点。实际I²C地址/总线由底层驱动读取的配置决定，不能仅凭YAML中存在同名字段认定会生效。

漂移文件必须为10Hz、Pa、秒单位的有效查表文件；总采集时间（含等待和标定）不得超出表的`lookup.end_s`，超出会报错。更换实测漂移模型使用`--pressure-drift-config`，不要只修改文件内的采样率标签。

## 控制规则

- 深度浅于目标时下潜，深于目标时主动上浮，精确等于目标时保持侧鳍姿态。
- 每组依次执行 `action2(delta, t, 0, 0)` 和 `action2(-delta, t, 1, b2)`。
  接口中的 theta 是相对中位偏移；4号舵机绝对目标为 `center_angle + delta`、`center_angle - delta`。
- 每完成一组，下一组的 b2 取反；上下潜切换和阶段切换均不重置交替顺序。
- 两个动作的角度在组开始时冻结；中途越过目标仍执行完整组，再依据最新深度决策。
  故障和 Ctrl+C 仍立即停止。
- 保留 10 Hz 压力采样、压力漂移修正、自动 P0 和 IMU 标定、手动 P0 覆盖。
  控制深度使用 `(raw_pressure_pa - P0 - drift_pa) / pressure_per_meter`。

`--k`（或 `--kp`）和 `--ki` 同时作用于下潜和上浮；`--ki 0` 默认退化为 P 控制。
正目标深度按原公式以 h0 归一化。0米阶段不能除以 h0，因此使用最近的正目标深度作为误差尺度 S：

```text
e = (h0 - h) / S
I = ki * ∫e dt
delta_raw = -(k * e + I) * (max_angle_4 - center_angle_4)
```

默认四阶段为：30秒目标1米 → 30秒目标0米 → 30秒目标1.5米 → 30秒目标0米；
对应 S 为 1、1、1.5、1.5 米。自定义流程从0米开始且无历史正目标时 S=1米。
阶段切换清零积分和积分计时，原有抗积分饱和、方向切换积分重置及舵机限幅继续生效。
阶段使用累计时间边界，切换时等当前侧鳍组合完成；默认在各阶段时长之和后收尾退出（原四阶段为120秒）。
持续时间从控制开始按累计时间计算，并非达到目标深度后才开始计时；动作组合可能推迟切换，过短的阶段可能被跳过。
控制计时不含启动等待、标定和退出回中。

## 运行示例

通过重复 `--stage h t` 设置任意 n 个阶段（n≥1）：h 为目标深度，单位米；t 为阶段持续时间，单位秒。
例如三个阶段：0.5米20秒 → 1米40秒 → 0米15秒，同时设置尾鳍左摆-12°、右摆18°、单次摆动0.4秒：

```bash
python scripts/run_fixed_dive_tail_20261009.py --dry-run --stage 0.5 20 --stage 1 40 --stage 0 15 --tail-left-deg -12 --tail-right-deg 18 --tail-action-duration-s 0.4
```

- 添加或删除 `--stage h t` 即改变阶段数量，无需另外填写 n。
- h 必须为有限非负数且低于安全深度上限，t 必须为有限正数。
- 使用 `--stage` 时覆盖 `--h0-dive1`、`--h0-dive2` 和 `--stage-duration-s`；省略则沿用旧四阶段配置。
- `--tail-left-deg` / `--tail-right-deg` 分别设置相对舵机中位的左右摆角，默认-15° / +15°；左值必须小于右值，目标角须在舵机限位内。
- `--tail-action-duration-s` 为左右共用的单次摆动时间，默认0.3秒；完整左右往复包含两次动作。三个尾鳍舵机同角度联动，这些参数在所有阶段共用。
- `--t` 是侧鳍单动作时长，与阶段持续时间、尾鳍摆动时间不同。

实机运行时将 `--dry-run` 替换为 `--confirm MOVE`。

在项目根目录先做模拟检查（不驱动硬件）：

```bash
python scripts/run_fixed_dive_tail_20261009.py --dry-run --h0-dive1 0.3 --h0-dive2 0.5 --stage-duration-s 1 --t 0.2 --tail-action-duration-s 0.1 --ki 0.1
```

实机四阶段示例（增益为示例，需根据实机响应调整）：

```bash
python scripts/run_fixed_dive_tail_20261009.py --k 1 --ki 0.1 --t 0.6 --tail-action-duration-s 0.3 --tail-left-deg -15 --tail-right-deg 15 --pressure-per-meter 750 --confirm MOVE
```

`--max-control-s` 可覆盖控制总时长；正常到时等待当前侧鳍组合完成。
`--calibration-window-s` 调整标定窗口（实机默认5秒）；`--pretrain-wait-s` 调整启动等待。
`--surface-pressure-pa` 可传入手动 P0（Pa），省略时自动标定。

日志保存在 `logs/fixed_dive_tail_20261009/<时间>_fixed_dive_tail/`。
`commands.jsonl` 区分 fin/tail，记录阶段、组合方向、P/I 输出和误差尺度；
`depth_control.jsonl` 记录实时修正深度；`events.jsonl` 记录动作完成和阶段切换。

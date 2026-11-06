# 水下单点与多点导航（20261026）

入口 `scripts/run_underwater_navigation_20261026.py`。一套采集线程持续读取 IMU、压力、UWB、电源，并按 YAML 开关采集相机；一个 PWM 线程同时控制尾鳍 1–3 和胸鳍 4–7。胸鳍复用固定深度 PI/两半程/镜像耦合，尾鳍复用当前水面导航 PI 及 ±45° 大角差转向规则。

## 使用

在项目根目录运行，使用原项目 Python 环境（需要 numpy、PyYAML，实机驱动依赖同旧程序）。

```powershell
# 无硬件流程验证；不模拟水动力，不生成真实相机图像
python scripts/run_underwater_navigation_20261026.py --dry-run --t1 0 --t2 1 --t3 15 --t4 2

# 真实传感器诊断、保存相机图像；绝不创建 PWM 后端
python scripts/run_underwater_navigation_20261026.py --no-servo --t3 60

# 实际下潜与导航
python scripts/run_underwater_navigation_20261026.py --h0 0.3 --x-d 10 --y-d 10 --t1 5 --t2 5 --t3 120 --t4 5 --confirm MOVE
```

`--config` 选择 robot.yaml；`--output` 可指定一个尚不存在的日志目录。`--help` 列出胸鳍和尾鳍控制参数。滤波、稳定性和超时参数在 `config/robot.yaml` 的 `underwater_navigation_20261026` 配置；尾鳍参数和基站几何默认复用 `surface_navigation_20261010`，也可在水下段覆盖尾鳍参数及目标点。

配置以 `robot.yaml` 为统一来源，运行时不再强制覆盖采样率、相机开关或驱动压力参考。缺少必需字段会在启动前报错。已有命令行参数仍可显式临时覆盖任务参数；本次生效值保存在 `metadata.yaml.config`，原始硬件配置保存在 `metadata.yaml.robot`。

| 配置内容 | robot.yaml 位置 |
|---|---|
| 舵机通道、中位、机械限位 | `servo.channels` |
| 胸鳍耦合标尺 | `discrete_action_coupling`；旧绝对动作使用其独立 `discrete_absolute_actions_20260725` 标尺 |
| PWM 指令频率 | `servo.discrete_executor.command_hz` |
| IMU/压力采样率、新鲜度、相机开关与存图 | `sensors`，新鲜度使用各传感器 `timeout_ms` |
| 基站几何、尾鳍策略 | `surface_navigation_20261010`；L/W/H 只在这里定义 |
| t1–t4、定深、滤波、UWB质量、标定阈值、重力/密度、运行周期、关闭时限 | `underwater_navigation_20261026` |
| 日志刷新与队列容量 | `logging.flush_interval_s`、`logging.write_queue_maxsize` |

任务动作序列仍由对应 mission YAML 定义，测试按文件内容检查，不固定旧动作数量。控制算法的数学常数、协议定义及“归中直接写中位”等阶段语义不属于可调物理标定。

## 多目标段（x, y, h, t）

在 `robot.yaml` 的 `underwater_navigation_20261026.waypoints` 配置目标点表。默认 `null` 保留原来 `x_d/y_d/h0` 的单点任务；设置非空列表后，每点独立指定水平目标、期望深度和切段时间。点表支持 `{x,y,h,t}` 映射或 `[x,y,h,t]` 四元列表，例如：

```yaml
underwater_navigation_20261026:
  waypoints:
    - {x: 10, y: 10, h: 0.3, t: 20}
    - {x: 15, y: 10, h: 0.5, t: 25}
    - {x: 15, y: 15, h: 0.4, t: null}
```

`x/y/h` 单位为米；`x/y` 必须位于配置的水池范围，`h` 向下为正，允许 0，必须低于 `safety.max_depth_m`。`t` 单位为秒，只能为有限正数或 `null`，不是胸鳍/尾鳍的动作时长。

| 当前点的 t | 切段与保持规则 |
|---|---|
| 正数 | 水平距离 `sqrt((x-x_d)^2+(y-y_d)^2) <= delta_s`，或当前点累计时间达到 t，任一条件满足就切下一点 |
| null | 始终控制当前目标，不自动切下一点；到点时沿用保持逻辑，漂出范围后继续纠偏 |

**只判断水平距离，h 仅用于胸鳍定深控制。** 到点坐标来自当前 IMU/EKF 估计，或上浮后已通过有效性检查的 n 帧平均 UWB 定位。部分 UWB 测距、尚未收齐 n 帧的数据不能触发按定位到点切段。单点任务及最后一个目标点的 `t` 必须为 `null`；中间点也可以设 `null`，此时后面的点不会自动执行。

第一个点的 t 从首次进入导航开始计时，后续点从切换到该点开始计时，均包含上浮、归中和等待 UWB 的时间。全局 `t3` 到期始终结束任务，当前点是否到达、是否超时不改变总时限。

可重复提供 `--waypoint X Y H T` 临时覆盖 YAML 点表，`NULL/null` 大小写均可。该选项不能与单点参数 `--x-d/--y-d/--h0` 同用：

```powershell
python scripts/run_underwater_navigation_20261026.py --waypoint 10 10 0.3 20 --waypoint 15 10 0.5 25 --waypoint 15 15 0.4 NULL --t3 120 --confirm MOVE
```

导航中切点时，先取消旧目标动作、尾鳍和胸鳍归中，等待 PWM 回执后按新目标继续运动，并清除航向和定深 PI 的旧积分。切点保留当前 IMU/EKF 位置、速度和积分状态，不重新标定，也不重置 `t3/t4`。因此连续经过多个点不会推迟本轮上浮。

若 `t4` 同时到期，先执行上浮；上浮或等待 UWB 期间某点 t 到期，只更新下一目标，不提前恢复尾鳍导航。仍需完成上浮、归中和 n 帧平均定位，再按校正坐标判断当前点、恢复导航。最后一点到达后沿用原控制：尾鳍归中，胸鳍定深；漂出范围后恢复尾鳍纠偏，`t4` 仍周期触发上浮，直到 `t3` 到期。

本点表用于 `run_underwater_navigation_20261026.py`。`only_one_yaw` 入口的三元点表保持独立，不读取这里的四元路线。

## 时间和阶段

| 参数 | 定义 |
|---|---|
| t1 | 采集开始后的人工放置时间，无 PWM；鱼头朝水池 +X |
| t2 | 固定静止标定窗口，默认 5 秒；失败退出，不用运动样本补齐 |
| t3 | 首次导航起总时间，包含上浮、到点保持、归中和等 UWB；不会重置 |
| t4 | 每次导航恢复起本段最长时间，包含下潜；到期开始上浮 |
| 每个目标点的 t | 当前目标的最长停留时间，包含上浮和等 UWB；正数时到点或超时切段，null 时保持当前目标 |

首次完成标定后，累计 `uwb_average_frames`（默认 5）个有效完整四路帧，逐路取算术平均建立比较基准并启动 PWM 归中。归中回执之后重新累计 5 帧，平均定位后才开始导航；两组数据不混用。初始定位整体仍受 `fix_timeout_s` 预算限制，初次进入导航才启动 t3/t4。n 可配置为正整数，包括 1。

导航中胸鳍定深与尾鳍转向并行，IMU 无 UWB 时继续粗推位置。有限 t 的中间目标到点后切下一点；`t:null` 的目标到点后尾鳍归中，胸鳍继续定深，t4 仍计时。

t4 到期取消当前尾鳍/定深动作并归中；确认 PWM 回执后，尾鳍保持归中，侧鳍开始固定 `action2` 序列：4 号最大角、`fin_duration_s`、`b1=0`，接着 4 号最小角、`fin_duration_s`、`b1=1`、当前 `b2`，依次往复。最大/最小角读取 `servo.channels`（当前 164°/58°），内部转换为相对 111° 中位的 +53°/−53°。两侧镜像和鳍尖耦合沿用原定义，完整行程在启动前校验，不自动缩幅。`b1=0` 时原 action2 定义忽略 `b2`，日志规范化为 0；相邻已完成 `b1=1` 动作的 `b2` 取反，跨下潜/上浮保留，只取消而未完成的动作不推进交替状态。

上浮不再使用 `h=0` 定深控制，也不按压力暂停。收到**一帧**新鲜 mc 测距、其中任意一路为正有限值，立即请求取消动作并全部归中；确认 PWM 回执后进入 UWB 再接收，累计 n 帧合格四路测距并平均后，校正位置、清零本航段积分状态并恢复导航。部分测距只触发停止拍鳍，不能用于重定位。上浮和再接收取消各自 30 秒超时，无合格数据时持续执行上浮序列/保持归中等待，但仍受总任务 `t3`、外部停止及安全监测约束。t3 到期由 PWM 线程归中并关闭七通道；故障或 Ctrl+C 紧急关闭。

`--no-servo` 在 t1/t2 后立即开始 t3 诊断计时，无需等 UWB；不执行主动下潜/上浮状态转换，也不积分位移。仍记录原始/去重力加速度并可进行 n 帧 UWB 平均定位；连续惯性诊断使用独立 IMU 诊断脚本。

正常 `t3` 到期后，立即取消运动并写入七路中位，随后保持中位 PWM 达到 `servo.discrete_executor.center_settle_s`（当前 1 秒），再关闭 PWM。保持时间属于结束清理，不额外延长导航；阶段转换的归中仍使用写入回执，不增加此等待。`shutdown_timeout_s` 必须大于保持时间加一个写入周期。安全监测在保持期间继续工作，故障或 Ctrl+C 可立即打断保持并关闭输出。`events.jsonl` 的 `terminal_center_status` 和 `result.json.terminal_center` 记录中位写入、保持结束、PWM 关闭时间；这些是命令确认，没有舵机角度反馈，不能证明受损或堵转的舵机实际归中。

## 坐标、校准与估计

- X：A0→A1；Y：A0→A3；输出 z 是压力观测约束的 EKF 深度，向下为正，单位米。暂停时 z 显示压力低通值，水面重定位瞬间为0。
- 基站使用同高矩形 `(0,0)、(L,0)、(L,W)、(0,W)`，高度 `−H`。L/W 必须为实际基站矩形，H 为基站与初始水面标签的天线高度差。50×26.5 m 是未来部署配置，不能解释此前小水池测距。
- 首版要求 IMU X 朝鱼头、yaw_sign=+1；鱼头在 t1/t2 朝水池 +X。原四元数去重力，启动 yaw 旋转同时用于加速度和航向。暂忽略标签/IMU/压力传感器杆臂。
- 压力采样率读取 `sensors.depth.rate_hz`，漂移文件读取 `underwater_navigation_20261026.pressure_drift_file`（相对项目根目录）。当前模型只支持 10 Hz，配置不匹配会报错，不会偷偷改回 10 Hz。首压力样本锁定漂移起点，t2 修正压力均值为 P0；上浮不重置它们。
- IMU 采样率读取 `sensors.imu.sample_rate_hz`，当前为 200 Hz，应与设备实际输出率一致。标定检查阈值、覆盖率和样本比例均读取水下配置段，样本数按配置频率×t2计算；当前为80%窗口覆盖、90%预期数量。机体系残差均值作为初始加速度零偏。
- 仅 `navigate` 阶段（含下潜）对去重力/启动零偏后的三轴加速度做两级因果低通，再按实际采样 dt 传播九状态 EKF；低通由 `accel_tau_s` 控制，默认每级 0.20 秒。上浮、等待、初始定位和无舵机诊断均不做惯性积分。导航中校准后的原始压力深度直接作为 EKF 观测，避免额外低通延迟；`depth_tau_s` 只用于暂停时压力显示。胸鳍和安全检查继续使用校准压力，独立于 EKF。
- IMU 有效时间间隔 >0.05 秒退出，不跨空缺积分；IMU/压力/电源年龄默认上限为 0.1/0.3/2 秒。UWB 水下失联属于正常状态。

## UWB 平均、分段重置与 EKF（20261101）

只处理同一条 mc 报文中的四路距离，不拼接不同帧，也不把 KT 显示数据重复计数。帧须在本阶段边界后采集、未重复，且年龄不超过 `uwb_age_s`（默认 0.3 秒）。初始第一帧没有比较基准时，只要求四路全部为正有限值。

四路分别与上一组已接受的平均测距比较，差值不得超过 `uwb_max_range_delta_m`（以 YAML 为准，当前 5 米）；首次无已接受基准时，组内与首个有效帧比较。累计 `uwb_average_frames: 5` 帧之后，**先对四路距离分别求平均，再用同高矩形平方差解算 XY**，而不是平均解算后的 XY。该组观测深度固定为 0，输出起点 `[x,y,0]`。同一 UART 批次可包含多条不同 mc，每条分别计数；坏帧、过期帧和重复帧不增加数量。每组只发布一次，归中前的数据不计入归中后这组。

每次定位后严格以平均 UWB 坐标作为下一段全局起点，清零相对位移、三轴速度、两级加速度低通、积分时间起点及上一航段状态；不回放上浮或等待阶段的 IMU。采集时间/TID 继续用于去重和失联检测，不能随积分计时一起清除。IMU 启动机体系零偏 `b_body`、初始 yaw 对齐、压力 P0 与压力漂移时间起点始终保留。压力继续约束下潜后的 z，不修改压力传感器标定零点。

EKF 状态为 `[x,y,z,vx,vy,vz,bx,by,bz]`，z 向下；`bx/by/bz` 是池坐标系等效加速度误差，与启动标定 `b_body` 分开。三轴加速度进入 EKF 前将 Z 向上转换为深度向下。压力以 `h_pressure=z` 持续校正 z/vz/bz；压力事件先用最近已知加速度传播到采集时刻，再更新观测，同一段时间不会因 IMU/压力交错到达而重复积分。压力直接约束垂向，不能保证消除水平误差。

水面使用平均四路测距与深度0组成五维联合观测：`h_i=sqrt((x-Axi)^2+(y-Ayi)^2+(H+z)^2)`，最后一维 `h_depth=z` 的观测值为0。迭代 EKF/Joseph 更新后重新设置 `[平均UWB x,平均UWB y,0]` 起点。上浮/等待冻结 EKF 均值，但在下一次更新时增加三轴未知运动方差，避免将未积分的上浮位移全部解释成 IMU 偏差；暂停时屏幕 z 仍可显示压力深度。

EKF 更新后仍将位置精确设置为平均测距解算坐标，将速度重置为有不确定性的零速度先验。NIS 超限时只跳过偏差学习，仍使用已判有效的平均 UWB 起点。位置方差按测距局部几何重建，速度协方差重新初始化；这是“UWB 起点重置 + EKF 偏差学习”，不是完全保留 EKF 后验的纯滤波器。

参数在 `underwater_navigation_20261026.ekf`：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `enabled` | true | 关闭后仍执行 n 帧平均和仅导航积分，但不学习 EKF 偏差 |
| `retain_bias_on_fix` | true | 保留学习到的等效偏差及其不确定性；false 则每次重新初始化该偏差。两种模式都保留启动标定 |
| `range_sigma_m` | 0.3 | 平均四路测距标准差(m)，R=σ²I；考虑帧间相关性，不自动除以√n |
| `accel_noise_density` | 0.03 | 连续加速度过程噪声幅值 m/s²/√Hz |
| `bias_random_walk` | 0.003 | 等效偏差随机游走幅值 (m/s²)/√s |
| `initial_bias_sigma_mps2` | 0.03 | 初始等效偏差标准差，均值为0 |
| `initial_position_sigma_m` | 0.7 | 初始/退化测距几何位置标准差 |
| `segment_velocity_sigma_mps` | 0.10 | 每段零速度先验的标准差，非实际测量速度 |
| `pause_velocity_sigma_mps` | 0.30 | 暂停位置方差增量为 `(σ × 暂停秒数)²` |
| `pressure_sigma_m` | 0.01 | 压力深度标准差(m)，也用于水面z=0观测。小于测距噪声，赋予压力更高权重 |
| `pressure_innovation_gate` | 9 | 一维压力创新 NIS 上限，拒绝异常压力更新；不改变独立安全监测 |
| `innovation_gate` | 100 | 四路测距与水面z=0的五维创新 NIS 上限，控制偏差学习 |
| `iterations` | 3 | 测距观测迭代线性化次数 |

默认过程噪声、XY初始不确定性等参考 `logs/underwater_navigation_20261026/ekf_replay_20261030/selected_config.yaml`。新增三维模型各轴先共用相同过程/偏差噪声。历史五次有效启动标定的压力标准差约26.27–46.46 Pa，折合深度2.7–4.7 mm，压力观测标准差暂取1 cm留余量，并非已测得的动态精度。保留当前 `accel_tau_s=0.20`，对应 `ekf_y_options_20261030/report.txt` 中单独对比的 0.2 s 低通。原 EKF 回放是六状态、单帧观测，不代表当前“五帧平均、九状态压力融合”已经得到实机精度验证。

`surface_timeout_s`、`settle_s`、窗口/MAD/拟合/边界参数、`resume_depth_*`、`auto_bias` 与 `bias_*` 仍为旧配置/历史回放兼容字段，不参与当前导航定位或上浮控制。旧 `InertialEstimator` 和 `StableUWB` 仅供历史分析，实际会话使用 `NavigationEstimator` 和批量 `AcceptedUWB`。

## 日志与验证边界

日志目录包含 `metadata.yaml`、`calibration.json`、`pressure_drift.json`、`result.json`，五类 `raw_*.jsonl`，`events.jsonl`、`commands.jsonl`、`navigation.jsonl`、`imu_filtered.jsonl`。实机相机按已有配置存入 `camera/` 与 `camera_index.jsonl`；保存错误/队列溢出会结束任务并报告。实时约 1 Hz 输出阶段、估计 XYZ、t3/t4 剩余时间及 UWB 拒绝原因。

多点任务的 metadata 保存生效点表；事件、动作和状态日志记录当前点编号与目标，切点事件区分水平到达和超时原因。`result.json` 保存结束时所在目标点，便于核对提前到点、上浮等待中的超时以及最终保持是否按配置执行。

`estimated_target_reached` 表示估计到点；`uwb_acquire_started` 表示已归中开始重新累计；`position_corrected` 表示 n 帧平均定位已用于重置起点。`uwb_accepted` 保存平均距离、组内每帧原始距离、帧 ID 与采集时间；结果同时保存在 `result.json.last_accepted_uwb`。`collecting` 表示数量不足，`range_jump` 表示超过 YAML 跳变门槛。`imu_filtered.jsonl` 增加 `integrating`、`relative_xyz`、三轴 `ekf_bias_mps2` 和九维 `ekf_variance`；定位事件记录更新前创新、NIS、偏差更新是否接受及重置后的 `[x,y,0]`。`navigation.jsonl` 的 `pressure_ekf_update` 记录压力观测、融合坐标和压力 NIS；metadata 明确记录九状态顺序及加速度/EKF的Z轴符号。

两级低通针对 20261019 日志约 1.47 Hz 摆鳍成分，理论幅值约 0.23、低频延迟约 0.4 秒。它不能消除慢变偏差。该日志缺少真实基站坐标和水平真值，仅能验证解码、失锁、滤波响应，不能证明水平导航精度。默认 `delta_s=0.2 m` 是控制判据，不是精度承诺。软件模拟同样不验证实际下潜、浮力和航向安装约定。

软件验证与已知限制见 `docs/superpowers/plans/2026-10-26-underwater-navigation-review.md`。原始日志回放脚本及结果在 `logs/20261019_145059_fixed_dive/navigation_analysis/replay_underwater_20261026.py` / `.json`，实测摆鳍频段幅值比约 0.224–0.227。早期16项标定/示例测试失败已按用户确认修正：预期读取原始YAML，保留实际中位、机械限位与动作序列；最新验证记录见上述报告。

2026-11-01 九状态版本验证：水下相关67项通过；全库646项、113个子测试通过，报告 `docs/superpowers/plans/2026-11-01-underwater-3d-ekf-validation.txt`。覆盖五帧平均、四帧不恢复、同批多帧去重、阶段重置、压力校正z/vz/bz、三轴符号、压力异常拒绝、交错采样不重复积分、EKF雅可比/协方差与偏差恢复。CLI模拟日志 `logs/underwater_navigation_20261101_3d_ekf_check` 正常 time_limit 结束，4次重定位均为五帧均值且z=0，36次压力更新，无暂停阶段积分、无故障或残留线程。没有运行真实硬件，九状态实机精度仍需水池数据验证。

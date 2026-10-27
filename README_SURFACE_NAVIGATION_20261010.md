# 水面定点导航（20261010）

入口：`scripts/run_surface_navigation_20261010.py`。复用项目的 `SensorManager`、IMU/UWB串口采集、五次平滑action1轨迹与单线程PWM执行器。只控制尾鳍1、2、3号舵机（当前对应PCA9685通道0、1、2），胸鳍4～7号不初始化运动、不写入驱动PWM。

## 1. 场地与配置

从上往下看，规定 **+y在初始鱼头左侧**。四基站的编号和位置必须为：

```text
A3(0,W) ---------------- A2(L,W)
   |                        |
   |       运动区域          |
   |                        |
A0(0,0) ---------------- A1(L,0)  → +x
```

在 `config/robot.yaml` 的 `surface_navigation_20261010` 段填写实测 `L`、`W`（米）和目标 `x_d`、`y_d`。这四项初始为 `null`，必须填写或从命令行传入，不会猜测水池尺寸。目标必须在闭矩形内。**初始朝向固定为+x，导航过程中朝向会随转向改变。** 初始位置由UWB求出，无须手填。

四个基站需要同高，但不必与鱼身标签同高。当前实测基站天线距水面1.93米，标签天线约在水面，因此配置`H: 1.93`。H表示**基站天线与标签天线的垂直距离**，不是到鱼身中心的距离；可用`--H`覆盖。以水面为z=0，基站分别为(0,0,H)、(L,0,H)、(L,W,H)、(0,W,H)，标签为(x,y,0)。

设距A0～A3的斜距为r0～r3，先将各路斜距投影到水平面，水平距离平方为`q_i = r_i² - H²`，再使用：

```text
x = (q0-q1+q3-q2+2L²)/(4L)
y = (q0-q3+q1-q2+2W²)/(4W)
```

共同H²在差分中抵消，因此这里的x、y公式与直接使用斜距平方差**数学等价**；以前的同高公式并非假定H=0。已知H时仍拒绝斜距小于H的报文；当前现场使用实测H=1.93。已取消四路距离反算及残差检查，直接采用平方差计算出的坐标。`range_residual_limit_m` 仅兼容旧配置/命令，其数值不再影响定位。未设置H或设为null时仍可计算x/y，不再估计共同高度来做残差校验。

已取消计算位置的矩形边界检查和边界裁剪：即使得到 `x<0`、`x>L`、`y<0` 或 `y>W`，也保留实际算出的坐标，用于日志和导航，不因越界本身回中等待。本次未增加位置滤波。`boundary_tolerance_m` 仅兼容旧配置/命令，不再影响定位。此行为同样用于 `--no-servo` 诊断模式。

四路距离有效性、坐标有限性和斜距不得小于已知高度的检查仍保留；计算位置不再接受矩形范围或测距残差筛选。例如r=(2.612,50.805,58.554,29.136)米、L=51.5、W=26、H=1.93时，直接返回约(0.731,−3.245)米，并用于导航，不因越界或残差大而回中等待。本次未增加位置滤波，`--no-servo`同样采用此计算逻辑。人工设定的目标点（含多点路线）仍要求位于 `[0,L] × [0,W]` 内。

## 2. UWB报文

采集与 `scripts/test_uwb_20261014.py` 共用 `SensorManager → UWBWorker → read_uwb_chunk`，统一使用 `read_until(b'\r\n', size=4096)`；收到CRLF立即返回，不等待凑满256字节。4096仅为异常输入的读取上限。超时可能返回半行，原始字节保存在日志并由流式解析器继续拼接；普通空读和进入等待定位时不清除半帧，串口异常或缓冲序号缺口才清除。断流的超时仍由 `uwb_timeout_s` 判断，缺基站复用规则见下文；标定前片段仍会清除，不能用于启动运动。所有其他通过公共 `SensorManager` 采集UWB的任务也使用同一读取方式。

以用户实测帧为准：

```text
mc 00 00001dfe 0000bfbc 0000d292 00005e51 4705 c7 002826cf t0:0 0000
$KT0,7.68,49.08,53.91,24.15,null,null,null,null,LO=[no solution]
```

`mc` 距离是十六进制毫米，前四路依次对应A0～A3，即 **7.678、49.084、53.906、24.145米**。`$KT0` 是相同距离的两位小数显示，本脚本不重复处理它。`LO=[no solution]` 是模块自身定位结果，不妨碍利用有效测距自行求解。实测第二字段为 `00`，不能将其解释成“四路全部无效”。

解析器从 `raw_hex` 恢复原始字节，按换行拼接任意拆包/粘包；不使用被 `strip()` 过的 `raw_text` 拼包。只接收配置的 `uwb_tag`（默认 `t0:0`）；四路/八路mc帧均提取前四路。使用标签、序号及模块时间字段对完整测距去重。报文结构正确但某路为0、ffffffff等无效距离时，保留上一组完整且通过定位校验的距离和坐标，绝不拼接不同帧的各路距离；新完整有效数据到来后整组替换。尚未收到完整有效数据时不启动导航动作。普通拆包继续拼接；格式错误、定位几何错误仍回中等待。

持续收到本标签的缺基站mc报文时，允许完整定位年龄超过 `uwb_timeout_s`，继续复用旧定位。此时该超时用于检测mc报文是否真正断流；串口错误、缓冲缺口或持续没有新mc报文仍回中等待。KT显示行和其他标签不能延续控制。

## 3. 标定、滤波与航向控制

1. 等待 `t1` 秒供人工放置机器鱼；此时不初始化尾鳍PWM。
2. 采集 `t2` 秒静止IMU。yaw/roll用圆周均值，pitch用算术均值建立零点，并检查样本数量、时间覆盖和角度稳定性。标定失败不会启动PWM。
3. 等待标定结束后的新有效UWB定位，启用尾鳍并开始控制。
4. IMU每个新样本做一次低通。yaw/roll沿最短角差滤波，避免179°和−179°被平均成0°。pitch原始范围为±90°，归零后保存相对初始姿态的线性差值，不强行裁剪这个差值。

目标航向及误差：

```text
yaw_d = degrees(atan2(y_d-y, x_d-x))
error = wrap180(yaw_d-yaw)        # [-180,180)
normalized = clamp(error/yaw_error_scale_deg, -1, 1)
e = clamp(kp*normalized + integral, -1, 1)
effort = abs(e)
```

代码默认 `yaw_error_scale_deg=180`，当前robot.yaml设置为90；因为最短角差最大幅值为180°，尺度须在(0,180]内。归一化后保留误差符号，PI输出e为[-1,1]，摆幅增量取abs(e)。用 `atan2` 处理全部象限和 `x_d=x`；比较最短角差的正负，不直接比较环绕后的两个航向数值。恰好180°时按−180°（右转）处理。

当尚未到点且 `error>=45°` 时，依次往复执行 `action1(30,t_0)`、`action1(20,t_0)`；当 `error<=-45°` 时，依次往复执行 `action1(-30,t_0)`、`action1(-20,t_0)`。每个动作完成后，使用最新标定、滤波后的IMU航向重新判断；`abs(error)<45°` 时从第一动作恢复原PI控制，若跨到相反方向的大角差则从该方向的±30°重新开始。正在执行的动作完成后才切换，尚未开始的旧模式动作取消。阈值固定为45°，与兼容参数 `delta_yaw` 无关。动作时长使用 `t_0`，不是任务总时长 `t`。

以下PI公式仅用于 `abs(error)<45°`。积分使用实际控制决策时间间隔，限幅防止饱和积累。新定位及动作完成时用最新IMU重算；缺基站期间每组动作之间也使用旧完整定位重算。强转向期间清零积分及积分计时，恢复PI时重新开始；转向变化和误差为零时清积分，长决策间隔不积分。积分限幅保持原实现。`ki=0` 为纯P。

```text
error < 0: theta_zuo = -abs(e)*(theta_max-theta_0)-theta_0
           theta_you = theta_00
error > 0: theta_zuo = -theta_0
           theta_you = abs(e)*(theta_max-theta_0)+theta_00
error = 0: theta_zuo = -theta_0
           theta_you = theta_00
```

注意右转增量严格沿用用户公式中的 `theta_max-theta_0`。当 `theta_00>theta_0` 时公式可能超过 `theta_max`，最终仍会限幅。三个尾鳍的绝对角度为各自 `center_angle+theta`，共用偏移取三个物理行程的交集，并再限制在±`theta_max`内。`theta_zuo/theta_you`名称和符号遵循此实验公式；不依据其他脚本里的舵机方向说明擅自翻转符号。

应确认鱼向+y转动时标定后yaw增大；安装方向相反可设 `yaw_sign: -1`。

`navigation.jsonl`及`commands.jsonl`中的决策包含`control_mode`（`pi`、`turn_positive`、`turn_negative`）、`pi_output`（带符号e）和`effort`（abs(e)），导航日志的`integral`也为带符号积分项。强转向时PI输出和积分均为0；日志中的 `theta_left/theta_right` 保留旧字段名，分别表示第一、第二动作角度，强转向时同号。PI公式沿用当前代码的方向关系，上述公式已同步修正文档中旧的方向说明。两种模式都受±`theta_max`及三个舵机物理行程交集约束，因此较小限位可能使30°/20°动作被缩幅。

## 4. UWB触发与停止

- PI模式正常测距时，一个新有效定位最多启动一组 `action1(theta_zuo,t_0)` → `action1(theta_you,t_0)`；模式切换时允许使用仍有效的当前定位立即启动新组。强转向持续往复，不必等待新UWB帧，但仍检查断流超时。第二个动作必须等第一个动作端点PWM确认后才提交。
- 同一PI组内角度固定；每个动作结束时检查是否需要切换模式。采集和IMU滤波持续运行，新UWB更新决策和到达判定，不积压过期动作。缺基站期间继续用上一完整定位，每组间结合最新IMU重新决策并继续动作，直到收到新完整有效测距；完整定位的原始时间戳保留，日志标明复用及年龄。
- `t_0>0` 沿用现有五次平滑轨迹；`t_0=0` 在下一个50Hz执行周期直接写目标。左、右仍分属两次已确认的写入，并非物理瞬时完成。
- 只要 `distance<delta_s` 就进入保持状态，不再要求航向误差满足阈值，允许打断当前组。在下一个PWM执行周期直接写入三个尾鳍各自的 `center_angle`，不做平滑插值、不等待当前动作完成，然后保持中位并继续采集和检查IMU/UWB。当前配置中位为85°、95°、95°；这是立即下发中位指令，机械转动仍需实际响应时间。若启动时已到点，则暂不初始化PWM。`delta_yaw` 仅保留以兼容旧配置和命令，不影响到点判定；航向PI仍用于距离范围外的转向控制。
- 保持期间收到新的有效UWB定位且 `distance>=delta_s` 时自动恢复导航，从左摆开始一组新动作，不补发被打断组合的右摆。只改变航向而位置仍在距离容差内，不触发恢复，以避免近点方位角噪声造成反复游动。暂停时清除PI积分，恢复时重新计算。
- 缺基站报文记录 `uwb_incomplete` 并沿用旧完整定位，不打断动作。真正断流、串口错误、缓冲缺口、报文格式错误或定位不合理时记录 `uwb_unavailable`，取消当前组合，直接写中位并持续保持，不退出。收到完整、有效且未重复的新定位后记录 `uwb_recovered`，清除PI积分重新决策；仍在目标距离范围内则继续保持，否则从左摆开始新组合。标定后一直没有完整有效定位时，超过 `uwb_timeout_s` 也会启用尾鳍中位保持。
- IMU样本超时、采集线程致命错误、PWM写入失败、日志失败、Ctrl+C仍会彻底停止并退出。上述监测在保持和等待UWB期间仍有效。退出时关闭尾鳍PWM，不再额外回中。
- 可选 `--t 30` 设置整个定点保持任务的时间上限30秒（兼容别名 `--max-control-s`）。从首次运动、初始到点保持或首次UWB异常回中完成开始计时，不含t1、t2和此前的初始化时间。到点暂停、UWB异常等待也计时，恢复不重置计时器；到时中止动作、关闭PWM并退出，不再恢复。独立定时线程发出停车请求，实际关断存在操作系统调度和硬件I/O延迟。
- YAML中的 `t: null` 表示不设时间上限，持续执行“导航—到点暂停—漂移恢复”，直到故障或人工停止。

## 5. 运行与调参

### 顺序导航到多个点（新增）

每个 `--waypoint X Y SECONDS` 表示一个目标点：X、Y单位为米，SECONDS为该段持续秒数。重复填写即可按命令中的顺序连续执行；只填写一次也可以。所有点必须在 `[0,L] × [0,W]` 内，每段时间必须为正有限数。

例如先以 `(3,1)` 为目标运行30秒，再以 `(5,2)` 为目标运行40秒，最后以 `(2,3)` 为目标运行20秒：

```bash
python scripts/run_surface_navigation_20261010.py \
  --waypoint 3 1 30 --waypoint 5 2 40 --waypoint 2 3 20 \
  --theta-max 25 --theta-0 10 --theta-00 10 --t-0 0.3 \
  --kp 1 --ki 0.05 --delta-s 0.2 --imu-filter-tau-s 0.15 \
  --t1 10 --t2 5 --confirm MOVE
```

- 全程只执行一次人工放置和IMU标定，切换目标时不重新标定。
- 第一段沿用第4节的控制计时起点；随后按各段累计时长切换。例中控制开始后的0～30秒、30～70秒、70～90秒分别对应三个目标。
- **每段时间从该段开始计算，包含前往目标、到点保持、漂移恢复和UWB异常等待；不是到达后才开始计时。** 提前到点时立即回中保持，漂出距离阈值后继续调节，直到本段到时。未到达也按时切换下一点。
- 切点时取消旧目标的未完成摆尾，直接回中，清除PI历史，再依据当前有效定位和新目标计算动作。各段间不重新初始化传感器；所有控制增益、幅值和阈值共用原参数。最后一段结束后关闭尾鳍PWM并退出。
- 每段到期由独立定时器请求回中，PWM仍由原执行线程写入；即使终端输出暂时阻塞，也不会因此一直执行旧目标动作。主线程恢复后按累计时长选择当前目标，已过期的段不补时。实际回中和关PWM仍存在系统调度及硬件I/O延迟。
- 使用 `--waypoint` 时，点表覆盖YAML的单点 `x_d/y_d/t`；L/W/H等其他参数仍按原优先级读取。**不要再同时传 `--x-d`、`--y-d` 或 `--t`（含别名）**，否则启动前报错；总时长就是各段时间之和。
- 点表目前通过命令行提供，无须修改robot.yaml结构。配合 `--dry-run` 可模拟验证；配合 `--no-servo` 仍只采集真实传感器、不执行导航或驱动舵机，采集时长为各段时间之和。
- 运动日志的 `metadata.yaml` 保存完整点表；`events.jsonl` 记录 `waypoint_started/waypoint_completed/route_completed`，其中completed表示该段时间结束，不保证到达。事件、导航、命令日志均标注从1开始的 `waypoint_index` 和当前 `x_d/y_d`。

原单点用法完全保留：

```bash
python scripts/run_surface_navigation_20261010.py --x-d 3 --y-d 1 --t 60 --confirm MOVE
```

也可以用 `--waypoint 3 1 60 --confirm MOVE` 运行一个点。需要单点不限时保持时，继续使用原 `--x-d/--y-d` 用法，省略 `--t` 并在YAML中设置 `t: null`。

下面26个导航参数都能在`robot.yaml → surface_navigation_20261010`设置；优先级为**命令行 > YAML > 代码默认值**。带下划线的名称同时支持连字符和下划线命令，如`--theta-max`与`--theta_max`等价；`--L`、`--W`、`--H`保持大写。所有数值必须有限。修改YAML后须重新启动脚本，不会在运行中热加载。

### 5.1 全部导航参数

“当前值”按仓库现有YAML列出；除四个必填项外，与当前代码默认值一致。

| YAML参数 | 命令行 | 当前值 | 单位、作用与限制 |
|---|---|---|---|
| `L` | `--L` | null，必填 | 米，A0→A1的矩形长度，>0 |
| `W` | `--W` | null，必填 | 米，A0→A3的矩形宽度，>0 |
| `H` | `--H` | 1.93 | 米，基站与水面标签天线垂直高度差，≥0；旧配置省略/null时沿用未知共同高度校验 |
| `x_d` | `--x-d` | null，必填 | 目标x坐标，米，0≤x_d≤L |
| `y_d` | `--y-d` | null，必填 | 目标y坐标，米，0≤y_d≤W |
| `theta_max` | `--theta-max` | 30 | 最大尾鳍相对幅值，度，0<theta_max≤30；最终还受三个舵机物理限位约束 |
| `theta_0` | `--theta-0` | 10 | 左摆基础幅值的正数大小，度；实际基础左摆为−theta_0；0≤theta_0<theta_max |
| `theta_00` | `--theta-00` | 10 | 右摆基础幅值，度；0≤theta_00<theta_max；与theta_0不同可设置左右偏置 |
| `t_0` | `--t-0` | 0.3 | 每个action1时长，秒，≥0；往复一组含两次动作；0表示直接写目标，不插值 |
| `t1` | `--t1` | 15 | 人工放置等待，秒，≥0；不初始化运动PWM |
| `t2` | `--t2` | 5 | 静止IMU标定窗口，秒，>0；窗口须有足够样本且时间覆盖至少80%，不得出现超时采样间断 |
| `t` | `--t` / `--max-control-s` | null | 任务总时间上限，秒，给值必须>0；null不限时。包含到点暂停和已开始控制后的UWB异常等待，计时规则见第4节 |
| `kp` | `--kp` | 1 | 航向PI比例增益，≥0；增大后同一角差产生更大转向幅值差 |
| `ki` | `--ki` | 0 | 航向PI积分增益，1/s，≥0；0关闭积分；按控制决策时间间隔积分 |
| `yaw_error_scale_deg` | `--yaw-error-scale-deg` | 180 | 归一化角差尺度，度，0<值≤180；归一化值为min(1,abs(error)/该值)，越小越早达到最大输入 |
| `delta_s` | `--delta-s` | 0.2 | 目标距离阈值，米，>0；距离<该值立即回中保持，距离≥该值恢复导航 |
| `delta_yaw` | `--delta-yaw` | 10 | 旧航向阈值，度，仍要求0<值≤180，但**不再参与到点暂停或恢复判定** |
| `imu_filter_tau_s` | `--imu-filter-tau-s` | 0.15 | IMU低通时间常数，秒，≥0；越大越平滑、响应越慢；0关闭低通 |
| `yaw_sign` | `--yaw-sign` | 1 | 只能为1或−1；使向+y转动时标定后yaw增大 |
| `calibration_max_std_deg` | `--calibration-max-std-deg` | 3 | 标定窗口三轴角度标准差上限，度，>0；越小越严格，任一轴超限则标定失败退出 |
| `calibration_min_samples` | `--calibration-min-samples` | 20 | 标定最少IMU样本数，整数≥2；需与t2及真实采样率配合 |
| `imu_timeout_s` | `--imu-timeout-s` | 0.5 | IMU有效样本最大年龄，秒，>0；超时退出，不自动恢复 |
| `uwb_timeout_s` | `--uwb-timeout-s` | 2 | 秒，>0；正常时检查定位年龄，缺基站复用时检查mc接收间隔。真正断流超时回中等待；格式/几何错误立即回中 |
| `boundary_tolerance_m` | `--boundary-tolerance-m` | 0.25 | 兼容旧参数，仍须为非负有限数；已取消定位越界判断和裁剪，此值不再影响定位 |
| `range_residual_limit_m` | `--range-residual-limit-m` | 0.5 | 兼容旧参数，仍须为正有限数；已取消测距残差检查，此值不再影响定位 |
| `uwb_tag` | `--uwb-tag` | `t0:0` | 鱼的UWB标签；格式为`t数字:数字`；只接收该标签mc帧 |

若YAML已经写了有限的`t`，仅省略命令行`--t`不会取消时间限制；要恢复不限时须把YAML改回`t: null`。命令行不接受`--t null`或`--t 0`。

### 5.2 入口参数

| 参数 | 默认值 | 作用 |
|---|---|---|
| `--config` | `config/robot.yaml` | 指定YAML文件；相对路径按当前工作目录解析 |
| `--dry-run` | 不启用 | 模拟采集与PWM，不连接硬件；不会自动缩短t1/t2，需显式传参 |
| `--no-servo` | 不启用 | 只测试真实IMU/UWB，不创建舵机控制器/执行器、不写PWM；与dry-run互斥 |
| `--confirm MOVE` | 未确认 | 实际运动必填；模拟和no-servo不需要 |
| `-h` / `--help` | — | 显示所有选项并退出 |

本脚本没有`--rl-config`、`--keep-pwm`；不读取RL训练配置。与下潜脚本不同，水面脚本`--t`是总时长，尾鳍单动作时间使用`--t-0`。

### 5.3 其他生效的robot.yaml参数

以下项不写在`surface_navigation_20261010`内，也没有本脚本的命令行别名。

| YAML路径 | 当前值/默认值 | 作用 |
|---|---|---|
| `runtime.log_dir` | `logs` | 日志根目录 |
| `sensors.imu.port` / `baudrate` | `/dev/ttyAMA4` / 460800 | IMU串口及波特率 |
| `sensors.imu.sample_rate_hz` | 200 | IMU采集配置频率，须与设备支持值匹配 |
| `sensors.imu.serial_timeout_s` | 未设置，默认0.1 | 串口单次读超时秒数；不同于导航的imu_timeout_s |
| `sensors.imu.buffer_size` | 1000 | IMU缓存样本数量，正整数 |
| `sensors.uwb.port` / `baudrate` | `/dev/ttyAMA0` / 115200 | UWB串口及波特率 |
| `sensors.uwb.serial_timeout_s` | 未设置，默认0.2 | UWB串口读超时秒数；不同于有效定位超时 |
| `sensors.uwb.buffer_size` | 300 | UWB缓存容量，正整数；片段丢失会丢弃未完成帧并回中等待 |
| `servo.enable_pin` | 4 | 舵机使能GPIO；null表示不使用该引脚 |
| `servo.pca9685_channels` | 未设置，默认16 | ServoKit通道总数 |
| `servo.pulse_min_us` / `pulse_max_us` | 500 / 2500 | 全局舵机脉宽端点，可在各通道用同名参数覆盖 |
| `servo.channels[].servo_id` / `channel` | 1/0、2/1、3/2 | 三个尾鳍的逻辑编号及PWM通道映射 |
| `servo.channels[].center_angle` | 85/95/95 | 三个尾鳍的物理中位，度；暂停及UWB异常时写入这些角度 |
| `servo.channels[].min_angle` | 55/65/65 | 三个尾鳍的最小物理角度，度 |
| `servo.channels[].max_angle` | 115/125/125 | 三个尾鳍的最大物理角度，度 |
| `servo.channels[].actuation_range` | 180 | 舵机驱动角度量程，度 |

尾鳍最终目标是`center_angle + theta`，不是直接把theta当物理角度。水面脚本不使用通道`sign`翻转这组角度；航向测量方向用`yaw_sign`处理。

程序强制环境为surface，只开启IMU/UWB、关闭深度/电源/相机，因此修改这些传感器的enabled不能增加控制源。UWB更新频率由模块输出决定，没有导航参数能将它设成IMU频率。

执行器固定50Hz、初始化插值0秒、退出不额外平滑回中、主循环等待约0.005秒；不读取`servo.discrete_executor`的同名设置。`servo.frequency_hz`当前未传给ServoKit构造器。`sensors.*.timeout_ms`及`robot.safety.stop_on_*`不替代本表中的导航超时参数；电压、电流、深度保护未在水面脚本启用。`runtime.print_hz`、`synchronized_sample_hz`和通用logging文件名也不控制本脚本输出（控制台约每秒一次，日志文件见第6节）。

### 5.4 运行示例

只测试真实IMU/UWB，采集30秒，不向舵机下发任何指令：

```bash
python scripts/run_surface_navigation_20261010.py --no-servo --t 30
```

此模式跳过放置等待和IMU归零标定，`t1/t2`不生效，`t`从采集启动开始计时；无有效UWB或IMU数据也能按时结束。原始IMU角度未归零。端口及场地参数仍从YAML读取，定位仍使用L/W/H校验；不会写入回中或关PWM指令，也不会改变此前其他进程留下的舵机状态。

仅模拟，不访问任何硬件：

```bash
python scripts/run_surface_navigation_20261010.py --dry-run --L 4 --W 3 --x-d 3 --y-d 1 --t1 0 --t2 .5 --max-control-s 3
```

模拟器使用固定坐标 `(L/4,W/2)`、约200Hz IMU和10Hz UWB，通过真实解析器及动作执行线程运行，用于检查流程和输出，不模拟水动力学，也不证明实际收敛。Windows定时精度较低，极短标定窗口可能样本不足，此时增大 `t2`。

填好YAML后实机运行：

```bash
python scripts/run_surface_navigation_20261010.py --confirm MOVE
```

总运动时间最多30秒，到时无论是否到达目标都停止：

```bash
python scripts/run_surface_navigation_20261010.py --t 30 --confirm MOVE
```

覆盖实验参数示例（`L/W`从YAML读取）：

```bash
python scripts/run_surface_navigation_20261010.py --x-d 3 --y-d 1 --theta-max 25 --theta-0 10 --theta-00 11 --t-0 .3 --kp 1 --ki .05 --delta-yaw 10 --delta-s .2 --t1 20 --t2 5 --confirm MOVE
```

采集端口和波特率仍使用 `sensors.imu`、`sensors.uwb`。此脚本只开启这两个采集源，不依赖深度、相机、电源传感器；只监测本任务所需的IMU/UWB有效性及执行/日志故障，没有复用下潜脚本的深度、电压、电流保护。

## 6. 日志和测试

日志目录：`logs/surface_navigation_20261010/<时间戳>_surface_navigation/`。

| 文件 | 内容 |
|---|---|
| `metadata.yaml` | 实际生效的参数和采集配置 |
| `calibration.yaml` | IMU零点、标准差和样本数量 |
| `raw_imu.jsonl`、`raw_uwb.jsonl` | 原始采集样本 |
| `imu_filtered.jsonl` | 归零滤波后的三轴角度 |
| `navigation.jsonl` | 定位、目标航向、误差、PI输出、限幅及到达判定 |
| `commands.jsonl` | 两个action1的请求及关联UWB帧 |
| `events.jsonl` | 阶段、动作启动/完成、无效定位 |
| `result.json` | 退出原因、完成组合数、故障和清理结果 |

运动模式的 `navigation.jsonl`、`commands.jsonl` 增加 `position_reused` 和 `position_age_s`，保留源完整测距的帧ID。`uwb_incomplete` 事件记录当前不完整距离与沿用的源帧。

`--no-servo` 模式输出 `raw_imu.jsonl`、`raw_uwb.jsonl`、`uwb_diagnostics.jsonl`、`navigation.jsonl`、`metadata.yaml`、`result.json`。其中诊断日志兼容UWB距离绘图脚本；坐标日志同时保存当前测距、上一有效坐标、复用标记及年龄。不生成动作命令或IMU归零滤波日志。

```bash
python -m pytest tests/test_surface_navigation_20261010.py tests/test_surface_navigation_session_20261010.py -q
```

自动化验证不涉及实机下水；实际转向效果和PI参数仍需结合水池测量验证。

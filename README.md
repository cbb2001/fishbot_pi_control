# fishbot_pi_control

树莓派机器鱼控制与数据采集项目。当前阶段的重点是：在树莓派 5 Debian Trixie Lite 上稳定采集多传感器数据，保存本地日志，并为后续预设动作实验、运动学建模和水动力学建模提供统一时间戳的数据。

当前阶段不是强化学习，不是完整闭环控制，也不是在线遥控机器人。默认入口不会驱动舵机，不会运行 gait，不会调用 RL policy，也不会默认输出 PCA9685 PWM。

## 首先确认树莓派当前 IP

树莓派的 IP 地址不是固定的，不要假设上一次使用的地址仍然有效。每次更换网络或重新连接 Wi-Fi/网线后，先在 Windows PC 和树莓派连接到同一个局域网，再重新查找树莓派 IP。

推荐流程：

1. 在 Windows PC 上查看当前局域网网段：

```powershell
ipconfig
```

找到当前正在使用的网卡 IPv4 地址。例如：

```text
IPv4 Address . . . . . . . . . . . : 10.170.225.113
```

这说明当前局域网网段通常是：

```text
10.170.225.0/24
```

2. 扫描当前网段中开放 SSH 端口 22 的设备：

```powershell
nmap -Pn -p 22 --open 10.170.225.0/24
```

如果当前 PC 地址是 `192.168.1.xxx`，则扫描：

```powershell
nmap -Pn -p 22 --open 192.168.1.0/24
```

3. 在扫描结果中找到树莓派 IP，并先测试 SSH：

```powershell
ssh fish@<树莓派IP>
```

4. 后续所有同步和远程运行命令都使用这个刚发现的 IP。例如：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\sync_to_pi.ps1 -PiIp <树莓派IP>
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/test_sensor_pipeline.py
```

如果 Windows 没有 `nmap` 命令，需要先安装 Nmap，或使用项目自带的 `codex_pi_workflow/find_pi.ps1` 辅助查找。

## 当前阶段目标

1. 稳定采集 IMU、UWB、深度传感器、功率传感器、USB 双目摄像头。
2. 将各传感器数据统一为标准消息格式。
3. 使用 `time.monotonic_ns()` 作为主时间戳。
4. 使用 RingBuffer 缓存每个传感器最近数据。
5. 以固定频率生成 `synchronized_sensors.jsonl`。
6. 将同步数据、原始传感器数据、摄像头图片和事件日志保存到树莓派本地 `logs/`。
7. 水下 Wi-Fi/SSH 断开后，树莓派仍能独立继续记录。
8. 后续再加入更多预设动作，用于建模和控制研究。

## 工作流约束

Windows 本机只用于编辑代码、静态检查、同步代码。树莓派 5 是唯一硬件执行端。

不要在 Windows 本机直接运行 I2C、UART、GPIO、PCA9685、摄像头相关硬件代码。硬件脚本必须通过：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> <script> <args>
```

同步代码到树莓派：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\sync_to_pi.ps1 -PiIp <树莓派IP>
```

如果树莓派系统 Python 已经安装了硬件库和 OpenCV，而虚拟环境缺少依赖，可以使用 `-NoVenv`：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/test_sensor_pipeline.py
```

同步脚本主要同步以下路径：

```text
main.py
requirements.txt
config/
drivers/
control/
scripts/
missions/
```

因此新增运行时代码优先放在 `control/runtime/`、`scripts/`、`missions/`、`config/`、`drivers/`。

## 项目结构

```text
fishbot_pi_control/
  config/
    robot.yaml                    # 机器人、传感器、舵机、日志、安全配置
  control/
    safety.py                     # 配置读取、安全限制、Windows 硬件运行保护
    pid.py
    gait.py                       # 当前阶段默认不使用
    runtime/
      sample.py                   # SensorSample / CommandSample / TrialEvent
      ring_buffer.py              # 线程安全固定长度 RingBuffer
      sensor_worker.py            # 传感器 worker 基类
      i2c_worker.py               # 深度和功率传感器共用 I2C 轮询线程
      uart_workers.py             # IMU / UWB 串口 worker
      camera_worker.py            # 摄像头采集和异步 JPEG 保存
      sensor_manager.py           # 创建和管理所有 worker
      sensor_synchronizer.py      # 固定频率生成同步样本
      data_logger.py              # 异步 JSONL 日志、raw 日志、日志目录创建
      event_logger.py             # events.jsonl 事件日志
      link_manager.py             # 网络状态事件记录，不触发停机
      discrete_actions.py         # action1/2/3 数学、标定读取和完整轨迹预检
      manual_action_provider.py   # 人工 YAML ActionProvider
      action_scheduler.py         # 三路并行、组内串行调度
      servo_state_tracker.py      # 无反馈参考/命令/估计状态历史
      servo_executor.py           # 唯一 PCA9685 所有者线程
      sensor_sync_worker.py       # 独立同步采样和 raw 日志搬运线程
  drivers/
    depth_sensor.py               # MS5837 深度传感器项目封装
    ina219.py                     # DFRobot INA219 adapter
    pca9685_servo.py              # PCA9685 舵机控制器
    DFRobot_MS5837.py
    DFRobot_INA219/
    Yesense-Decode-Python3-V2.0/
    uwb-read-python/
  scripts/
    test_sensor_pipeline.py       # 传感器管线状态测试
    run_sensor_observe.py         # 实时 observe
    record_sensors.py             # 本地传感器记录
    start_offline_recording.py    # 离线 detached 记录启动器
    check_latest_log.py           # 检查最新日志
    run_action_pectoral_tail_1.py # 动作 1：侧鳍尖端镜像倾斜 + 尾鳍周期摆动
    run_action_pectoral_tail_2.py # 动作 2：侧鳍上下拍动 + 尾鳍周期摆动
    run_action_smooth_pectoral_tail_20260713.py # 左右胸鳍光滑拍动 + 尾鳍正弦不等幅摆动
    run_discrete_action_sequence_20260716.py # 人工离散动作序列入口
    run_mission_dive_cruise_surface.py # 综合任务：下潜-巡航-上浮
    calibrate_servo.py            # 单舵机标定
    test_servo_center.py
    test_i2c.py
    test_uart.py
    test_usb_camera.py
  codex_pi_workflow/
    sync_to_pi.ps1
    run_on_pi.ps1
    setup_pi_venv.ps1
  missions/
    discrete_action_example.yaml # 人工离散动作示例
  main.py                         # sensor status / observe / record CLI
  requirements.txt
```

## 硬件接口配置

所有硬件配置集中在 `config/robot.yaml`。

当前运行配置摘要：

```yaml
runtime:
  mode: sensor_test
  environment: underwater
  mock: false
  log_dir: logs
  synchronized_sample_hz: 30
  print_hz: 2
  flush_interval_s: 1.0
  offline_recording: true
  telemetry_enabled: false

logging:
  base_dir: logs
  synchronized_file: synchronized_sensors.jsonl
  events_file: events.jsonl
  camera_index_file: camera_index.jsonl
  metadata_file: metadata.yaml
```

### IMU

```text
接口：UART4
设备：/dev/ttyAMA4
波特率：460800
例程：drivers/Yesense-Decode-Python3-V2.0
```

要求：

- 固定使用 `/dev/ttyAMA4`。
- 不要写成 `/dev/ttyAMA0` 或 `/dev/serial0`。
- IMU 是高频传感器，按串口数据到达频率读取。
- 当前代码复用 Yesense 解码器，输出标准字段：
  - `acc_mps2`
  - `gyro_radps`
  - `roll_deg`
  - `pitch_deg`
  - `yaw_deg`
  - `quat`
  - `temperature_c`
  - `raw_hex`
- 注意：当前代码按 Yesense 输出中 `acc` 为 g、`gyro` 为 deg/s 进行单位换算，仍需要结合实物和厂家文档最终确认。

### UWB

```text
接口：UART0
设备：/dev/ttyAMA0
波特率：115200
例程：drivers/uwb-read-python
```

要求和注意事项：

- 固定使用 `/dev/ttyAMA0`。
- UWB 在水下基本没有可靠信号，这是正常情况。
- UWB 是 optional sensor。
- UWB invalid、timeout、`LO=[no solution]` 不会让程序崩溃，也不会触发 emergency stop。
- 当前 UWB worker 会记录 `raw_hex` 和 `raw_text`，初版不强行解析坐标。
- 如果 `/dev/ttyAMA0` 被 Linux serial console 占用，在树莓派上执行：

```bash
sudo systemctl stop serial-getty@ttyAMA0.service
sudo systemctl disable serial-getty@ttyAMA0.service
```

### 深度传感器

```text
接口：I2C
传感器：MS5837 类深度计
Linux 7-bit 地址：0x76
默认采样频率：20 Hz
当前水面压力基准：1144.0 mbar
例程：scripts/test_depth_sensor.py
```

注意：

- STM32 例程里的写地址 `0xEC`、读地址 `0xED` 对应 Linux/smbus 的 7-bit 地址 `0x76`。
- 树莓派代码使用 `0x76`。
- 标准字段：
  - `depth_m`
  - `pressure_pa`
  - `temperature_c`
- 如果看到负深度，通常和当前 surface pressure / zero 设置有关，不代表日志系统错误。后续实验前可以根据实际水面环境做深度零点校准。

### 功率传感器

```text
接口：I2C
传感器：INA219
地址：0x45
默认采样频率：2 Hz
例程：drivers/DFRobot_INA219/Python/RespberryPi/examples/get_voltage_current_power
```

注意：

- 功率传感器不需要高频采集。
- 不强行让 power 跟 `synchronized_sample_hz` 一样高频。
- 同步样本里使用最近一次 power 数据，并记录 `age_ms` 和 `valid`。
- 标准字段：
  - `voltage_v`
  - `current_a`
  - `power_w`
- 当前 `drivers/ina219.py` 是对 DFRobot 供应商驱动的 adapter。实际校准参数和电流方向需要结合实物确认。

### USB 双目摄像头

```text
接口：USB
设备：/dev/video*
当前配置：enabled: true
采集频率：rate_hz: 15
图片保存频率：save_fps: 20
```

配置：

```yaml
sensors:
  vision:
    enabled: true
    camera_index: 0
    width: 2560
    height: 960
    fourcc: MJPG
    rate_hz: 15
    buffer_size: 100
    timeout_ms: 300
    save_frames: true
    save_fps: 20
    image_format: jpg
    jpeg_quality: 85
    save_combined_frame: true
    split_stereo: false
    max_image_queue_size: 100
```

摄像头规则：

- 摄像头仍是 optional sensor。
- 如果 `cv2` 不存在，vision enabled 时会输出 `ok=false, error="opencv_not_available"`，系统继续运行。
- 摄像头当前请求 combined frame `2560x960`，格式 `MJPG`。
- 摄像头采集频率 `rate_hz=15` 不等于图片保存频率。
- JPEG 保存频率由 `save_fps=20` 限制，即每秒最多保存 20 张；实际保存速度仍受 USB 带宽、OpenCV 解码和写盘速度影响。
- 图片保存为 combined frame，例如双目左右拼接的 `2560x960` 当前先整体保存。
- 当前不做 left/right 分割，后续再扩展。对于 `2560x960` combined frame，左右单目约为 `1280x960`。
- 图片二进制不会写入 `synchronized_sensors.jsonl`。
- 图片保存由 `ImageWriter` 后台线程完成，写盘慢时可以丢弃图片保存任务，但不能影响 IMU/depth/power/UWB 采集。

## 传感器采集和同步机制

传感器系统分两层：原始采集层和同步样本层。

### 原始采集层

每类传感器由独立 worker 或调度线程采集：

- IMU：UART worker，高频读取。
- UWB：UART worker，timeout 不阻塞主流程。
- Depth/Power：同一个 I2C worker 轮询，避免两个线程抢 I2C 总线。
- Camera：camera worker 读取帧，ImageWriter 后台保存 JPEG。

每个传感器样本统一为：

```python
SensorSample(
    name: str,
    t_ns: int,
    seq: int,
    data: dict,
    ok: bool = True,
    error: str | None = None,
)
```

其中 `t_ns` 使用：

```python
time.monotonic_ns()
```

### RingBuffer

每个传感器有自己的固定长度 RingBuffer：

- 满了丢弃最旧数据。
- 不使用无限队列。
- 高频 IMU 不会导致内存无限增长。
- 支持 `latest()`、`get_latest_before()`、`get_nearest()`、`get_bracketing()`、`snapshot()`。

### 同步样本层

`SensorSynchronizer` 按 `runtime.synchronized_sample_hz` 固定频率生成统一样本，当前配置为：

```yaml
runtime:
  synchronized_sample_hz: 30
```

当前同步策略是“最近值同步”：

- 每次构建 synchronized sample 时，从每个 RingBuffer 取最近一条数据。
- 每个传感器字段包含：
  - `valid`
  - `age_ms`
  - `error`
  - `interpolated`
  - `interpolation_gap_ms`
  - `seq`
  - `sample_t_ns`
  - `data`
- 某个传感器 invalid 不会停止输出。
- UWB invalid 在水下是正常情况。
- vision disabled 或 OpenCV 不可用时系统继续运行。

这不是硬实时硬件触发同步，而是“并行原始采集 + 统一 monotonic 时间戳 + 固定频率最近值同步”。当前阶段足够用于前期运动学和水动力学建模数据采集。

## 日志目录结构

每次 record 或动作测试会在树莓派本地创建日志目录：

```text
logs/YYYYMMDD_HHMMSS_sensor_test/
```

摄像头动作或普通 record 的典型结构：

```text
logs/YYYYMMDD_HHMMSS_sensor_test/
  metadata.yaml
  synchronized_sensors.jsonl
  events.jsonl
  camera_index.jsonl
  raw_imu.jsonl
  raw_depth.jsonl
  raw_power.jsonl
  raw_uwb.jsonl
  raw_vision.jsonl
  camera/
    frame_00000001.jpg
    frame_00000002.jpg
```

动作脚本还会额外生成：

```text
commands.jsonl
```

### synchronized_sensors.jsonl

保存固定频率的统一同步样本。示例结构：

```json
{
  "t_ns": 123456789,
  "t_s": 12.345,
  "imu": {
    "valid": true,
    "age_ms": 4.1,
    "error": null,
    "data": {}
  },
  "depth": {
    "valid": true,
    "age_ms": 20.5,
    "depth_m": 0.42,
    "depth_rate_mps": 0.01,
    "data": {}
  },
  "power": {
    "valid": true,
    "age_ms": 310.0,
    "voltage_v": 12.1,
    "current_a": 0.8,
    "power_w": 9.68,
    "data": {}
  },
  "uwb": {
    "valid": false,
    "age_ms": null,
    "error": "no_signal_or_timeout",
    "data": {}
  },
  "vision": {
    "valid": true,
    "age_ms": 32.5,
    "data": {
      "frame_id": 120,
      "width": 2560,
      "height": 960,
      "mean_brightness": 83.5,
      "image_file": "camera/frame_00000120.jpg",
      "image_t_ns": 123456789000
    },
    "error": null
  },
  "status": {
    "environment": "underwater",
    "all_sensors_ok": false,
    "missing_sensors": ["uwb"],
    "warnings": []
  }
}
```

### raw_*.jsonl

保存每个传感器原始 `SensorSample` 流：

```text
raw_imu.jsonl
raw_depth.jsonl
raw_power.jsonl
raw_uwb.jsonl
raw_vision.jsonl
```

这些文件不是占位文件，record 模式下会写入实际样本。它们用于后续离线重同步、插值、协议解析和数据质量检查。

### camera_index.jsonl

记录每张 JPEG 图片的信息。示例：

```json
{
  "t_ns": 123456789000,
  "frame_id": 120,
  "filename": "camera/frame_00000120.jpg",
  "width": 2560,
  "height": 960,
  "format": "jpg",
  "jpeg_quality": 85,
  "saved": true,
  "error": null
}
```

`synchronized_sensors.jsonl` 里的 `vision.data.image_file` 会引用这里的相对路径。图片本体保存在 `camera/` 子目录。

### commands.jsonl

动作脚本保存每次舵机命令：

```json
{
  "t_ns": 123456789000,
  "action": "pectoral_tip_mirror_tail_sweep_1",
  "elapsed_s": 1.23,
  "tail_offset_deg": -10.5,
  "commands_deg": {
    "0": 70.0,
    "1": 80.0,
    "2": 80.0,
    "4": 120.0,
    "6": 60.0
  }
}
```

后续建模时，可以用 `commands.jsonl` 和 `synchronized_sensors.jsonl` 的 `t_ns` 对齐动作命令和传感器响应。

## main.py 用法

`main.py` 是传感器采集入口，支持三种模式：

```bash
python3 main.py --mode status
python3 main.py --mode observe
python3 main.py --mode record --duration 30
```

mock 模式不会访问 UART、I2C、GPIO、PCA9685 或摄像头：

```bash
python3 main.py --mode status --mock
python3 main.py --mode observe --mock
python3 main.py --mode record --mock --duration 10
```

### status

- 启动传感器几秒。
- 打印各 buffer 的 latest、age_ms、valid。
- 不驱动舵机。
- 常用于水面硬件检查。

### observe

- 固定频率生成 synchronized sample。
- 低频打印简短状态。
- 可写事件日志。
- 不驱动舵机。

### record

- 当前最重要的水下实验模式。
- 本地保存所有日志。
- 支持 `--duration`，到时自动结束。
- Ctrl+C 时优雅退出。
- SSH/Wi-Fi 断开后，只要进程是本地独立运行，树莓派会继续采集和写日志。
- 不调用 gait、RL、PCA9685 或舵机输出。

## 常用脚本

### 1. 同步代码

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\sync_to_pi.ps1 -PiIp <树莓派IP>
```

### 2. mock 测试传感器管线

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/test_sensor_pipeline.py --mock
```

### 3. 真实传感器状态检查

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/test_sensor_pipeline.py
```

该脚本会访问真实 IMU、UWB、depth、power、camera，但不驱动舵机。

### 4. 水面 observe

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/run_sensor_observe.py
```

### 5. 记录 30 秒传感器数据

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/record_sensors.py --duration 30
```

### 6. 检查最新日志

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/check_latest_log.py
```

输出会包含：

- 最新日志目录。
- `synchronized_sensors.jsonl` 是否存在。
- `camera_index.jsonl` 是否存在。
- `camera/` 下 jpg 数量。
- `raw_imu/depth/power/uwb/vision.jsonl` 的大小和行数。
- 估算图片保存 FPS。
- 最近几条 synchronized sample、events、camera index。

### 7. 离线记录

水下实验不能依赖 SSH 会话。推荐在树莓派上启动本地 detached 记录：

```bash
cd /home/fish/fishbot_pi_control
source .venv/bin/activate
nohup python3 scripts/record_sensors.py --duration 300 > logs/last_run.out 2>&1 &
```

或者使用脚本：

```bash
python3 scripts/start_offline_recording.py --duration 300
```

systemd service 安装：

```bash
bash scripts/install_recorder_service.sh
sudo systemctl start fishbot-recorder.service
sudo systemctl status fishbot-recorder.service
```

## 舵机配置

舵机配置在 `config/robot.yaml` 的 `servo.channels`。

当前 7 个舵机：

| 舵机 ID | 名称 | 角色 | PCA9685 channel | 当前中心 | 最小 | 最大 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | WQ1 | 尾部前段 | 0 | 80 | 50 | 110 |
| 2 | WQ2 | 尾部中段 | 1 | 90 | 60 | 120 |
| 3 | WQ3 | 尾部后段 | 2 | 90 | 60 | 120 |
| 4 | ZQ1 | 左胸鳍根部 | 3 | 121 | 68 | 174 |
| 5 | ZQ2 | 左胸鳍尖端 | 4 | 90 | 0 | 180 |
| 6 | YQ1 | 右胸鳍根部 | 5 | 143 | 90 | 196 |
| 7 | YQ2 | 右胸鳍尖端 | 6 | 90 | 0 | 180 |

方向定义：

- 从尾部往头看，1 号从左到右角度变小：`110 -> 80`。
- 从尾部往头看，2/3 号从左到右角度变小：`120 -> 60`。
- 4 号从上往下角度变大：`68 -> 174`。
- 6 号从上往下角度变小：`196 -> 90`。
- 5 和 7 号是左右鳍尖镜面对称：
  - 5 号角度变大时左侧鳍向前转。
  - 7 号角度变小时右侧鳍向前转。
  - 离散动作的物理上下标定为 5 号 `top=180 / bottom=0`、7 号 `top=0 / bottom=180`。

安全注意：

- 单舵机标定使用 `scripts/calibrate_servo.py`。
- 所有会移动舵机的脚本必须显式传 `--confirm MOVE`。
- 运行舵机前确认机械限位、供电、鱼体固定和水环境安全。

## 动作脚本

### 动作 1：侧鳍尖端镜像倾斜 + 尾鳍周期摆动

脚本：

```text
scripts/run_action_pectoral_tail_1.py
```

动作内容：

- 4、6 号舵机会被显式保持在配置的初始中心位置：
  - 4 号：`center_angle=114`
  - 6 号：`center_angle=140`
- 5、7 号舵机从初始状态镜面对称倾斜：
  - 5 号：`center + pectoral_tilt`
  - 7 号：`center - pectoral_tilt`
  - 默认 `pectoral_tilt=30`，即 5 号到 120 度、7 号到 60 度。
- 1、2、3 号尾鳍同步周期摆动：
  - 同相位正弦摆动。
  - 默认从右侧开始，随后向左摆动。
  - 频率可调：`--tail-frequency`
  - 幅度可调：`--tail-amplitude`
  - 脚本会检查幅度不超过各自安全范围。
- 动作期间传感器正常采集和记录。
- 额外保存 `commands.jsonl`。

第一次建议小幅度短时间测试：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/run_action_pectoral_tail_1.py --confirm MOVE --duration 5 --tail-frequency 0.3 --tail-amplitude 10 --pectoral-tilt 30
```

默认幅度测试：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/run_action_pectoral_tail_1.py --confirm MOVE --duration 10 --tail-frequency 0.5 --tail-amplitude 20 --pectoral-tilt 30
```

参数说明：

```text
--confirm MOVE          必须提供，否则拒绝移动舵机
--duration              动作持续时间，秒
--tail-frequency        尾鳍摆动频率，Hz
--tail-amplitude        尾鳍摆动幅度，度
--pectoral-tilt         5/7 号舵机镜像倾斜角，度
--command-hz            舵机命令更新频率
--hold-pwm              动作结束后保持 PWM，不释放
--mock-sensors          传感器用 mock 数据，但舵机仍会真实移动
```

日志目录示例：

```text
logs/YYYYMMDD_HHMMSS_pectoral_tip_mirror_tail_sweep_1/
```

动作结束后，脚本会把 1、2、3、4、5、6、7 号舵机回到中心位置，并根据配置释放 PWM。其中 4、6 号在动作全过程都会被保持在初始中心位置，避免停在上一次动作残留位置。

### 动作 2：侧鳍上下拍动 + 尾鳍周期摆动

脚本：

```text
scripts/run_action_pectoral_tail_2.py
```

动作内容：

- 4、6 号舵机作为左右胸鳍根部，做上下周期性拍动。
- 周期起点在最上方：
  - 4 号在 `top_reference_angle=68`
  - 6 号在 `top_reference_angle=196`
  - 5、7 号均为 `90`
- 前半周期为下拍：
  - 4 号从上方向下方运动，默认到 `bottom_reference_angle=174`
  - 6 号从上方向下方运动，默认到 `bottom_reference_angle=90`
  - 5、7 号保持 `90`
- 后半周期为上行回位：
  - 4、6 号从下方回到上方
  - 5 号以 `180` 开始上行阶段
  - 7 号以 `0` 开始上行阶段
  - 快到最上方时，5、7 号线性回到 `90`
- 1、2、3 号尾鳍与动作 1 一样同步正弦周期摆动。
- 动作期间传感器正常采集和记录。
- 额外保存 `commands.jsonl`。

注意：这里按镜像关系实现为“5 号上行阶段从 180 回 90，7 号上行阶段从 0 回 90”。

第一次建议用较小根部行程比例和小尾鳍幅度：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/run_action_pectoral_tail_2.py --confirm MOVE --duration 5 --pectoral-frequency 0.25 --pectoral-root-scale 0.5 --tail-frequency 0.3 --tail-amplitude 10
```

接近完整动作：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/run_action_pectoral_tail_2.py --confirm MOVE --duration 10 --pectoral-frequency 0.5 --pectoral-root-scale 1.0 --tail-frequency 0.5 --tail-amplitude 20
```

参数说明：

```text
--confirm MOVE             必须提供，否则拒绝移动舵机
--duration                 动作持续时间，秒
--pectoral-frequency       侧鳍上下拍动频率，Hz
--pectoral-root-scale      4/6 号从 top 到 bottom 的行程比例，0..1
--tip-return-fraction      上行阶段中 5/7 开始回到 90 的时刻，默认 0.8
--tail-frequency           尾鳍摆动频率，Hz
--tail-amplitude           尾鳍摆动幅度，度
--command-hz               舵机命令更新频率
--hold-pwm                 动作结束后保持 PWM，不释放
--mock-sensors             传感器用 mock 数据，但舵机仍会真实移动
```

日志目录示例：

```text
logs/YYYYMMDD_HHMMSS_pectoral_root_flap_tail_sweep_2/
```

动作结束后，脚本会把 1、2、3、4、5、6、7 号舵机回到中心位置，并根据配置释放 PWM。

### 综合任务：下潜 - 水下巡航 - 上浮

脚本：

```text
scripts/run_mission_dive_cruise_surface.py
```

用途：

- 将动作 1 和动作 2 组合成一个连续任务。
- 全程只创建一个日志目录，传感器连续采集。
- 全程持续写入 `synchronized_sensors.jsonl`、`commands.jsonl`、`events.jsonl`。
- 这是开环动作测试，不是自动深度闭环控制，也不是 RL policy。

三个 phase：

1. `dive`
   - 复用动作 1 类型。
   - 5 号舵机：`90 + dive_pectoral_tilt`
   - 7 号舵机：`90 - dive_pectoral_tilt`
   - 1、2、3 号尾鳍周期摆动。
   - 4、6 号舵机保持中位。
2. `cruise_underwater`
   - 复用动作 2 类型。
   - 4、6 号舵机上下周期拍动。
   - 5、7 号舵机按动作 2 的上行阶段规则配合。
   - 1、2、3 号尾鳍周期摆动。
3. `surface`
   - 使用动作 1 的反向攻角。
   - 5 号舵机：`90 - surface_pectoral_tilt`
   - 7 号舵机：`90 + surface_pectoral_tilt`
   - 1、2、3 号尾鳍周期摆动。
   - 4、6 号舵机回到中位。

建议先做不动舵机 dry-run，确认任务流程、日志和 commands 正常：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/run_mission_dive_cruise_surface.py --dry-run --mock-sensors --dive-duration 3 --cruise-duration 3 --surface-duration 3
```

第一次真实下水前，先在安全条件下做小幅短时间测试：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/run_mission_dive_cruise_surface.py --confirm MOVE --dive-duration 3 --cruise-duration 3 --surface-duration 3 --dive-pectoral-tilt 10 --surface-pectoral-tilt 10 --tail-amplitude 8 --pectoral-root-scale 0.3
```

较接近完整任务的示例：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/run_mission_dive_cruise_surface.py --confirm MOVE --dive-duration 10 --cruise-duration 20 --surface-duration 10 --dive-pectoral-tilt 20 --surface-pectoral-tilt 20 --tail-frequency 0.5 --tail-amplitude 20 --pectoral-frequency 0.5 --pectoral-root-scale 1.0
```

参数说明：

```text
--confirm MOVE              真实舵机动作必须提供，否则拒绝移动舵机
--dry-run                   不初始化 PCA9685，不发 PWM，只验证流程和日志
--mock-sensors              使用 mock 传感器；单独使用时舵机仍会真实移动
--dive-duration             下潜阶段持续时间，秒，默认 10
--cruise-duration           水下巡航阶段持续时间，秒，默认 20
--surface-duration          上浮阶段持续时间，秒，默认 10
--dive-pectoral-tilt        下潜攻角，5 增大、7 减小，默认 20 度
--surface-pectoral-tilt     上浮反向攻角，5 减小、7 增大，默认 20 度
--tail-frequency            1/2/3 尾鳍摆动频率，Hz
--tail-amplitude            1/2/3 尾鳍摆动幅度，度
--pectoral-frequency        cruise 阶段 4/6 侧鳍上下拍动频率，Hz
--pectoral-root-scale       cruise 阶段 4/6 从 top 到 bottom 的行程比例，0..1
--tip-return-fraction       cruise 阶段 5/7 开始回到 90 的上行时刻，默认 0.8
--command-hz                舵机命令更新频率
--transition-s              phase 之间平滑过渡时间，秒，默认 1.0
--hold-pwm                  任务结束后保持 PWM，不释放
```

日志目录示例：

```text
logs/YYYYMMDD_HHMMSS_mission_dive_cruise_surface/
```

任务结束或 Ctrl+C 中断后，脚本会尝试把 1、2、3、4、5、6、7 号舵机回到中心位置，并根据配置释放 PWM。实际能否稳定下潜和上浮取决于浮力、重心、速度、攻角、侧鳍频率和尾鳍推力，需要从小幅、短时、浅水测试逐步调参。

测试后检查最新日志：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/check_latest_log.py
```

### 左右光滑拍动测试：统一胸鳍时钟 + 不等幅尾鳍

脚本：

```text
scripts/run_action_smooth_pectoral_tail_20260713.py
```

这个脚本用于测试 4、5、6、7 号左右胸鳍的连续光滑运动，同时让 1、2、3 号尾鳍从中位开始做可配置的左右不等幅周期摆动。它是开环预设动作，不是强化学习、闭环姿态控制或在线遥控。

脚本继续使用现有 `SensorManager`、`SensorSynchronizer` 和日志系统，动作期间同时记录 IMU、UWB、Depth、Power、Camera。所有传感器样本、事件和舵机目标命令都使用 `time.monotonic_ns()` 时间基准。

#### 安全语义

三个参数的含义必须区分：

- `--dry-run`：不初始化 PCA9685，不向真实舵机发送 PWM；这是唯一明确禁止真实舵机动作的参数。
- `--mock-sensors`：只把 IMU、UWB、Depth、Power、Camera 换成 mock 数据；如果没有同时提供 `--dry-run`，舵机仍然是真实舵机。
- `--confirm MOVE`：所有非 dry-run 真实动作都必须显式提供，否则脚本在初始化真实舵机前退出。

默认动作结束后会平滑回中，并按照 `config/robot.yaml` 的 `release_pwm_after_tests` 设置释放 PWM。只有明确需要持续保持回中力矩时才使用 `--keep-pwm`；使用该参数后应准备好通过 `scripts/emergency_stop.py` 停止 PWM。

#### 启动和停止顺序

脚本按照以下顺序运行：

1. 读取 `config/robot.yaml`。
2. 一次性检查命令行参数和机械角度极限。
3. 执行 `--start-delay-s` 本地倒计时。
4. 倒计时结束后才创建实验日志目录。
5. 初始化并启动传感器、同步器和日志线程。
6. 初始化真实 PCA9685 控制器，或创建不接触硬件的 dry-run 控制器。
7. 首先只发送 1～7 号舵机的配置中位目标。
8. 使用五次 smoothstep 平滑进入动作初始姿态。
9. 保持 `--initial-hold-s` 基线时间。
10. 从胸鳍统一周期相位 0 开始动作。
11. 执行 `--duration` 指定的周期动作时间。
12. 使用五次 smoothstep 平滑减幅并回到安全中位。
13. 刷新并关闭全部日志，根据配置决定是否释放 PWM。

`--duration` 表示正式周期动作段的持续时间，不包含 start delay、进入初始姿态、initial hold 和回中时间。默认胸鳍频率为 `0.2 Hz`，周期为 `5 s`，默认 `duration=10 s`，因此正好执行两个完整胸鳍周期。建议将 duration 设置为：

```text
duration = N / pectoral_frequency_hz
```

其中 `N` 是正整数。若 duration 不是完整胸鳍周期，脚本会在指定时间结束周期段，然后从当时的目标位置平滑回中；不会直接停止尾鳍 PWM。

#### Start delay 入水前缓冲

默认：

```text
--start-delay-s 20
```

倒计时期间只在树莓派本地低频打印：

```text
Action starts in 20 s
Action starts in 19 s
...
```

倒计时期间：

- 不创建本次实验日志目录。
- 不启动传感器。
- 不初始化 PCA9685。
- 不执行舵机周期动作。
- 不等待 ping、SSH、电脑确认或任何网络条件。

可使用 `--start-delay-s 0` 关闭缓冲。该参数当前只接入本节的新动作脚本；`run_action_pectoral_tail_1.py`、`run_action_pectoral_tail_2.py` 和 `run_mission_dive_cruise_surface.py` 本次保持未修改。

#### 4、6 号胸鳍根部轨迹

4、5、6、7 号舵机共用唯一胸鳍时钟：

```text
T = 1 / pectoral_frequency_hz
t_cycle = action_elapsed_time % T
phase = t_cycle / T
```

始终满足：

```text
0 <= phase < 1
```

根部位置使用余弦轨迹：

```text
q = [1 - cos(2π t_cycle / T)] / 2
```

因此：

```text
q(0)   = 0     根部在物理最下方
q(T/2) = 1     根部在本次实际最高位置
q(T)   = 0     根部回到物理最下方
```

脚本不会假设左右根部舵机的数值方向相同。每个舵机分别读取配置中的 `bottom_reference_angle` 和 `top_reference_angle`，再计算：

```text
actual_top = bottom + root_amplitude_ratio × (top_reference - bottom)
theta_root = bottom + q × (actual_top - bottom)
```

当前 `config/robot.yaml` 和默认 `root_amplitude_ratio=0.5` 对应：

```text
4号：bottom=174°，配置物理上限=68°，本次 top=121°
6号：bottom=90°，配置物理上限=196°，本次 top=143°
```

这说明 4、6 号舵机在数值角度上沿相反方向运动，但在物理空间中同步从下向上再向下运动。

#### 5、7 号胸鳍尖端轨迹

5、7 号中性位置分别读取各自配置的 `center_angle`，不在代码中固定为 90°。尖端包络使用五次 smoothstep：

```text
S(u) = 10u³ - 15u⁴ + 6u⁵
```

每个胸鳍周期的前半周期执行：

1. 从中位平滑进入偏转。
2. 保持偏转。
3. 在半周期前平滑返回中位。
4. 后半周期保持中位。

目标角为：

```text
theta_5 = center_5 - tip_amplitude_deg × h
theta_7 = center_7 + tip_amplitude_deg × h
```

默认 `tip_amplitude_deg=20`，当前配置下目标为 5 号 `70°`、7 号 `110°`。脚本在倒计时和控制器初始化前检查这两个目标是否处于各自机械范围内。

尖端过渡时间必须满足：

```text
0 < tip_transition_s < 1 / (4 × pectoral_frequency_hz)
```

#### 1、2、3 号尾鳍轨迹

尾鳍从各自配置中位开始，不会预先移动到左极限或右极限。三个尾鳍舵机共享一个逻辑相位：

```text
s = sin(2π × tail_frequency_hz × action_elapsed_time)
```

逻辑偏移直接使用正弦值：

```text
s >= 0: tail_offset = tail_left_amplitude_deg × s
s < 0:  tail_offset = tail_right_amplitude_deg × s
```

使用正弦而不是 `sin³` 后，尾鳍经过中位时保持较高速度，不会因为 `sin³` 在零点斜率为零而产生明显停顿。左右幅度不相等时，目标角度在中位处仍连续，但中位交叉前后的目标速度允许发生跳变。

`--tail-first-direction right` 会反转第一次离开中位的方向，但仍从中位开始。每个尾鳍舵机再分别应用：

```text
target = center_angle + sign × amplitude_scale × tail_offset
```

其中 `center_angle`、`sign` 和 PCA9685 channel 来自配置；若配置没有 `amplitude_scale`，使用 `1.0`。左右两个极值直接按 `config/robot.yaml` 中每个舵机的 `min_angle` / `max_angle` 检查；这些配置限位已经包含机械安全余量，脚本不再二次内缩。

如果角度非法，脚本不会静默缩小幅度或在运行时 clamp。例如：

```text
servo_id=1 requested left amplitude=40.000 deg produces target=120.000 deg;
configured mechanical range from config/robot.yaml is 50.000..110.000 deg.
```

#### 参数和默认值

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--confirm` | 空 | 真实动作必须传 `--confirm MOVE` |
| `--dry-run` | false | 禁止初始化 PCA9685 和发送 PWM |
| `--mock-sensors` | false | 只模拟传感器，不模拟舵机 |
| `--keep-pwm` | false | 回中后保持 PWM；默认按安全配置释放 |
| `--duration` | `10.0` | 正式周期动作段时间，秒 |
| `--pectoral-frequency-hz` | `0.2` | 4、5、6、7 号共享胸鳍频率，Hz |
| `--root-amplitude-ratio` | `0.5` | 4、6 号各自最大安全物理行程比例，范围 `0..1` |
| `--tip-amplitude-deg` | `20.0` | 5、7 号相对各自中位的相反偏转幅度 |
| `--tip-transition-s` | `0.8` | 尖端进入或退出偏转的 smoothstep 时间 |
| `--tail-frequency-hz` | `0.3` | 1、2、3 号共享尾鳍频率，Hz |
| `--tail-left-amplitude-deg` | `10.0` | 逻辑向左最大幅度，度 |
| `--tail-right-amplitude-deg` | `10.0` | 逻辑向右最大幅度，度 |
| `--tail-first-direction` | `left` | 第一次离开中位的方向：`left` 或 `right` |
| `--start-delay-s` | `20.0` | 日志、传感器和舵机初始化前的本地倒计时 |
| `--initial-hold-s` | `1.0` | 初始姿态基线保持时间 |
| `--return-to-center-s` | `2.0` | 动作结束后的平滑回中时间 |
| `--command-hz` | `50.0` | 舵机目标命令更新频率 |

新脚本不再根据最大角速度或 `max_step_deg` 拒绝动作参数，也不会在相邻周期目标之间插入限速补点。`--command-hz` 只决定目标轨迹的离散更新频率。配置中的逐舵机机械 `min_angle` / `max_angle`、`--confirm MOVE` 和 `--dry-run` 检查仍然有效。高频率、大幅度或很短的过渡时间可能超过舵机实际响应能力，必须由实机测试人员自行从小幅、低频参数逐步验证。

#### 推荐测试顺序

1. Windows 本机只做语法和单元测试：

```powershell
python -m compileall -q control scripts tests
python -m unittest discover -s tests -v
```

2. 同步到树莓派：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\sync_to_pi.ps1 -PiIp <树莓派IP>
```

3. 在树莓派上使用 mock 传感器和 dry-run，不访问真实传感器或舵机：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/run_action_smooth_pectoral_tail_20260713.py --dry-run --mock-sensors --start-delay-s 0 --duration 10
```

dry-run 会在控制台打印 `t=0`、尖端过渡边界、`T/2` 和 `T` 的目标角度，便于人工核对。它仍会创建完整实验日志。

4. 在树莓派上使用真实传感器但禁止舵机 PWM：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/run_action_smooth_pectoral_tail_20260713.py --dry-run --start-delay-s 0 --duration 10
```

5. 完成 `test_i2c.py`、`test_uart.py`、`test_servo_center.py` 和 `test_servo_small_step.py` 后，再进行小幅真实动作：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/run_action_smooth_pectoral_tail_20260713.py --confirm MOVE --duration 10 --pectoral-frequency-hz 0.2 --root-amplitude-ratio 0.2 --tip-amplitude-deg 10 --tip-transition-s 0.8 --tail-frequency-hz 0.2 --tail-left-amplitude-deg 5 --tail-right-amplitude-deg 5 --start-delay-s 20 --initial-hold-s 1 --return-to-center-s 2
```

6. 左右尾鳍不等幅示例：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/run_action_smooth_pectoral_tail_20260713.py --confirm MOVE --duration 10 --pectoral-frequency-hz 0.2 --root-amplitude-ratio 0.2 --tip-amplitude-deg 10 --tip-transition-s 0.8 --tail-frequency-hz 0.2 --tail-left-amplitude-deg 8 --tail-right-amplitude-deg 5 --tail-first-direction left --start-delay-s 20
```

#### SSH 断开后继续运行

动作本身不依赖 Wi-Fi 或 SSH，但普通前台 SSH 进程可能随会话结束而收到挂断信号。需要把机器鱼放入水中并允许 SSH 断开时，应在树莓派本地使用 `nohup` 或 systemd 等 detached 方式启动。例如：

```bash
cd /home/fish/fishbot_pi_control
mkdir -p fish_logs
nohup python3 scripts/run_action_smooth_pectoral_tail_20260713.py \
  --confirm MOVE \
  --duration 10 \
  --pectoral-frequency-hz 0.2 \
  --root-amplitude-ratio 0.2 \
  --tip-amplitude-deg 10 \
  --tip-transition-s 0.8 \
  --tail-frequency-hz 0.2 \
  --tail-left-amplitude-deg 5 \
  --tail-right-amplitude-deg 5 \
  --start-delay-s 20 \
  --initial-hold-s 1 \
  --return-to-center-s 2 \
  > fish_logs/last_smooth_action.out 2>&1 < /dev/null &
```

倒计时开始后，即使 Wi-Fi 或 SSH 断开，树莓派也会根据本地单调时钟继续启动动作和记录数据。

#### 日志和 commands.jsonl

日志目录：

```text
logs/YYYYMMDD_HHMMSS_smooth_pectoral_tail_20260713/
```

目录继续包含：

```text
metadata.yaml
synchronized_sensors.jsonl
events.jsonl
commands.jsonl
raw_imu.jsonl
raw_uwb.jsonl
raw_depth.jsonl
raw_power.jsonl
raw_vision.jsonl
camera_index.jsonl
camera/
```

`commands.jsonl` 中的重要字段：

```text
t_ns                         与传感器共用的 monotonic 时间戳
action_state                 safe_center_initialized / enter_initial_pose /
                             initial_hold / periodic_action / return_to_center
action_elapsed_s             正式周期动作开始后的时间
pectoral_t_cycle_s           当前胸鳍周期内时间
pectoral_phase               统一胸鳍相位，范围 [0, 1)
pectoral_cycle_index         已进入的胸鳍周期编号
root_cosine_q                4、6 号共用余弦位置因子
tip_envelope_h               5、7 号共用尖端偏转包络
tail_offset_deg              尾鳍逻辑正弦偏移
commands_deg                 以 PCA9685 channel 为键的目标角度
commands_by_servo_id_deg     以舵机 ID 为键的同一组目标角度
```

`commands.jsonl` 记录的是发送给舵机的目标角度，不是带位置反馈的真实舵机角度。实机是否准确达到目标仍需通过机械检查、外部测量或后续增加位置反馈来验证。

## 左右滚转开环测试（2026-07-13）

脚本：

```text
scripts/run_action_roll_20260713.py
```

该脚本独立测试 `roll right` 和 `roll left`。4、6 号胸鳍根部在物理空间中反相运动；5、7 号胸鳍尖端相对各自中位向同一个数值方向、同一幅度偏转；1、2、3 号尾鳍从中位开始共享一个可左右不等幅的正弦相位。IMU 只记录滚转响应，不参与实时修正，这是开环动作而不是姿态闭环控制。

物理上下关系严格读取 `config/robot.yaml` 的明确标定字段，不从 `min_angle` / `max_angle` 猜测：

```text
servo_4_top_deg    = servo 4 direction.top_reference_angle
servo_4_bottom_deg = servo 4 direction.bottom_reference_angle
servo_6_top_deg    = servo 6 direction.top_reference_angle
servo_6_bottom_deg = servo 6 direction.bottom_reference_angle
```

当前配置对应 4 号 `top=68° / bottom=174°`、6 号 `top=196° / bottom=90°`。所有中心、物理端点和计算目标都直接按各舵机在 `config/robot.yaml` 中的 `min_angle` / `max_angle` 做含边界验证；配置限位已经包含机械安全余量，脚本不再提供或叠加第二层 margin。

本脚本不叠加 `safety.servo.max_test_amplitude_deg` 的全局试验行程限制。4、5、6、7 的目标以及 1、2、3 经过各自 `sign` 和 `amplitude_scale` 映射后的目标，只按对应舵机在 `config/robot.yaml` 中配置的 `min_angle..max_angle` 做含边界预检；越界仍会在倒计时和硬件初始化之前报错，不会缩幅或 clamp。

因此 `--root-amplitude-ratio` 的有效范围是完整的 `0..1`。当前 4、6 的 top↔bottom 行程均为 `106°`，`ratio=1` 会从本次起始物理极限移动到另一物理极限，目标恰好位于配置边界，属于合法请求。

胸鳍共用唯一周期时钟：

```text
T = 1 / pectoral_frequency_hz
t_cycle = action_elapsed_s % T
phase = t_cycle / T
q = [1 - cos(2π t_cycle / T)] / 2
```

右滚：

```text
theta_4_start  = servo_4_bottom_deg
theta_4_target = servo_4_bottom_deg
                 + root_amplitude_ratio × (servo_4_top_deg - servo_4_bottom_deg)
theta_6_start  = servo_6_top_deg
theta_6_target = servo_6_top_deg
                 + root_amplitude_ratio × (servo_6_bottom_deg - servo_6_top_deg)
theta_4 = theta_4_start + q × (theta_4_target - theta_4_start)
theta_6 = theta_6_start + q × (theta_6_target - theta_6_start)
```

左滚交换物理起点：4 号从 top 沿 `top→bottom` 全行程移动相同比例，6 号从 bottom 沿 `bottom→top` 全行程移动相同比例。`root_amplitude_ratio=0` 表示保持在本次起始极限，`1` 表示到达另一物理极限；计算基准始终是明确标定的 top/bottom，不是中位角。当前 4、6 的完整物理行程均为 `106°`，所以默认比例 `0.1` 对应两侧各 `10.6°`。

尖端包络继续使用五次 smoothstep，在前半周期进入偏转、保持、返回中位，后半周期保持中位：

```text
theta_5 = center_5 + tip_sign × tip_amplitude_deg × h
theta_7 = center_7 + tip_sign × tip_amplitude_deg × h
```

`--tip-direction positive` 对应 `tip_sign=+1`，`negative` 对应 `-1`。因此始终满足：

```text
theta_5 - center_5 = theta_7 - center_7
```

尾鳍共享：

```text
s >= 0: tail_offset = tail_left_amplitude_deg × s
s < 0:  tail_offset = tail_right_amplitude_deg × s
target_i = center_i + sign_i × amplitude_scale_i × tail_offset
```

其中 `s = ±sin(2π × tail_frequency_hz × action_elapsed_s)`，正负号由 `tail_first_direction` 决定。该指定分段公式在中位处角度连续且从中位启动；如果左右幅度不等，其左右导数在数学上分别与两个幅度成比例，因此不能同时声称中位交叉处严格速度连续。

参数及默认值：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--confirm` | 空 | 非 dry-run 必须传 `MOVE` |
| `--dry-run` | false | 不初始化 PCA9685、不发送真实 PWM |
| `--mock-sensors` | false | 只模拟传感器，不自动禁止真实舵机 |
| `--keep-pwm` | false | 回中并关闭日志后保持 PWM |
| `--roll-direction` | `right` | `left` 或 `right` |
| `--tip-direction` | `negative` | 5、7 共同的数值偏转方向 |
| `--duration` | `10.0` | 正式动作时间，不含缓冲和回中 |
| `--pectoral-frequency-hz` | `0.2` | 4、5、6、7 共用频率 |
| `--root-amplitude-ratio` | `0.1` | 4、6 各自 top↔bottom 完整物理行程比例，范围 `0..1` |
| `--tip-amplitude-deg` | `10.0` | 5、7 共用偏转幅度 |
| `--tip-transition-s` | `0.8` | 尖端 smoothstep 单程时间 |
| `--tail-frequency-hz` | `0.2` | 1、2、3 共用频率 |
| `--tail-left-amplitude-deg` | `5.0` | 逻辑向左幅度 |
| `--tail-right-amplitude-deg` | `5.0` | 逻辑向右幅度 |
| `--tail-first-direction` | `left` | 第一次离开中位的方向 |
| `--start-delay-s` | `20.0` | 日志/传感器/舵机初始化前本地倒计时 |
| `--initial-hold-s` | `1.0` | 初始姿态基线保持时间 |
| `--tail-ramp-down-s` | `1.0` | 动作结束后尾鳍平滑减幅时间 |
| `--return-to-center-s` | `2.0` | 进入初始姿态及 4～7 回中的平滑时间 |
| `--command-hz` | `50.0` | 舵机目标更新频率 |

无硬件验证：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/run_action_roll_20260713.py --dry-run --mock-sensors --start-delay-s 0 --duration 5 --pectoral-frequency-hz 0.2 --root-amplitude-ratio 0.05 --tip-amplitude-deg 5 --tip-transition-s 0.8 --tail-left-amplitude-deg 4 --tail-right-amplitude-deg 2
```

dry-run 会打印胸鳍关键边界和尾鳍左右极值检查点，并在本次日志目录额外保存一个同时覆盖至少一个完整胸鳍周期和一个完整尾鳍周期的 `dry_run_trajectory.jsonl`。正式日志格式和现有采样架构不变。`--duration` 按动作执行时间处理；若它不是完整胸鳍周期，脚本会从当时目标进入尾鳍平滑减幅和回中，不会突然停止 PWM。

第一次右滚小幅实机命令应在树莓派端以 detached 方式启动：

```bash
cd /home/fish/fishbot_pi_control
mkdir -p fish_logs
nohup python3 scripts/run_action_roll_20260713.py \
  --confirm MOVE \
  --roll-direction right \
  --tip-direction negative \
  --duration 5 \
  --pectoral-frequency-hz 0.2 \
  --root-amplitude-ratio 0.05 \
  --tip-amplitude-deg 5 \
  --tip-transition-s 0.8 \
  --tail-frequency-hz 0.2 \
  --tail-left-amplitude-deg 4 \
  --tail-right-amplitude-deg 2 \
  --tail-first-direction left \
  --start-delay-s 20 \
  > fish_logs/last_roll_right.out 2>&1 < /dev/null &
```

第一次左滚小幅实机命令：

```bash
cd /home/fish/fishbot_pi_control
mkdir -p fish_logs
nohup python3 scripts/run_action_roll_20260713.py \
  --confirm MOVE \
  --roll-direction left \
  --tip-direction negative \
  --duration 5 \
  --pectoral-frequency-hz 0.2 \
  --root-amplitude-ratio 0.05 \
  --tip-amplitude-deg 5 \
  --tip-transition-s 0.8 \
  --tail-frequency-hz 0.2 \
  --tail-left-amplitude-deg 4 \
  --tail-right-amplitude-deg 2 \
  --tail-first-direction left \
  --start-delay-s 20 \
  > fish_logs/last_roll_left.out 2>&1 < /dev/null &
```

上面左右滚转示例故意保持相同的 `--tip-direction negative` 和 `--tail-first-direction left`，用来单独比较 `roll_direction`。尖端正方向对照实验只需改为 `--tip-direction positive`，不应交换 5、7 号的方向，也不需要改变 `--roll-direction`。

上述 `nohup` 命令已经将 stdin、stdout 和 stderr 与 SSH 会话分离；动作倒计时和执行只依赖树莓派本地单调时钟，不等待 Wi-Fi、ping 或电脑确认。正式入水动作不要以前台 `run_on_pi.ps1` 进程代替 detached 启动。

## 先俯冲再滚转组合动作（2026-07-13）

脚本：

```text
scripts/run_mission_dive_roll_20260713.py
```

该脚本在一次实验目录、一次传感器会话和同一 `time.monotonic_ns()` 时间基准中完成：

```text
start delay
→ 全部舵机安全到中位
→ smoothstep 平滑进入俯冲姿态
→ 定时开环俯冲
→ smoothstep 平滑衔接滚转初始物理极限
→ roll 本地时钟重新从 0 开始
→ 执行 run_action_roll_20260713 的滚转轨迹
→ 尾鳍平滑减幅
→ 全部舵机回中
→ 关闭传感器和日志
→ 按 --keep-pwm 决定 PWM
```

俯冲胸鳍语义沿用 `run_mission_dive_cruise_surface.py`：

```text
servo 4 = center_4
servo 5 = center_5 + dive_pectoral_tilt
servo 6 = center_6
servo 7 = center_7 - dive_pectoral_tilt
```

俯冲段不使用深度闭环，Depth 和 IMU 只记录。尾鳍改用当前 roll 脚本的中位起步不对称正弦波；俯冲和滚转分别使用独立相位时钟，因此两段的 `t=0` 都从尾鳍中位开始。俯冲末端到滚转初始姿态的转换会延续俯冲尾相位并用五次 smoothstep 将其幅度减到 0，同时把 4～7 平滑移动到滚转 phase 0 姿态；转换末端与滚转第一条命令完全相同。

新增参数：

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--dive-duration` | `10.0` | 俯冲动作时间，不含缓冲和转换，必须 `> 0` |
| `--dive-pectoral-tilt` | `20.0` | 5 号增加、7 号减少的镜像俯冲攻角，必须 `>= 0` |
| `--transition-s` | `1.0` | 中位→俯冲以及俯冲→滚转的 smoothstep 时间，必须 `> 0` |

其余滚转、尾鳍、start delay、dry-run、mock、确认和 PWM 参数与 `run_action_roll_20260713.py` 相同。`--duration` 只表示滚转段时间；总动作时间还包括两次 `--transition-s`、`--dive-duration`、`--initial-hold-s`、尾鳍减幅和最终回中。

所有目标只按 `config/robot.yaml` 中每个舵机自己的 `min_angle..max_angle` 做含边界预检，不叠加 `safety.servo.max_test_amplitude_deg`、额外 margin、静默 clamp 或自动缩幅。因此 `--root-amplitude-ratio 1` 可合法到达已配置的物理 top/bottom 端点；真正超过某个舵机配置范围时仍会在 start delay、日志、传感器及 PCA9685 初始化之前报错。

完整组合 dry-run：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/run_mission_dive_roll_20260713.py --dry-run --mock-sensors --start-delay-s 0 --dive-duration 5 --dive-pectoral-tilt 8 --transition-s 1 --roll-direction right --duration 5 --pectoral-frequency-hz 0.2 --root-amplitude-ratio 0.05 --tip-amplitude-deg 5 --tip-transition-s 0.8 --tail-frequency-hz 0.2 --tail-left-amplitude-deg 4 --tail-right-amplitude-deg 2
```

第一次右滚组合动作建议命令：

```bash
cd /home/fish/fishbot_pi_control
mkdir -p fish_logs
nohup python3 scripts/run_mission_dive_roll_20260713.py \
  --confirm MOVE \
  --dive-duration 5 \
  --dive-pectoral-tilt 8 \
  --transition-s 1 \
  --roll-direction right \
  --tip-direction negative \
  --duration 5 \
  --pectoral-frequency-hz 0.2 \
  --root-amplitude-ratio 0.05 \
  --tip-amplitude-deg 5 \
  --tip-transition-s 0.8 \
  --tail-frequency-hz 0.2 \
  --tail-left-amplitude-deg 4 \
  --tail-right-amplitude-deg 2 \
  --tail-first-direction left \
  --start-delay-s 20 \
  > fish_logs/last_dive_roll_right.out 2>&1 < /dev/null &
```

向左滚转只需将上例改为：

```text
--roll-direction left
```

不要同时改变 `--tip-direction`，这样才能单独比较左右滚转根部相位。真实俯冲方向、滚转方向、水动力效果和安全幅度仍须在树莓派实机及水池中确认。

## 离散动作序列控制（2026-07-16）

> 本节为旧的相对位移/指定时长动作接口，保留用于兼容已有实验。新任务请使用下节的
> 2026-07-17 绝对角度接口；两个入口和动作领域模块彼此隔离。

入口：

```text
scripts/run_discrete_action_sequence_20260716.py
```

人工任务示例：

```yaml
name: discrete_action_example

tail_actions:
  - {t: 1.0, A: 10.0}
  - {t: 2.0, A: -20.0}

left_fin_actions:
  - {t: 2.0, b: 1, r: 0.3333333333}

right_fin_actions:
  - {t: 2.0, b: -1, r: 0.3333333333}
```

`tail_actions`、`left_fin_actions`、`right_fin_actions` 三个键必须存在，单个列表允许为空，但整个 mission 至少需要一个动作。三个序列从同一个 `mission_start_t_ns` 并行启动，每个序列内部严格串行；当前动作规定时间经过且终点成功写入后，下一动作才从该实际完成时刻启动。

动作只使用以下参数：

- `action1(t, A)`：1～3 号从各自当前累计模型位置移动逻辑相对角 `A`。`A>0` 为物理向左，数值方向由每个舵机的左右参考标定确定；动作结束不自动回中。
- `action2(t, b, r)`：4、5 号左鳍从中位开始并回到中位。`b=1` 时两者先向各自的 `bottom_reference` 偏转，即4号角度增大、5号角度减小；`b=-1` 时两者先向各自的 `top_reference` 偏转。
- `action3(t, b, r)`：6、7 号右鳍使用相同数学结构。`b=1` 时两者先向各自的 `bottom_reference` 偏转，即6号角度减小、7号角度增大；`b=-1` 时两者先向各自的 `top_reference` 偏转。配置标定不做反转。

全部运动段使用：

```text
S(u) = 10u^3 - 15u^4 + 6u^5
```

侧鳍根部在 `0、t/4、3t/4、t` 之间分三段运动；末端在 `t/4～3t/8` 向相同的物理上下方向进入偏转、保持到 `3t/4`、在 `3t/4～7t/8` 回中。`r=0` 全程保持中位，`r=1` 到达 `robot.yaml` 明确标定的对应参考角。

启动前会验证完整累计轨迹，唯一角度范围是各舵机自己的 `min_angle..max_angle`。程序不使用 `max_test_amplitude_deg`，不会 clamp、自动缩幅或添加第二层软限位；运行时每次写入前仍再次调用控制器限位。

### 无反馈状态和时间对齐

舵机没有位置反馈，日志严格区分：

```text
reference_angles_deg   连续动作函数在指定 monotonic 时间戳的理论参考角
commanded_angles_deg   不晚于查询时间的最近一次成功 PCA9685 命令
estimated_angles_deg   假设舵机完全跟随时的模型估计角
```

当前 `estimated_angles_deg = reference_angles_deg`，估计模式为 `assumed_perfect_tracking`。它不是实测角度，不能证明舵机已经到达目标。

同步记录的 `servo_state_by_sensor` 按 IMU、Depth、Power、UWB、Vision 各自的 `sample_t_ns` 查询状态。例如深度样本早于一次舵机写入而 IMU 样本晚于该写入时，两者会得到不同的 `commanded_angles_deg`，不会使用日志写入时刻覆盖历史样本。

### 线程和日志

只有 `ServoExecutor` 线程创建并访问 ServoKit/PCA9685。它使用：

```text
scheduled_t_ns = executor_start_t_ns + tick_index * period_ns
```

传感器采集、同步采样、raw 日志搬运、JSONL 写盘和相机图片保存均在其他长期线程中完成。任意 PWM 写入失败会停止动作推进并进入安全回中；commands/events 队列丢记录或可捕获写盘错误按配置继续运动，并在实验摘要中报告。

`commands.jsonl` 保存计划时间、实际写入起止时间、lateness、写入耗时、七路参考/命令/估计角、三路动作索引和局部进度。`events.jsonl` 保存 mission/action 生命周期、PWM 失败、安全回中和 logger 停止事件。`metadata.yaml` 保存完整人工序列、`robot.yaml` 快照、执行器配置和无反馈语义。

### 运行方法

Windows 本机或树莓派上的完全无硬件验证：

```powershell
python scripts/run_discrete_action_sequence_20260716.py --mission missions/discrete_action_example.yaml --dry-run --mock-sensors --start-delay-s 0
```

树莓派上使用真实传感器但禁止真实舵机 PWM：

```bash
python3 scripts/run_discrete_action_sequence_20260716.py \
  --mission missions/discrete_action_example.yaml \
  --dry-run \
  --start-delay-s 0
```

完成 I2C、UART、中位和小步测试后，真实任务应以 detached 方式启动：

```bash
cd /home/fish/fishbot_pi_control
mkdir -p fish_logs
nohup python3 scripts/run_discrete_action_sequence_20260716.py \
  --mission missions/discrete_action_example.yaml \
  --confirm MOVE \
  --start-delay-s 20 \
  > fish_logs/last_discrete_action.out 2>&1 < /dev/null &
```

倒计时期间不初始化 PCA9685，因此软件只保证不发送新命令，不能主动纠正物理位置。操作员必须在启动前确认机器鱼已安全居中。倒计时结束后，Executor 会重新发送中位目标、等待稳定、执行任务，任务结束后再独立安全回中，并按 `--keep-pwm` 和安全配置决定是否释放 PWM。

## 绝对角度和名义速度离散动作（2026-07-17）

入口为 `scripts/run_discrete_absolute_actions_20260717.py`，人工 mission 示例为
`missions/discrete_absolute_action_example.yaml`。尾鳍动作字段为
`theta1/theta2/theta3/v1/v2/v3`，左右鳍动作字段均为 `theta/v/b`；所有 theta 都是
PCA9685 驱动使用的绝对角度，不是相对中心或上一个动作的位移。

每个舵机的时间严格使用 `abs(target-start)/v`，其中 v 是名义平均速度。参考轨迹使用
`S(u)=10u³-15u⁴+6u⁵`，因此瞬时峰值速度可以高于 v，但不会将动作时间乘以 1.875。
尾鳍三路保留各自速度和持续时间，动作总时间取三者最大值；提前到达的舵机保持终点。

4→5 和 6→7 的配合峰值统一使用：

```text
tip_peak = tip_center + (root_target - root_start) / 106 × 90
```

106/90 从 `robot.yaml.discrete_action_coupling` 读取，只是耦合映射标尺，不是机械限位。
`b=0` 时尖端全程保持配置中位；`b=1` 时尖端在动作前四分之一平滑进入峰值、中间
二分之一保持、后四分之一平滑回中。左右侧严格使用同号公式，不因镜像反号。

启动前会从七路配置中位模拟完整三条序列，验证速度、所有绝对目标和所有耦合峰值。
唯一角度限位来自每路 `min_angle/max_angle`；不 clamp、不自动缩幅，也不使用
`max_test_amplitude_deg`。三路在同一 `mission_start_t_ns` 启动、路内串行；只有规定时间
结束且本组终点成功写入后才能推进。只有一个 `ServoExecutor` 线程访问 PCA9685。

完全无硬件验证：

```powershell
python scripts/run_discrete_absolute_actions_20260717.py --mission missions/discrete_absolute_action_example.yaml --dry-run --mock-sensors --start-delay-s 0
```

通过本地上传工作流在树莓派运行 dry-run：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp 192.168.1.111 -NoVenv scripts/run_discrete_absolute_actions_20260717.py --mission missions/discrete_absolute_action_example.yaml --dry-run --mock-sensors --start-delay-s 0
```

完成 I2C、UART、居中和小步安全检查后，真实运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp 192.168.1.111 -NoVenv scripts/run_discrete_absolute_actions_20260717.py --mission missions/discrete_absolute_action_example.yaml --confirm MOVE --start-delay-s 20
```

`ServoStateTracker.get_servo_pose_at(t_ns)` 可查询任务期间任意历史单调时间戳。日志区分
连续参考角、指定时刻之前最近成功写入的 commanded 角，以及假设完美跟随的 estimated
角；当前 `feedback_available=false`、`estimation_mode=assumed_perfect_tracking`，这些值均
不是舵机实测角度。同步传感器记录按各传感器自身 `sample_t_ns` 查询相应姿态。

## 人工定时绝对离散动作（2026-07-25）

入口：

```text
scripts/run_discrete_absolute_actions_20260725.py
```

示例任务：

```text
missions/discrete_absolute_actions_20260725_example.yaml
```

本入口和 20260716、20260717 两套旧动作领域模块相互隔离，不再接受速度、频率、相位、
旧行程比例或多个尾鳍绝对角字段。新的严格接口只有：

```text
action1(theta, t)
action2(theta, t, b1, b2)
action3(theta, t, b1, b2)
```

人工 YAML 顶层必须严格包含 `name`、`tail_actions`、`left_fin_actions` 和
`right_fin_actions`。三个列表可有不同长度，每个动作也可有不同 `t`；三组使用同一
`mission_start_t_ns` 并行启动，每组内部串行推进，不设置组间动作屏障。

所有发生角度变化的轨迹统一使用：

```text
S(u) = 10u³ - 15u⁴ + 6u⁵
u = clamp(elapsed / t, 0, 1)
q = q0 + (q1 - q0) × S(u)
```

实现会在 `elapsed<=0` 和 `elapsed>=t` 显式返回精确起点、终点。相邻 theta 相等时，
动作仍完整保持 `t` 秒，但直接返回固定角度，不执行无意义插值。

### action1(theta, t)

尾鳍 `theta` 是相对固定参考中位的统一偏移，不是相对上一条命令的增量：

```text
servo1 = 85 + theta
servo2 = 95 + theta
servo3 = 95 + theta
-30 <= theta <= 30
t > 0
```

运行时 `previous_action1_theta` 初值为 `0`。三个舵机从
`center + previous_action1_theta` 在完整 `t` 内同步平滑到
`center + current_action1.theta`。

### action2/action3(theta, t, b1, b2)

`action2` 控制 4、5 号，`action3` 控制 6、7 号。根部 `theta` 是绝对角度，
`b1` 只能为 `0/1`，`b2` 只能为 `-1/1`，`t>0`。左右根部 previous theta
初值分别为当前配置中的 `111°` 和 `143°`。

根部在完整 `t` 内从 previous theta 平滑到 current theta。若二者相等，根部和尖端
均保持完整 `t`，`b1/b2` 不触发额外动作。若二者不同：

```text
delta_abs = abs(current_theta - previous_theta)
tip_peak = tip_center + b2 × delta_abs / 106 × 90
```

`b1=0` 时尖端全程精确保持中位。`b1=1` 时尖端在 `0..t/4` 从中位平滑到
`tip_peak`，在 `t/4..3t/4` 精确保持峰值，再在 `3t/4..t` 平滑回中位。
当前左右尖端中位均为 `90°`；右侧不会因镜像结构擅自反转 `b2`。根部完整
移动 `106°` 时，`b2=1` 对应尖端峰值 `180°`，`b2=-1` 对应 `0°`。

20260725 独立配置段只保存与现有字段不等价的 `theta` 范围和 `106/90` 映射标尺。
七路 `85/95/95/111/90/143/90` 参考中位直接复用
`servo.channels.center_angle`，避免两套同义配置漂移。动作启动前会模拟完整三条序列，
并将根部物理 top/bottom 闭区间与每路 `min_angle/max_angle` 取交集；尾鳍目标、根部
目标和尖端峰值任何一个越界都会在硬件初始化前拒绝，不会 clamp、缩幅、改方向或延长
动作时间。

### previous theta、完成判据和历史姿态

三个 previous theta 只有在以下两个条件同时满足时才提交：

```text
动作规定 t 已经过
终点统一命令已成功写入 PCA9685
```

写入失败会停止全部动作、保留尚未提交的 previous theta 并进入安全清理。下一动作的
`actual_start_t_ns` 等于上一动作的 `completion_t_ns`。公开接口：

```python
ServoStateTracker.get_servo_pose_at(t_ns)
ServoStateTracker.get_servo_pose_at_elapsed_s(elapsed_s)
```

可查询任务前、动作中、保持、历史动作和序列结束后的七路参考姿态，以及不晚于查询时刻
的最近成功命令。当前：

```text
estimated_angles_deg = reference_angles_deg
feedback_available = false
estimation_mode = assumed_perfect_tracking
```

这些字段不是实测位置。同步传感器记录继续使用各传感器自己的 `sample_t_ns` 查询姿态。
任务结束后的安全回中属于显式 `control_phase=safe_recenter` 参考段；该阶段及其结束后，
参考/估计角会反映回中命令，而 previous theta 和动作历史仍保留最后一次成功提交结果。

### 单执行器、日志和运行

只有一个 `ServoExecutor` 长期线程创建和访问 PCA9685。它在每个绝对截止时间合并三组
参考角为 1～7 号统一命令。保持阶段命令完全不变时可跳过重复 I2C 写入，但动作起点和
终点仍强制成功写入；`commands.jsonl` 会记录 `pwm_write_performed` 和
`write_skip_reason=unchanged_command`。

控制器初始化后，执行器以各通道配置中位作为无反馈模型起点，用五次曲线在
`servo.discrete_executor.initial_move_s` 内进入七路新动作初始化姿态，再等待
`center_settle_s`。默认两组中位相同，因此参考轨迹为精确常值保持；这仍只是命令与
模型初始化，不代表物理舵机已由位置反馈确认到位。

每次实验目录会保存原始人工任务副本 `manual_action_sequence.yaml`，并生成
`metadata.yaml`、`commands.jsonl`、`events.jsonl`、`synchronized_sensors.jsonl`、
raw 传感器日志和相机文件。

本入口的 `--start-delay-s` 默认值为 `5` 秒；需要立即开始 dry-run 时可显式传入
`--start-delay-s 0`。倒计时结束前不会创建日志、启动传感器或初始化 PCA9685。

本机或树莓派完全无硬件验证：

```powershell
python scripts/run_discrete_absolute_actions_20260725.py --mission missions/discrete_absolute_actions_20260725_example.yaml --dry-run --mock-sensors --start-delay-s 0
```

完成 I2C、UART、居中和小步安全检查后，在树莓派真实运行：

```bash
python3 scripts/run_discrete_absolute_actions_20260725.py \
  --mission missions/discrete_absolute_actions_20260725_example.yaml \
  --confirm MOVE
```

通过 Windows 本机先同步（同步范围已包含 `missions/`），再远程运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\codex_pi_workflow\sync_to_pi.ps1 `
  -PiIp 192.168.1.111

powershell -NoProfile -ExecutionPolicy Bypass `
  -File .\codex_pi_workflow\run_on_pi.ps1 `
  -PiIp 192.168.1.111 `
  -NoVenv `
  scripts/run_discrete_absolute_actions_20260725.py `
  --mission missions/discrete_absolute_actions_20260725_example.yaml `
  --confirm MOVE
```

`--mock-sensors` 只模拟传感器，不能替代 `--dry-run`。真实动作仍必须显式提供
`--confirm MOVE`。默认安全回中后释放 PWM；`--keep-pwm` 会继续保持回中力矩。

## 摄像头独立测试

USB 双目摄像头测试脚本：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/test_usb_camera.py --frames 10
```

如果需要保存测试图片，可使用 `test_usb_camera.py` 的保存参数。普通 record 和动作脚本中的图片会保存到当前 run 的 `logs/.../camera/`，不是旧的 `captures/`。

## 依赖

`requirements.txt` 包含：

```text
PyYAML
smbus2
pyserial
adafruit-blinka
adafruit-circuitpython-servokit
adafruit-circuitpython-ina219
gpiozero
lgpio
```

OpenCV 不强制写入 `requirements.txt`。摄像头 worker 会 optional import `cv2`。在树莓派上可用系统包安装：

```bash
sudo apt update
sudo apt install -y python3-opencv
```

如果使用虚拟环境但硬件依赖缺失，需要安装对应依赖，或在测试时使用 `run_on_pi.ps1 -NoVenv` 走系统 Python。

## 已验证结果示例

在树莓派实机上曾完成：

- mock 管线测试成功。
- 真实传感器状态检查成功。
- 10 秒和 30 秒 record 成功。
- 30 秒记录中：
  - `synchronized_sensors.jsonl` 存在。
  - `camera_index.jsonl` 存在。
  - `camera/` 下 JPEG 正常保存。
  - 历史配置 `save_fps=5` 时，图片保存估算 FPS 约 4.7 到 4.8；当前配置已改为 `save_fps=20`，需要重新实机验证实际保存速度。
  - raw sensor logs 均有数据。
  - `events.jsonl` 显示 logger 正常开始和结束。
  - `sync_dropped_count=0`。

10 秒记录 raw 日志示例：

```text
raw_imu.jsonl:    485 lines
raw_depth.jsonl:  185 lines
raw_power.jsonl:   20 lines
raw_uwb.jsonl:     55 lines
raw_vision.jsonl: 117 lines
```

## 故障排查

### raw_*.jsonl 是 0B

旧版本只创建 raw 占位文件。当前版本 record 模式会写入实际 raw 样本。请重新同步并运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\sync_to_pi.ps1 -PiIp <树莓派IP>
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/record_sensors.py --duration 10
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp <树莓派IP> -NoVenv scripts/check_latest_log.py
```

### 摄像头没有数据

检查：

- 摄像头是否插入并出现在 `/dev/video*`。
- 是否安装 `python3-opencv`。
- 是否使用了 `-NoVenv`。
- `config/robot.yaml` 中 `sensors.vision.enabled` 是否为 `true`。

### UWB no solution

`LO=[no solution]` 或 timeout 是正常情况，尤其在水下。UWB invalid 不会停止 recorder。

### 深度为负数

通常是 surface pressure 或 zero 设置和当前环境不一致。日志系统仍正常，后续实验前应做深度零点校准。

### 树莓派 venv 缺少 lgpio

如果 venv 中运行 Blinka/ServoKit 报：

```text
ModuleNotFoundError: No module named 'lgpio'
```

可以安装 `lgpio`，或在当前测试中使用 `-NoVenv` 走系统 Python。

## 安全原则

当前 `config/robot.yaml` 中的安全阈值：

```yaml
safety:
  min_voltage_v: 10.5
  max_current_a: 10.0
  max_depth_m: 5.0
  stop_on_imu_timeout: false
  stop_on_uwb_timeout: false
  stop_on_vision_timeout: false
  stop_on_network_lost: false
  servo:
    default_center_angle: 90
    default_min_angle: 70
    default_max_angle: 110
    small_step_delta_deg: 5
    max_test_amplitude_deg: 30
    max_step_deg: 1
    step_delay_seconds: 0.08
    release_pwm_after_tests: true
```

- 默认 `main.py`、`record_sensors.py`、`run_sensor_observe.py` 不驱动舵机。
- 舵机动作脚本必须使用 `--confirm MOVE`。
- 水下实验前确认日志是本地写入，不依赖电脑实时接收。
- SSH/Wi-Fi 断开不能作为 emergency stop。
- UWB timeout 不能作为 emergency stop。
- 摄像头写盘慢时允许丢图片，不能阻塞 IMU/depth/power/UWB 采集。
- 实验前确认供电、电流、线缆、机械限位和鱼体固定。

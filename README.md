# fishbot_pi_control

树莓派机器鱼控制与数据采集项目。当前阶段的重点是：在树莓派 5 Debian Trixie Lite 上稳定采集多传感器数据，保存本地日志，并为后续预设动作实验、运动学建模和水动力学建模提供统一时间戳的数据。

当前阶段不是强化学习，不是完整闭环控制，也不是在线遥控机器人。默认入口不会驱动舵机，不会运行 gait，不会调用 RL policy，也不会默认输出 PCA9685 PWM。

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
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp 192.168.1.111 <script> <args>
```

同步代码到树莓派：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\sync_to_pi.ps1 -PiIp 192.168.1.111
```

如果树莓派系统 Python 已经安装了硬件库和 OpenCV，而虚拟环境缺少依赖，可以使用 `-NoVenv`：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp 192.168.1.111 -NoVenv scripts/test_sensor_pipeline.py
```

同步脚本主要同步以下路径：

```text
main.py
requirements.txt
config/
drivers/
control/
scripts/
```

因此新增运行时代码优先放在 `control/runtime/`、`scripts/`、`config/`、`drivers/`。

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
    calibrate_servo.py            # 单舵机标定
    test_servo_center.py
    test_i2c.py
    test_uart.py
    test_usb_camera.py
  codex_pi_workflow/
    sync_to_pi.ps1
    run_on_pi.ps1
    setup_pi_venv.ps1
  main.py                         # sensor status / observe / record CLI
  requirements.txt
```

## 硬件接口配置

所有硬件配置集中在 `config/robot.yaml`。

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
图片保存频率：save_fps: 5
```

配置：

```yaml
sensors:
  vision:
    enabled: true
    camera_index: 0
    rate_hz: 15
    buffer_size: 100
    timeout_ms: 300
    save_frames: true
    save_fps: 5
    image_format: jpg
    jpeg_quality: 85
    save_combined_frame: true
    split_stereo: false
    max_image_queue_size: 100
```

摄像头规则：

- 摄像头仍是 optional sensor。
- 如果 `cv2` 不存在，vision enabled 时会输出 `ok=false, error="opencv_not_available"`，系统继续运行。
- 摄像头采集频率 `rate_hz=15` 不等于图片保存频率。
- JPEG 保存频率由 `save_fps=5` 限制，即每秒最多保存 5 张。
- 图片保存为 combined frame，例如双目左右拼接的 `1280x480`、`2560x720`、`2560x960` 当前先整体保存。
- 当前不做 left/right 分割，后续再扩展。
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
      "width": 1280,
      "height": 480,
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
  "width": 1280,
  "height": 480,
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
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\sync_to_pi.ps1 -PiIp 192.168.1.111
```

### 2. mock 测试传感器管线

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp 192.168.1.111 -NoVenv scripts/test_sensor_pipeline.py --mock
```

### 3. 真实传感器状态检查

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp 192.168.1.111 -NoVenv scripts/test_sensor_pipeline.py
```

该脚本会访问真实 IMU、UWB、depth、power、camera，但不驱动舵机。

### 4. 水面 observe

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp 192.168.1.111 -NoVenv scripts/run_sensor_observe.py
```

### 5. 记录 30 秒传感器数据

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp 192.168.1.111 -NoVenv scripts/record_sensors.py --duration 30
```

### 6. 检查最新日志

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp 192.168.1.111 -NoVenv scripts/check_latest_log.py
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
| 4 | ZQ1 | 左胸鳍根部 | 3 | 114 | 74 | 164 |
| 5 | ZQ2 | 左胸鳍尖端 | 4 | 90 | 0 | 180 |
| 6 | YQ1 | 右胸鳍根部 | 5 | 117 | 82 | 192 |
| 7 | YQ2 | 右胸鳍尖端 | 6 | 90 | 0 | 180 |

方向定义：

- 从尾部往头看，1 号从左到右角度变小：`110 -> 80`。
- 从尾部往头看，2/3 号从左到右角度变小：`120 -> 60`。
- 4 号从上往下角度变大：`74 -> 164`。
- 6 号从上往下角度变小：`192 -> 82`。
- 5 和 7 号是左右鳍尖镜面对称：
  - 5 号角度变大时左侧鳍向前转。
  - 7 号角度变小时右侧鳍向前转。

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

- 4、6 号舵机不发命令，保持不动。
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
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp 192.168.1.111 -NoVenv scripts/run_action_pectoral_tail_1.py --confirm MOVE --duration 5 --tail-frequency 0.3 --tail-amplitude 10 --pectoral-tilt 30
```

默认幅度测试：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp 192.168.1.111 -NoVenv scripts/run_action_pectoral_tail_1.py --confirm MOVE --duration 10 --tail-frequency 0.5 --tail-amplitude 20 --pectoral-tilt 30
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

动作结束后，脚本会把 1、2、3、5、7 号舵机回到中心位置，并根据配置释放 PWM。4、6 号舵机不会被该脚本命令。

## 摄像头独立测试

USB 双目摄像头测试脚本：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp 192.168.1.111 -NoVenv scripts/test_usb_camera.py --frames 10
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

在树莓派 `192.168.1.111` 上曾完成：

- mock 管线测试成功。
- 真实传感器状态检查成功。
- 10 秒和 30 秒 record 成功。
- 30 秒记录中：
  - `synchronized_sensors.jsonl` 存在。
  - `camera_index.jsonl` 存在。
  - `camera/` 下 JPEG 正常保存。
  - 图片保存估算 FPS 约 4.7 到 4.8，接近 `save_fps=5`。
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
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\sync_to_pi.ps1 -PiIp 192.168.1.111
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp 192.168.1.111 -NoVenv scripts/record_sensors.py --duration 10
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 -PiIp 192.168.1.111 -NoVenv scripts/check_latest_log.py
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

- 默认 `main.py`、`record_sensors.py`、`run_sensor_observe.py` 不驱动舵机。
- 舵机动作脚本必须使用 `--confirm MOVE`。
- 水下实验前确认日志是本地写入，不依赖电脑实时接收。
- SSH/Wi-Fi 断开不能作为 emergency stop。
- UWB timeout 不能作为 emergency stop。
- 摄像头写盘慢时允许丢图片，不能阻塞 IMU/depth/power/UWB 采集。
- 实验前确认供电、电流、线缆、机械限位和鱼体固定。

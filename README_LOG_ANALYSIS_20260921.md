# 本地运行日志分析

脚本：`scripts/analyze_control_logs_20260921.py`。仅离线读取日志，不访问树莓派、不控制硬件。
支持固定下潜策略、PPO 及使用相同 `raw_*.jsonl` 格式的采集任务。

## 安装与运行

在本机项目根目录执行：

```bash
python -m pip install -r requirements-analysis.txt

# 指定已拷贝到本机的完整运行目录
python scripts/analyze_control_logs_20260921.py logs/fixed_dive_debug_20260920/20260920_155516_fixed_dive --show

# 自动查找 logs 下最新的非模拟运行
python scripts/analyze_control_logs_20260921.py --latest --show

# 只分析动作开始后的0～60秒
python scripts/analyze_control_logs_20260921.py --latest --start-s 0 --end-s 60 --show
```

`--logs <目录>` 可指定查找根目录；`--latest` 优先使用运行文件夹名的日期时间，而非拷贝时间。
默认跳过 metadata 中标记 `dry_run: true` 的模拟数据；可用 `--include-simulation` 纳入。
同一次日志若保留多个副本，建议直接指定目录避免选择歧义。

## 需要拷贝哪些文件

推荐拷贝整次运行目录。核心数据：

- `raw_depth.jsonl`：绝对压力（Pa）、日志自带深度、温度。
- `raw_imu.jsonl`：roll/pitch/yaw（度）、acc_mps2（三轴加速度）、gyro_radps（三轴角速度）。
- `raw_power.jsonl`：电压、电流、功率。
- `commands.jsonl`、`events.jsonl`：动作开始与完成时刻。
- `metadata.yaml`、`calibration.json` 或 `calibration_20260914.json`：本次参数与压力换算依据。

传感器文件不齐也可以运行，报告会提示缺失项；三个传感器都没有有效数据时退出。
独立压力记录脚本生成的 `pressure.csv` 与诊断脚本的另一种 JSONL 格式不在本脚本支持范围内。

## 输出

默认保存在运行目录下的 `analysis_20260921/`：

- `report.html`：浏览器报告，将全部曲线、数据提示、参数和统计汇总在一起。
- `pressure_depth.png`：压力、深度及压力传感器温度。
- `imu_attitude.png`：横滚、俯仰、航向角。
- `imu_motion.png`：三轴加速度和角速度。
- `power.png`：电压、电流、功率。
- `summary.json`：各字段的有效数量、缺失数量、最小值、最大值、均值、标准差及采样间隔。

`--show` 自动用默认浏览器打开报告；不传也会保存结果。可用 `--output <目录>` 指定输出位置。
重复分析会覆盖同一输出目录中的同名文件，不修改原始日志；需要保留不同时间窗口分析时分别指定输出目录。

## 时间与物理量说明

所有传感器以同一单调时间戳对齐。默认零点是首个动作提交时刻，负时间是启动/标定阶段；
没有动作记录时，零点为各传感器中最早的有效采样。绿色背景是首个动作提交至最后一个完成事件，包含保持阶段。
未设置起止时间时统计整个采集过程；若只关心控制过程，指定 `--start-s 0 --end-s <控制结束时刻>`。

压力图显示 kPa，源日志的压力值是 Pa。新版固定策略日志含 `pressure_per_meter` 时，深度按
`(pressure_pa - p_surface_pa) / pressure_per_meter` 重算；旧日志仍使用该次标定中的水面压力、密度和重力重算。
经验系数日志的 `--rho`/`--gravity` 不改变该经验公式。
不使用当前 robot.yaml 的数值。没有参考压力时，只绘制日志自带的 depth_m 并明确提示。
支持以下参数仅用于离线对比，不更改运行配置或源日志：

```bash
python scripts/analyze_control_logs_20260921.py --latest --surface-pressure-pa 104100 --rho 1000 --gravity 9.80665 --output logs/reference_comparison --show
```

IMU 欧拉角直接绘制原始值，未减标定零点，航向角未展开，跨±180°可能有跳变；
三轴加速度包含重力，不能直接当作平动加速度；角速度保持 rad/s。
功率使用传感器报告的 power_w，不强制等于电压与电流的乘积；有限采样可能漏掉瞬时电流峰值。
图中不做平滑，缺失字段用断线表示；明显采样间隔也断开连接。数据问题在报告中列出。

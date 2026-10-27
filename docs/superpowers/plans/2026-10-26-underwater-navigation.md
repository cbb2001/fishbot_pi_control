# 水下导航实施计划

Goal: 实现用户确认并修改后的单目标水下导航，包含定时上浮、稳定 UWB 重定位、压力约束的 IMU 积分。
Architecture: 单 SensorManager、单 PWM 执行器；纯计算与会话分离，复用现有胸鳍/尾鳍策略。
Tech Stack: Python、numpy、现有 YAML/采集/日志模块；pytest。
Spec: `docs/superpowers/specs/2026-10-26-underwater-navigation-design.md`（2026-10-26 用户已确认执行）。

## Global Constraints

- 保留当前用户未提交修改，在当前工作目录实施，不提交/合并；该基线包含用户要求复用的新转向规则。
- 首次校准定位后立即动作；t4 仅限制每一段导航长度。t3 包含等待且不重置。
- 用户新增的“保存摄像头图像”优先于设计前文“关闭相机”；沿用现有相机采集/存图机制。
- 测试和 dry-run 不访问硬件。实机必须 MOVE 确认，no-servo 不创建 PWM。
- 所有完成声明基于实际运行结果；IMU 粗定位不宣称准确度。

## Tasks（各项先运行失败测试，再实现并回归）

1. 执行器分组归中与原子停止。接口：request_hold(agents=..., center=True)、request_center_stop；PWM 线程保留兄弟动作，停止后拒绝提交。测试取消活动/待执行、原子封口、旧暂停保护。
2. 纯计算模块。接口：UnderwaterConfig、UnderwaterScheduler、InertialEstimator、StableUWB；测试坐标、压力约束、延迟回放、数据缺口、稳定位拒绝与去重。
3. 会话状态机及标定。接口：run_session(config, robot, output_dir, ...)，用真实策略/执行器和模拟传感器边界；测试首次立即导航、多轮循环、t3/t4、零点暂停/恢复、取消回执、无舵机诊断。
4. CLI、robot.yaml、中文 README，摄像头和原始日志集成。运行 --help 与全模拟任务，检查产物和退出码。
5. 原始日志回放，记录实际滤波衰减与适用限制；相关旧测试和完整测试回归。
6. 独立代码审阅；重要问题加复现测试后修复，再最终回归。

测试命令统一前缀：`& E:/anaconda3/envs/fishbot-analysis/python.exe -m pytest -q -p no:cacheprovider`，使用工作区内 `--basetemp`。
Task 1: tests/test_underwater_executor_20261026.py + tests/test_rl_servo_executor_20260914.py；Expected: 新行为先失败，实施后全部通过。
Task 2: tests/test_underwater_navigation_20261026.py；Expected: 缺模块先失败，实施后全部通过。
Task 3: tests/test_underwater_navigation_session_20261026.py；Expected: 缺会话先失败，实施后全部通过。
Task 4–6: 上述全部及 tests/test_fixed_dive_20260920_100122.py、tests/test_surface_navigation*.py；Expected: 全部通过，实机 I/O 未调用。

## Review Focus

重点审阅跨线程取消/停止竞争、阶段边界旧回执、采集序号/新鲜度、校准窗口覆盖、压力漂移原点、UWB 半帧/重复与异常筛除后覆盖、延迟校正低通状态恢复、相机生命周期，以及模拟测试无法验证的真实时间和硬件接口。

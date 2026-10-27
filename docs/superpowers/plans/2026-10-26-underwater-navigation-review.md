# 水下导航审阅与验证记录

## 用户确认后的配置统一与回归更新

以下结果替代后文早期“16 failed”的状态。用户确认更新旧测试，并要求配置以 `config/robot.yaml` 为准。

- 旧测试的中位、机械限位、耦合跨度预期改为从原始 robot YAML 独立计算；示例任务内容逐项对照 mission YAML，保留实际标定和动作序列。相关测试 78 passed、23 subtests passed。
- 水下任务的采样率、传感器超时、PWM 写入频率、相机开关及存图、日志参数读取已有共享 YAML 配置；标定阈值、物理参数、循环周期和关闭时限集中到水下配置段。生产入口缺少必要字段或压力采样率与漂移模型不匹配时明确报错，不再静默改写传感器设置。已有命令行参数仅作为显式临时覆盖，实际生效值写入 metadata。
- 新增五项配置来源回归，先确认失败再实现；水下相关测试合计 30 passed。
- 最新全仓库回归：**595 passed、113 subtests passed，190.98 秒，零失败**。命令：`python -m pytest tests -q -p no:cacheprovider --basetemp tmp/robot_yaml_full`；报告：`robot-yaml-regression.txt`。
- YAML 配置集成模拟：临时配置设 IMU 100 Hz、PWM 80 Hz、相机及存图关闭、t1/t2/t3/t4=0/1/8/1 秒，仅传入 `--config`、`--dry-run` 和输出目录。日志中的生效配置一致，完成 Navigate→Surface→Acquire→Navigate，并正常 time_limit 结束；fault=null、cleanup_errors=[]、alive_workers=[]。记录：`logs/underwater_navigation_20261026_yaml_check/metadata.yaml` 和 `result.json`。
- 本次没有运行真实硬件；下文实机验证边界仍适用。相机行为以最新用户要求为准，遵循 YAML 开关；当前正式 robot.yaml 仍开启相机及存图。

独立只读审阅：无 Critical；发现三项 Important，均已添加复现测试并修复：

1. 相机没有首帧、读失败或数据过期时不能正常返回成功。新增运行期图像健康检查；存图 API 返回 false 时也持久记录错误。
2. t3 截止与安全故障并发时不能把低压/过流/传感器故障覆盖成正常结束。停止安全线程后收集 failure，并排空 PWM 关闭错误队列。
3. 相机关闭后的驱动异常、仍运行的图像 writer/index logger 必须报告清理失败。

四个新增故障测试在修复前失败，修复后通过。另新增 TID 重复/反向/回绕、延迟压力重放、异常点恢复、纳秒时限校验测试。

暂留 Minor：PWM started 回执队列尚未消费，长时间任务会累计动作记录；默认 120 秒任务影响有限。完成回执正常消费，未使用 started 队列推进动作。

实机 PWM/I²C 阻塞及相机驱动关闭时延、IMU 硬件同步、压力漂移模型适用性、定位精度无法由本次软件测试确认。有界等待超时会记录失败，但不能保证被底层驱动阻塞的 daemon 线程立即退出。

## 验证依据

- 新任务测试：25 passed（包含多轮 Navigate→Surface→Acquire→Navigate、无舵机模式和故障收尾）。相关回归合计231 passed，见 `underwater-related-final.txt`。
- CLI 15 秒全模拟运行：`logs/underwater_navigation_20261026_implementation_check/result.json`，time_limit、无 fault/cleanup_errors/存活 worker。
- 真实日志回放：`logs/20261019_145059_fixed_dive/navigation_analysis/replay_underwater_20261026.py` 及同名 JSON；12088 IMU，287 mc（264 完整、23 部分），完整帧最长断流 15.456 秒，1.2–1.8 Hz 三轴滤波幅值比 0.224/0.227/0.227。无水平真值，未给出定位精度承诺。
- 第一次全仓库回归：17 failed、567 passed、111 subtests passed。其中 16 项旧中位/几何/示例失败在独立基线副本中复现（撤除本次共享执行器/调度器/YAML 改动，保留用户其余原有修改）。详见 `underwater-baseline-failures.txt`。
- 另一次水面单目标 CLI 日志顺序断言失败，单独基线复跑和最终相关回归均通过。本次不改写用户水面导航逻辑以掩盖不稳定测试。
- 最终全仓库回归：16 failed、576 passed、111 subtests passed，失败项与独立基线复现的16项一致，无新增失败；见 `underwater-test-suite-final.txt`。中间一次因临时父目录缺失的测试环境错误已修复后重跑，未作为实现验证结论。
- 最终8秒CLI模拟在Acquire阶段正常截止，见 `logs/underwater_navigation_20261026_final_check/result.json`：无故障、无清理错误、无存活采集线程。

实现决策：保留当前目录中的用户改动，不自动提交或切换分支；用户新增的相机存图要求优先于设计前文关闭相机的描述。实机与定位准确度仍需水池验证。

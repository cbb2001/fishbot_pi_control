# 10 Hz 水面压力漂移标定

配置文件：`config/pressure_drift_10hz.json`。模型参数、72001 个查表值、单位、时间定义、来源和验证信息均在其中。PPO入口 `scripts/train_rl_ppo_20260914.py` 已接入此补偿，自动以10Hz采集压力并用默认5秒窗口建立本次P0；shell启动器调用同一入口。

## 使用约定

- 条件：同一传感器、相同 10 Hz 采样及转换设置；机器鱼在水面，压力测点完全浸没。启动状态、浸没姿态应尽量与采集实验一致。
- 时间零点为开始连续采样的第一个有效样本，使用单调时钟。不是电机开始运动的时刻。电机停止但传感器持续采样时，计时继续。
- 初始压力 `P0` 为本次同姿态水面初始压力。历史各文件减去各自第一个有效压力值，拟合的 `f(0)=0`。
- 若启动取多个样本降噪，建议 `P0 = mean(P_i - f(t_i))`，避免启动采样窗口内的漂移带入基准；查表计时不能在取均值后重新开始。
- 压力单位为 **Pa**，不是 mbar；若使用 `pressure_mbar`，先乘以 100。
- 水面基准：`P_surface(t) = P0 + f(t)`。
- 相对水面姿态的深度：`h_m = (P_measured - P0 - f(t)) / (rho * g)`，向下为正。测点在初始水面姿态下的浸没深度已包含在 P0 中；这不是测点到自由液面的绝对距离。
- 中断采样、复位或断电后不应直接续用旧计时与 P0；须重新建立相同采集条件并在水面标定。

## 查表读取示例

```python
import json
import math

with open("config/pressure_drift_10hz.json", encoding="utf-8") as f:
    calibration = json.load(f)  # 一次读入整个配置文件

lookup = calibration["lookup"]
values = lookup["pressure_drift_pa"]

def drift_pa(elapsed_s):
    if not math.isfinite(elapsed_s) or not 0 <= elapsed_s <= lookup["end_s"]:
        raise ValueError("elapsed_s must be within [0, 7200]")
    index = (elapsed_s - lookup["start_s"]) / lookup["step_s"]
    left = min(int(index), len(values) - 1)
    right = min(left + 1, len(values) - 1)
    fraction = index - left
    return values[left] + fraction * (values[right] - values[left])

# P0 由本次水面标定得到；elapsed_s 从连续采样起点开始累计。
# surface_pressure_pa = P0 + drift_pa(elapsed_s)
# depth_m = (measured_pressure_pa - surface_pressure_pa) / (1000.0 * 9.80665)
```

表格覆盖 0～7200 秒，间隔 0.1 秒，包含两个端点。不应循环索引。7200 秒以外不属于该表的覆盖范围。

## 拟合与预测范围

采用全部指定的 12 份记录，不包含 `2026092801`，不额外剔除任何指定文件。每条记录先减自身首样本压力，再取 5 秒均值，各记录在最小二乘中具有相同总权重，防止长记录完全主导拟合。

模型（t 单位秒，f 单位 Pa；精确参数以 JSON 为准）：

`f(t) = -2301.091945 * (1-exp(-t/121.493809)) + 943.886937 * (1-exp(-t/1563.954366))`

| 采样时间 | f(t)，Pa |
|---|---:|
| 0 min | 0.00 |
| 10 min | -1983.86 |
| 30 min | -1655.80 |
| 60 min | -1451.66 |
| 120 min | -1366.66 |

模型最终趋向约 -1357.21 Pa。选用趋稳的慢回升项，是依据最长记录中回升减慢的形状，并避免线性模型无界外推；这是一项经验假设，不是已确认的物理规律。

共同曲线的各记录 5 秒均值拟合 RMSE 为 71.6～283.1 Pa；每次留出一整份记录、用其余记录重新拟合，验证 RMSE 为 82.1～320.5 Pa（淡水约 0.84～3.27 cm）。这些是各记录 RMSE 的范围，并非最大误差或置信区间。原始 10 Hz 点的误差另存在 JSON 中。

最长数据约 3600 秒，只有一份记录达到 60 分钟。60～120 分钟为未验证外推；生成满 2 小时数据不等于已验证 2 小时精度。仅以前 30 分钟拟合再预测后续时，60 分钟记录尾段仍有约 299 Pa RMSE，说明初段不足以可靠确定慢回升。后续最好以独立 2 小时恒定浸没记录复核。

图：`logs/analysis_drift_20261001/pressure_drift_fit.png`。
详细误差：`logs/analysis_drift_20261001/fit_metrics.json`。
重建脚本：`scripts/fit_pressure_drift_10hz.py`，需要 numpy、scipy、matplotlib；运行将重新生成本配置及分析产物。

漂移补偿不包含游动流场造成的动压、实际深度变化或与标定不同的环境变化。

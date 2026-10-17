# UWB独立定时采集与绘图

入口：`scripts/record_uwb_20261013.py`。仅打开UWB串口，不创建IMU、压力、电源、相机或舵机设备，不发送运动指令。运行前停止占用同一串口的导航/采集程序。

在项目根目录运行，当前Python环境需安装`pyserial`、`PyYAML`、`matplotlib`；也已在项目requirements中列出：

```bash
python -m pip install pyserial PyYAML matplotlib
python scripts/record_uwb_20261013.py --t 30
```

`t`从打开串口后开始计时，包括没有信号的时间，不含后续解析和绘图。采集复用导航及诊断的公共 `read_uwb_chunk` 接口，使用 `read_until(b'\r\n', size=4096)`，遇到CRLF即返回，4096仅是读取上限。超时片段和空读之前的半帧在解析时保留。串口读取等待按剩余时间缩短；系统调度和文件写入可能带来少量时间误差。Ctrl+C停止后仍导出已收到的数据和图片；串口中途出错会结束采集、保留已有数据，并尝试绘图。

有图形桌面时显示图窗：

```bash
python scripts/record_uwb_20261013.py --t 30 --show
```

普通SSH使用不带`--show`的命令，采集后打开保存的PNG即可。图窗在采集结束后显示，不是实时曲线。

已有日志重新画图，不打开任何串口：

```bash
python scripts/record_uwb_20261013.py --input logs/20261013_140732_surface_navigation/raw_uwb.jsonl
```

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--t` / `--seconds` | 30 | 采集秒数，正有限数；离线模式不使用 |
| `--config` | `config/robot.yaml` | 读取UWB端口、波特率、标签和日志根目录 |
| `--port` | YAML的`sensors.uwb.port`，缺省`/dev/ttyAMA0` | 指定UWB串口 |
| `--baudrate` | YAML的`sensors.uwb.baudrate`，缺省115200 | 波特率，正整数 |
| `--tag` | YAML的`surface_navigation_20261010.uwb_tag`，缺省`t0:0` | 指定解析的mc标签 |
| `--output-dir` | 采集：`runtime.log_dir/uwb_capture_20261013`；离线：输入文件旁的`uwb_plots` | 输出根目录，每次自动新建含微秒时间戳的子目录 |
| `--input` | 无 | 已有`raw_uwb.jsonl`路径；只解析画图 |
| `--show` | 关闭 | 保存图片后弹出窗口，需要可用图形桌面/后端 |

输出：

- `raw_uwb.jsonl`：所有原始串口片段、时间戳及空读/错误记录（仅采集模式生成）。
- `uwb_ranges.csv`：时间（秒）及A0～A3距离（米）。空读/错误用空测距行标记图中断点。
- `uwb_ranges.png`、`uwb_ranges.pdf`：四路距离分图，原始点连线、不平滑，各图纵轴独立。
- `metadata.json`：采集参数、来源、退出原因。
- `summary.json`：完整有效mc帧数、无效mc帧数、空读/错误次数、各路最小/中位/最大值。

解析复用导航的mc流式解析器，按换行拼接拆包，从十六进制毫米转为米；忽略KT0重复显示帧，对重复mc帧去重。0、无效编码、缺字段帧不进入四路距离曲线，但原始字节仍完整保存在日志。这里的“有效帧”仅表示格式和四路数字有效，**不表示定位有效**：不按H、矩形范围或定位残差筛掉异常测距，不计算x/y，因此H不影响纯采集和画图。没有有效帧时仍输出CSV表头和标有“No valid UWB ranges”的图片。

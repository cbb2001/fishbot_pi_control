# fishbot_pi_control Codex/Raspberry Pi Workflow

本项目的开发方式固定为：Codex 在 Windows 本机 VS Code 环境中编辑本地项目，树莓派 5 只作为远程硬件执行端。

## 固定路径

- 本机项目：当前 VS Code 打开的 `fishbot_pi_control/`
- 树莓派项目：`/home/fish/fishbot_pi_control`
- 树莓派登录：`fish@<自动发现的 IP>`
- SSH 用户名：`fish`
- SSH 密码：请使用本地私下保存的密码，不要提交到 GitHub。
- 本机辅助脚本目录：`codex_pi_workflow/`

不要把 VS Code 切换到 Remote SSH 工作流。树莓派不需要安装 Codex，也不需要能访问 OpenAI。

树莓派 IP 地址不是固定的。工作流脚本会默认用 `nmap -sn 192.168.1.0/24` 扫描同一局域网，再用 SSH key 验证哪个地址可以通过 `fish` 用户登录。如果 `nmap` 因权限不足失败，用管理员 PowerShell 运行脚本，或通过 `-PiIp` 手动指定地址。

如果 SSH/SCP 提示输入密码，输入你本地私下保存的树莓派密码。为避免把密码暴露到 PowerShell 历史、命令行参数或进程列表中，辅助脚本默认不把密码硬编码进 `.ps1` 文件，也不要把密码写进将提交到 GitHub 的文件。

当前 Windows 本机的 `~/.ssh/id_rsa.pub` 已安装到树莓派 `fish` 用户的 `~/.ssh/authorized_keys`，后续应优先使用 SSH key 免密登录。发现并验证树莓派 IP：

```powershell
.\codex_pi_workflow\find_pi.ps1
```

如果 Windows PowerShell 提示“禁止运行脚本”，使用一次性的执行策略旁路：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\find_pi.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\sync_to_pi.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\setup_pi_venv.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\codex_pi_workflow\run_on_pi.ps1 scripts/test_i2c.py
```

## 本机与树莓派边界

Windows 本机只用于：

- 编辑代码
- 运行 Codex
- 同步文件到树莓派
- 做不触碰硬件的静态检查，例如 `python -m py_compile`

树莓派只用于：

- 运行 I2C/UART/GPIO/PWM/PCA9685/IMU/INA219/深度传感器/舵机相关 Python 文件
- 执行硬件检测命令

不要在 Windows 本机直接运行硬件脚本。

## 常用命令

发现树莓派当前 IP：

```powershell
.\codex_pi_workflow\find_pi.ps1
```

同步本机代码到树莓派：

```powershell
.\codex_pi_workflow\sync_to_pi.ps1
```

如果树莓派端还没有虚拟环境：

```powershell
.\codex_pi_workflow\setup_pi_venv.ps1
```

默认远程运行 `main.py`：

```powershell
.\codex_pi_workflow\run_on_pi.ps1
```

远程运行指定脚本：

```powershell
.\codex_pi_workflow\run_on_pi.ps1 scripts/test_i2c.py
```

如果已经知道当前 IP，可以跳过扫描：

```powershell
.\codex_pi_workflow\sync_to_pi.ps1 -PiIp 192.168.1.113
.\codex_pi_workflow\run_on_pi.ps1 -PiIp 192.168.1.113 scripts/test_i2c.py
```

如果局域网不是 `192.168.1.0/24`，可以指定网段：

```powershell
.\codex_pi_workflow\find_pi.ps1 -Subnet 192.168.0.0/24
```

也可以在发现 IP 后直接 SSH：

```powershell
ssh fish@<树莓派IP> "cd /home/fish/fishbot_pi_control && source .venv/bin/activate && python3 scripts/test_i2c.py"
```

## 同步范围

`sync_to_pi.ps1` 只同步以下内容：

- `main.py`
- `requirements.txt`
- `config/`
- `drivers/`
- `control/`
- `scripts/`
- `missions/`

不同步：

- `.git`
- `.venv`
- `__pycache__`
- `*.pyc`
- `.vscode`
- `codex_pi_workflow/`

`codex_pi_workflow/` 是本机工作流文件夹，不复制到树莓派。

同步脚本会在确认远程目录严格等于 `/home/fish/fishbot_pi_control` 后，先删除远端同步范围内的同名文件/目录，再上传当前本机版本。因此，本机在 `config/`、`drivers/`、`control/`、`scripts/`、`missions/` 内删除的文件，会在下一次同步后从树莓派项目目录中消失。脚本不会删除 `/home/fish` 下其他文件，也不会同步或删除树莓派项目里的 `.venv`。

同步完成后，脚本会确保树莓派项目根目录下存在 `captures/`，用于保存摄像头采集数据。`captures/` 不纳入镜像同步范围，避免后续同步误删树莓派上已经采集的图片或视频。

## 硬件安全顺序

舵机相关测试必须保守推进：

1. 不要一开始运行完整游动步态。
2. 不要直接大角度扫动。
3. 先同步代码。
4. 先运行非运动检测：`scripts/test_i2c.py`、`scripts/test_uart.py`。
5. 深度计 MS5837 读取测试可运行：`scripts/test_depth_sensor.py`。
6. 确认供电、机械限位、鱼体固定安全后，才运行 `scripts/test_servo_center.py`。
7. 居中正常后，才运行 `scripts/test_servo_small_step.py`。
8. 小角度测试默认只允许单个舵机在约 +/-5 度范围内运动。
9. 需要停止 PWM 时运行 `scripts/emergency_stop.py`。

深度计例程中的 IIC/I2C 地址：

- STM32 例程写地址：`0xEC`
- STM32 例程读地址：`0xED`
- 树莓派 Linux/smbus 使用的 7-bit 地址：`0x76`

推荐顺序：

```powershell
.\codex_pi_workflow\sync_to_pi.ps1
.\codex_pi_workflow\run_on_pi.ps1 scripts/test_i2c.py
.\codex_pi_workflow\run_on_pi.ps1 scripts/test_uart.py
.\codex_pi_workflow\run_on_pi.ps1 scripts/test_depth_sensor.py
.\codex_pi_workflow\run_on_pi.ps1 scripts/test_servo_center.py
.\codex_pi_workflow\run_on_pi.ps1 scripts/test_servo_small_step.py
```

如果某个测试失败，读取完整 SSH 输出，先判断原因，再修改本机代码、同步、远程重试。不要盲目重复运行同一个失败命令。

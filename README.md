# fishbot_pi_control

树莓派鱼形机器人控制项目。当前项目在 Windows 本机编辑代码，通过 GitHub 管理版本，并同步到树莓派执行硬件测试。

## 项目结构

```text
fishbot_pi_control/
├── config/              # 机器人配置
├── control/             # 控制、安全、PID、步态等逻辑
├── drivers/             # 传感器、舵机、UWB、IMU 等驱动代码
├── scripts/             # 硬件测试脚本
├── codex_pi_workflow/   # Windows 到树莓派的同步/运行辅助脚本
├── captures/            # 摄像头采集数据，本目录不提交到 GitHub
├── main.py
└── requirements.txt
```

## 推荐开发流程

1. 在 Windows 本机修改代码。
2. 用 Git 提交并推送到 GitHub：

   ```powershell
   git status
   git add .
   git commit -m "说明这次改了什么"
   git push
   ```

3. 同步代码到树莓派：

   ```powershell
   .\codex_pi_workflow\sync_to_pi.ps1 -PiIp 192.168.1.111
   ```

4. 在树莓派上运行测试脚本。
5. 测试通过后，再次 `git commit` / `git push` 保存稳定版本。

## 常用命令

查看 Git 状态：

```powershell
git status
```

查看提交历史：

```powershell
git log --oneline
```

上传到 GitHub：

```powershell
git push
```

同步到树莓派：

```powershell
.\codex_pi_workflow\sync_to_pi.ps1 -PiIp 192.168.1.111
```

远程运行树莓派脚本：

```powershell
.\codex_pi_workflow\run_on_pi.ps1 -PiIp 192.168.1.111 scripts/test_i2c.py
```

## 摄像头采集

双目 USB 摄像头测试脚本：

```bash
cd ~/fishbot_pi_control/scripts
python3 test_usb_camera.py --frames 10 --save-all
```

采集结果默认保存在树莓派：

```text
~/fishbot_pi_control/captures/
```

`captures/` 用于保存图片/视频数据，不提交到 GitHub。

## 注意事项

- 不要把树莓派密码、SSH 私钥、GitHub token 等敏感信息提交到 GitHub。
- `.venv/`、`.vscode/`、`__pycache__/`、`captures/` 已通过 `.gitignore` 忽略。
- Windows 本机只用于编辑代码、Git 管理和同步文件。
- 树莓派用于运行 I2C、UART、GPIO、PWM、摄像头等硬件测试。
- 舵机测试前先确认供电、机械限位和鱼体固定安全。

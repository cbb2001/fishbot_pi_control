# Fishbot Sensor Recording Notes

This project stage is sensor recording only. Do not use these commands to start
servo motion, gait, PCA9685 output, RL policy, or closed-loop control.

## UWB UART0 serial console

UWB uses `/dev/ttyAMA0` at 115200 baud. If `/dev/ttyAMA0` is occupied by the
Linux serial console, disable the serial getty on the Raspberry Pi:

```bash
sudo systemctl stop serial-getty@ttyAMA0.service
sudo systemctl disable serial-getty@ttyAMA0.service
```

## Offline recording

Underwater Wi-Fi or SSH can disconnect. Start recording as a local process on
the Raspberry Pi before submerging the robot.

Manual nohup example:

```bash
cd /home/fish/fishbot_pi_control
source .venv/bin/activate
nohup python3 scripts/record_sensors.py --duration 300 > logs/last_run.out 2>&1 &
```

Detached launcher example:

```bash
python3 scripts/start_offline_recording.py --duration 300
```

Systemd service setup:

```bash
bash scripts/install_recorder_service.sh
sudo systemctl start fishbot-recorder.service
sudo systemctl status fishbot-recorder.service
```

After recovery, inspect the latest local logs:

```bash
python3 scripts/check_latest_log.py
```

## Camera image saving

When `sensors.vision.enabled=true` and `sensors.vision.save_frames=true`, camera
frames are saved on the Raspberry Pi under the current run log directory:

```text
logs/YYYYMMDD_HHMMSS_sensor_test/
  synchronized_sensors.jsonl
  camera_index.jsonl
  camera/
    frame_00000001.jpg
```

The camera may be read at 15 Hz, but JPEG saving is capped separately by
`sensors.vision.save_fps`, default 5 FPS. The synchronized JSONL file stores
only camera metadata and the latest relative image path, never image bytes.
If disk writing is too slow, image saves may be skipped while the sensor
recorder continues running.

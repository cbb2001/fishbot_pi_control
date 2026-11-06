#!/usr/bin/env python3
"""Read ONLY the navigation IMU worker; no actuator/controller imports or writes.

Uses the same Yesense decoding and TID-based timestamps as SensorManager.
The hardware-open audit guard allows only the configured IMU device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config/robot.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=600.)
    parser.add_argument('--calibration', type=Path, help='Optional frozen calibration; log corrected data without controlling anything')
    parser.add_argument('--tid-modulus',type=int,help='Explicit firmware TID period; this tested unit wraps 60000 to 1')
    parser.add_argument('--periodic-t4',type=float,help='Bench simulation: one rest update per t4 seconds')
    parser.add_argument('--periodic-params',type=Path,default=ROOT/'config/imu_static_candidate_20261031.json',help='Use prior noise settings only; startup bias is fitted anew')
    parser.add_argument('--periodic-window',type=float,default=2.)
    parser.add_argument('--bench-static-confirmed',action='store_true')
    args = parser.parse_args()
    if args.duration <= 0:
        parser.error('duration must be positive')
    import yaml
    robot = yaml.safe_load(args.config.read_text(encoding='utf8'))
    cfg = robot['sensors']['imu']
    if args.tid_modulus is not None:
        cfg['tid_modulus'] = args.tid_modulus
    port = cfg['port']
    device_opens = []

    def audit(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes)):
            name = str(values[0])
            if name.startswith('/dev/') and name not in (port, '/dev/null', '/dev/urandom'):
                raise PermissionError('IMU-only session blocked device: ' + name)
            if name.startswith('/dev/'):
                device_opens.append(name)
        if event == 'import' and any(word in str(values[0]).lower() for word in
                                      ('servo', 'pca9685', 'smbus', 'pigpio', 'gpio')):
            raise PermissionError('IMU-only session blocked actuator import')

    sys.addaudithook(audit)
    from control.runtime.ring_buffer import RingBuffer
    from control.runtime.uart_workers import IMUWorker
    calibration = None
    if args.calibration:
        from control.runtime.imu_static_calibration_20261031 import FrozenIMUCalibration
        calibration = FrozenIMUCalibration.load(args.calibration)
    periodic = None
    if args.periodic_t4 is not None:
        if not args.bench_static_confirmed:
            parser.error('--periodic-t4 requires --bench-static-confirmed; bench simulation only')
        from control.runtime.imu_periodic_static_20261031 import PeriodicStaticExperiment
        noise_params=json.loads(args.periodic_params.read_text(encoding='utf8'))
        nav=robot['underwater_navigation_20261026']
        periodic=PeriodicStaticExperiment(noise_params['noise'],t4=args.periodic_t4,
            window_s=args.periodic_window,warmup_s=nav['t1'],startup_s=nav['t2'],
            tau=nav['accel_tau_s'],externally_confirmed_static=True)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(started_utc=datetime.now(timezone.utc).isoformat(), imu_config=cfg,
                    duration_requested_s=args.duration, executor_started=False,
                    protocol='Operator confirms fish fixed on stand throughout; no repositioning.',
                    hardware_policy='Only IMU UART opened; no controller, I2C, PWM or servo initialization.',
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__), ROOT/'control/runtime/uart_workers.py',
                                             ROOT/'config/robot.yaml']})
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    (args.output/'capture_source.py').write_bytes(Path(__file__).read_bytes())
    if args.calibration:
        (args.output/'frozen_calibration.json').write_bytes(args.calibration.read_bytes())
        metadata['calibration_sha256'] = hashlib.sha256(args.calibration.read_bytes()).hexdigest()
        metadata['correction_module_sha256'] = hashlib.sha256(
            (ROOT/'control/runtime/imu_static_calibration_20261031.py').read_bytes()).hexdigest()
        (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    if periodic:
        metadata['periodic']=dict(t4=args.periodic_t4,window_s=args.periodic_window,
            warmup_s=periodic.warmup_s,startup_s=periodic.startup_s,tau=periodic.tau,
            externally_confirmed_static=True,updates='once per boundary only',
            position_anchor='separate simulated perfect origin fix; not actual UWB',noise=noise_params['noise'])
        (args.output/'periodic_source.py').write_bytes((ROOT/'control/runtime/imu_periodic_static_20261031.py').read_bytes())
        (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    worker = IMUWorker(RingBuffer(1), port=port, baudrate=cfg['baudrate'],
                       timeout_s=cfg.get('timeout_ms', 100)/1000,
                       sample_rate_hz=cfg['sample_rate_hz'],tid_modulus=cfg.get('tid_modulus',65536))
    stop = False

    def on_signal(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    started = last_report = last_received = time.monotonic()
    count = 0
    fault = None
    first_ns = last_ns = None
    periodic_log = None
    previous_tid = None
    try:
        # PySerial POSIX lock prevents another exclusive reader sharing this UART.
        worker._ensure_serial().exclusive = True
        if periodic:periodic_log=(args.output/'periodic_live.jsonl').open('x',encoding='utf8',buffering=1)
        with (args.output/'raw_imu.jsonl').open('x', encoding='utf8', buffering=1) as output:
            while not stop and time.monotonic()-started < args.duration:
                samples = worker.read_once()
                receive_ns = time.monotonic_ns()
                for sample in samples or []:
                    if not sample.ok:
                        raise RuntimeError(sample.error)
                    row = sample.to_dict()
                    row['receive_t_ns'] = receive_ns
                    if calibration is not None:
                        linear_acc, gyro = calibration.correct(sample.data)
                        row['calibrated_linear_acc_world_mps2'] = linear_acc.tolist()
                        row['calibrated_gyro_radps'] = gyro.tolist()
                    output.write(json.dumps(row, separators=(',', ':'))+'\n')
                    if periodic:
                        tid=sample.data['tid']
                        if previous_tid is not None and (tid-previous_tid)%worker.tid_modulus != 1:
                            raise RuntimeError('Periodic simulation requires contiguous frame sequence')
                        previous_tid=tid
                        event=periodic.step(count/worker.sample_rate_hz,sample.data)
                        if event:
                            periodic_log.write(json.dumps(event,separators=(',', ':'))+'\n')
                        if periodic.models and count%20==0:
                            periodic_log.write(json.dumps(dict(event='state',time_s=count/worker.sample_rate_hz-periodic.origin,
                                models=periodic.snapshot()),separators=(',', ':'))+'\n')
                    count += 1
                    first_ns = sample.t_ns if first_ns is None else first_ns
                    last_ns = sample.t_ns
                    last_received = time.monotonic()
                now = time.monotonic()
                if now-last_received > 3:
                    raise RuntimeError('No valid IMU frames for 3 seconds')
                if now-last_report >= 30:
                    output.flush()
                    print(json.dumps(dict(elapsed_s=round(now-started, 2), samples=count,
                                          last_data=None if not samples else samples[-1].data)), flush=True)
                    last_report = now
    except Exception as exc:
        fault = f'{type(exc).__name__}: {exc}'
    finally:
        worker.close()
        if periodic_log:periodic_log.close()
        result = dict(executor_started=False, device_opens=device_opens, sample_count=count,
                      elapsed_s=time.monotonic()-started, first_t_ns=first_ns, last_t_ns=last_ns,
                      fault=fault, reason='fault' if fault else 'signal' if stop else 'time_limit')
        if periodic:
            result['periodic']=dict(boundaries=periodic.boundary-1,accepted=periodic.corrections,
                eskf_updates=0 if not periodic.models else periodic.models['periodic_eskf_zupt'].update_count)
        (args.output/'result.json').write_text(json.dumps(result, indent=2), encoding='utf8')
        print(json.dumps(result), flush=True)
    return 1 if fault or stop else 0


if __name__ == '__main__':
    raise SystemExit(main())

"""PCA controller construction, writes, recentering and PWM stop share one thread."""
from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .rl_actions_20260914 import (ALL_SERVO_IDS, FIN_SERVO_IDS, TAIL_SERVO_IDS,
                                  TailAction, FinAction, smooth_motion)
from .rl_action_scheduler_20260914 import RLActionScheduler, ScheduledRLAction
from .rl_servo_state_tracker_20260914 import RLServoStateTracker


@dataclass
class RLServoExecutorConfig:
    command_hz: float = 50.0
    safe_recenter_s: float = 2.0
    initial_move_s: float = 0.0

    def __post_init__(self):
        if not math.isfinite(self.command_hz) or self.command_hz <= 0:
            raise ValueError('command_hz must be positive and finite')
        if any(not math.isfinite(v) or v < 0 for v in (self.safe_recenter_s, self.initial_move_s)):
            raise ValueError('servo durations must be finite and nonnegative')

    @property
    def period_ns(self):
        return max(1, int(round(1e9 / self.command_hz)))


class DryRunRLServoController:
    def __init__(self, centers=None, max_commands=2048):
        self.commands = deque(maxlen=max(1, int(max_commands)))
        self.centers = centers or {i:90. for i in ALL_SERVO_IDS}

    def write_angles(self, angles):
        self.commands.append(dict(angles))

    def write_angle(self, channel, angle):
        self.commands.append({int(channel):float(angle)})

    def stop_all(self, *args, **kwargs):
        return None


@dataclass
class _Request:
    request_id: int
    agent: str
    action: TailAction | FinAction
    context: Any


@dataclass(frozen=True)
class RLStartedAction:
    request_id: int
    agent: str
    action: TailAction | FinAction
    trajectory: Any
    start_t_ns: int
    context: Any = None

    @property
    def end_t_ns(self):
        return self.start_t_ns + round(self.action.t * 1e9)


@dataclass
class RLHoldAcknowledgement:
    ack_t_ns: int | None = None
    cancelled: list = field(default_factory=list)
    angles_deg: dict = field(default_factory=dict)
    failure: BaseException | None = None
    _event: threading.Event = field(default_factory=threading.Event, repr=False)

    def wait(self, timeout=None):
        return self._event.wait(timeout)

    def is_set(self):
        return self._event.is_set()


class RLServoExecutor:
    def __init__(self, controller, scheduler: RLActionScheduler,
                 tracker: RLServoStateTracker, config=None, *, clock_ns=None,
                 shutdown_event=None, failure_queue=None, keep_pwm=False,
                 servo_id_to_channel=None, active_servo_ids=ALL_SERVO_IDS, **_):
        self.controller, self.scheduler, self.tracker = controller, scheduler, tracker
        self.config = config if isinstance(config, RLServoExecutorConfig) else RLServoExecutorConfig(**(config or {}))
        self.clock_ns = clock_ns or time.monotonic_ns
        self.shutdown_event = shutdown_event or threading.Event()
        self.stop_event = threading.Event()
        self.emergency_event = threading.Event()
        self.keep_pwm = bool(keep_pwm)
        self.servo_id_to_channel = dict(servo_id_to_channel or {})
        self.active_servo_ids = tuple(active_servo_ids)
        if not self.active_servo_ids or not set(self.active_servo_ids).issubset(ALL_SERVO_IDS):
            raise ValueError('active_servo_ids must contain known servos')
        channels = [self.servo_id_to_channel.get(i, i) for i in self.active_servo_ids]
        if len(channels) != len(set(channels)):
            raise ValueError('active servo channels must be unique')
        self.failure_queue = failure_queue
        self.failure = None
        self._thread = None
        self._ready = threading.Event()
        self._requests = queue.Queue()
        self._started = queue.Queue()
        self._completed = queue.Queue()
        self._next_id = 0
        self._request_lock = threading.Lock()
        self._pending = deque()
        self._recorded_actions = set()
        self.last_lateness_us = 0.
        self.last_successful_servo_ids = ()
        self.last_write_ack_t_ns = None

    def submit(self, agent, action, context=None):
        required = TAIL_SERVO_IDS if isinstance(action, TailAction) else FIN_SERVO_IDS
        if not isinstance(action, (TailAction, FinAction)) or not set(required).issubset(self.active_servo_ids):
            raise ValueError('action uses inactive servo channels')
        expected = 'tail' if isinstance(action, TailAction) else 'fin'
        if agent in ('left_fin', 'action2'):
            agent = 'fin'
        if agent != expected:
            raise ValueError('agent/action mismatch')
        with self._request_lock:
            if self.stop_event.is_set() or self.shutdown_event.is_set():
                raise RuntimeError('servo executor stopped')
            self._next_id += 1
            self._requests.put(_Request(self._next_id, agent, action, context))
            return self._next_id

    def request_hold(self):
        ticket = RLHoldAcknowledgement()
        with self._request_lock:
            if self.stop_event.is_set() or self.shutdown_event.is_set():
                raise RuntimeError('servo executor stopped')
            self._requests.put(ticket)
        return ticket

    @staticmethod
    def _drain(q):
        out = []
        while True:
            try:
                out.append(q.get_nowait())
            except queue.Empty:
                return out

    def drain_started(self):
        return self._drain(self._started)

    def drain_completed(self):
        return self._drain(self._completed)

    @property
    def started_queue(self):
        return self._started

    @property
    def completion_queue(self):
        return self._completed

    def wait_ready(self, timeout=None):
        return self._ready.wait(timeout)

    def is_alive(self):
        return bool(self._thread and self._thread.is_alive())

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)

    def snapshot(self):
        return {'running': self.is_alive(), 'ready': self._ready.is_set(),
                'last_lateness_us': self.last_lateness_us,
                'failure': None if self.failure is None else str(self.failure),
                'active_servo_ids': list(self.active_servo_ids),
                'last_write_ack_t_ns': self.last_write_ack_t_ns,
                'active_actions': {k:v is not None for k,v in self.scheduler.active.items()}}

    def _write(self, refs, *, best_effort=False):
        if self.is_alive() and threading.current_thread() is not self._thread:
            raise RuntimeError('only the servo executor thread may write PWM')
        successes, errors = [], []
        self.last_successful_servo_ids = ()
        for sid in self.active_servo_ids:
            angle = float(refs[sid])
            lo, hi = self.scheduler.limits[sid]
            if not math.isfinite(angle) or not lo <= angle <= hi:
                raise ValueError(f'servo {sid} angle outside limits')
        for sid in self.active_servo_ids:
            if self.emergency_event.is_set() or self.shutdown_event.is_set():
                raise RuntimeError('emergency stop: angle writes cancelled')
            channel = self.servo_id_to_channel.get(sid, sid)
            try:
                # 单通道确认保留部分成功信息，批量 API 无法报告中途总线失败。
                if hasattr(self.controller, 'write_angle'):
                    self.controller.write_angle(channel, refs[sid])
                else:
                    self.controller.write_angles({channel:refs[sid]})
                ack = self.clock_ns()
                self.tracker.record_successful_batch(ack, {sid:refs[sid]})
                self.last_write_ack_t_ns = ack
                successes.append(sid)
                self.last_successful_servo_ids = tuple(successes)
            except BaseException as exc:
                errors.append(exc)
                if not best_effort:
                    raise
        if errors:
            raise errors[0]
        return self.last_write_ack_t_ns

    def tick_once(self, scheduled_t_ns=None):
        now = self.clock_ns()
        scheduled = now if scheduled_t_ns is None else int(scheduled_t_ns)
        refs, done = self.scheduler.tick(now)
        refs = {sid:refs[sid] for sid in self.active_servo_ids}
        for item in self.scheduler.active.values():
            if item is not None and id(item) not in self._recorded_actions:
                self.tracker.record_trajectory(item.start_t_ns, item.trajectory)
                self._recorded_actions.add(id(item))
        ack = self._write(refs)
        self.last_lateness_us = max(0., (ack-scheduled)/1000.)
        for item in done:
            committed = self.scheduler.complete(item.agent, endpoint_written=True, completion_t_ns=ack)
            self._recorded_actions.discard(id(item))
            self._completed.put(committed)
        return refs

    def _hold(self, ticket):
        try:
            now = self.clock_ns()
            refs = self.scheduler.references_at(now)
            refs = {sid:refs[sid] for sid in self.active_servo_ids}
            ack = self._write(refs)
            ticket.cancelled = self.scheduler.hold(refs)
            ticket.cancelled.extend(self._pending)
            self._pending.clear()
            self._recorded_actions.clear()
            self.tracker.record_reference_segment(ack, ack, refs, refs, phase='hold')
            ticket.ack_t_ns, ticket.angles_deg = ack, refs
        except BaseException as exc:
            ticket.failure = exc
            raise
        finally:
            ticket._event.set()

    def _consume_requests(self):
        for request in self._drain(self._requests):
            if isinstance(request, RLHoldAcknowledgement):
                self._hold(request)
            else:
                self._pending.append(request)
        waiting = deque()
        while self._pending:
            req = self._pending.popleft()
            if self.scheduler.active[req.agent] is not None:
                waiting.append(req)
                continue
            item = self.scheduler.schedule(req.agent, req.action, start_t_ns=self.clock_ns())
            item.context, item.request_id = req.context, req.request_id
            self.tracker.record_trajectory(item.start_t_ns, item.trajectory)
            self._recorded_actions.add(id(item))
            self._started.put(RLStartedAction(req.request_id, item.agent, item.action,
                                             item.trajectory, item.start_t_ns, req.context))
        self._pending = waiting

    def _report_failure(self, exc, code='SERVO_EXECUTOR_FAILED'):
        if self.failure is None:
            self.failure = exc
        if self.failure_queue is not None:
            self.failure_queue.put({'fault_source':'RLServoExecutor', 'fault_code':code,
                                    'message':str(exc), 't_ns':self.clock_ns(),
                                    'successful_servo_ids':list(self.last_successful_servo_ids)})

    def _recenter(self):
        origin = self.tracker.latest_commanded_angles()
        target = {sid:self.scheduler.centers[sid] for sid in self.active_servo_ids}
        origin = {sid:origin[sid] for sid in self.active_servo_ids}
        start = self.clock_ns()
        duration = self.config.safe_recenter_s
        end = start + round(duration*1e9)
        self.scheduler.hold(origin)
        self.tracker.record_reference_segment(start, end, origin, target, phase='safe_recenter')
        next_due = start
        first_error = None
        while True:
            if self.emergency_event.is_set() or self.shutdown_event.is_set():
                break
            now = self.clock_ns()
            elapsed = (now-start)/1e9
            refs = {sid:(smooth_motion(origin[sid], target[sid], elapsed, duration)
                         if duration > 0 else target[sid]) for sid in self.active_servo_ids}
            try:
                self._write(refs)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
                    self._report_failure(exc, 'SERVO_RECENTER_FAILED')
                break
            if now >= end:
                break
            # 落后的周期直接跳过，避免退出时也形成补发积压。
            next_due = max(next_due+self.config.period_ns, self.clock_ns()+self.config.period_ns)
            time.sleep(max(0., min(next_due, end)-self.clock_ns())/1e9)
        return first_error

    def _run(self):
        controller_ready = False
        try:
            if self.stop_event.is_set() or self.shutdown_event.is_set():
                return
            if callable(self.controller) and not hasattr(self.controller, 'write_angle') and not hasattr(self.controller, 'write_angles'):
                self.controller = self.controller()
            controller_ready = True
            self.tick_once()
            settle_end = self.clock_ns()+round(self.config.initial_move_s*1e9)
            while self.clock_ns() < settle_end and not self.stop_event.is_set() and not self.shutdown_event.is_set():
                self.tick_once()
                self.stop_event.wait(min(self.config.period_ns/1e9, .02))
            if self.stop_event.is_set() or self.shutdown_event.is_set():
                return
            self._ready.set()
            due = self.clock_ns()
            while not self.stop_event.is_set() and not self.shutdown_event.is_set():
                self._consume_requests()
                now = self.clock_ns()
                if due > now:
                    self.stop_event.wait((due-now)/1e9)
                if self.stop_event.is_set() or self.shutdown_event.is_set():
                    break
                self.tick_once(due)
                now = self.clock_ns()
                due += max(1, (now-due)//self.config.period_ns+1)*self.config.period_ns
        except BaseException as exc:
            self._report_failure(exc)
            self.stop_event.set()
        finally:
            if controller_ready:
                # 故障时不再移动回中，避免低电压、过流或未知姿态下继续施力。
                if self.failure is None and not self.emergency_event.is_set() and not self.shutdown_event.is_set():
                    try:
                        self._recenter()
                    except BaseException as exc:
                        self._report_failure(exc, 'SERVO_RECENTER_FAILED')
                if not self.keep_pwm or self.failure is not None or self.emergency_event.is_set() or self.shutdown_event.is_set():
                    channels = [self.servo_id_to_channel.get(i, i) for i in self.active_servo_ids]
                    if hasattr(self.controller, 'stop'):
                        # 某通道关闭失败也继续尝试其余通道；仅操作侧鳍活动通道。
                        for channel in channels:
                            try:
                                self.controller.stop(channel)
                            except BaseException as exc:
                                self._report_failure(exc, 'SERVO_PWM_STOP_FAILED')
                    else:
                        try:
                            self.controller.stop_all(channels)
                        except BaseException as exc:
                            self._report_failure(exc, 'SERVO_PWM_STOP_FAILED')
            self.stop_event.set()
            self.scheduler.hold({sid: self.tracker.latest_commanded_angles()[sid]
                                 for sid in self.active_servo_ids})
            self._pending.clear()
            for request in self._drain(self._requests):
                if isinstance(request, RLHoldAcknowledgement):
                    request.failure = self.failure or RuntimeError('executor stopped before hold')
                    request._event.set()

    def start(self):
        if self._thread is not None:
            raise RuntimeError('executor can only be started once')
        self._thread = threading.Thread(target=self._run, name='RLServoExecutor', daemon=True)
        self._thread.start()

    def stop(self, timeout=5.):
        self.stop_event.set()
        if self._thread:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise TimeoutError('servo executor did not stop within timeout')

    def request_emergency_stop(self):
        """可由监测线程调用；只置位，关闭 PWM 的 I/O 仍在执行器线程执行。"""
        self.emergency_event.set()
        self.stop_event.set()

    shutdown = stop


def create_rl_servo_executor(controller, scheduler, tracker, config=None, **kwargs):
    return RLServoExecutor(controller, scheduler, tracker, config, **kwargs)


ServoExecutor = RLServoExecutor

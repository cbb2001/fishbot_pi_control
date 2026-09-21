"""Thread-safe causal RL state history and independent periodic sampling."""
from __future__ import annotations

import math
import queue
import statistics
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any

from .ring_buffer import RingBuffer


@dataclass(frozen=True)
class _StampedState:
    t_ns: int
    state: Any


class RLStateBuffer:
    """Reuse RingBuffer locking/storage and never interpolate future samples."""
    def __init__(self, maxlen=20000):
        self.buffer = RingBuffer(int(maxlen))
        self._append_lock = threading.Lock()

    def append(self, state, value=None):
        if value is None:
            t_ns = state.get('t_ns') if isinstance(state,dict) else getattr(state,'t_ns',None)
            value = state
        else:
            t_ns = state
        if t_ns is None:
            raise ValueError('state requires t_ns')
        item = _StampedState(int(t_ns),value)
        with self._append_lock:
            latest = self.buffer.latest()
            if latest is not None and item.t_ns <= latest.t_ns:
                raise ValueError('state timestamps must strictly increase')
            self.buffer.append(item)
        return value

    put = append

    def latest(self):
        sample = self.buffer.latest()
        return None if sample is None else sample.state

    def latest_before(self, t_ns):
        sample = self.buffer.get_latest_before(int(t_ns))
        return None if sample is None else sample.state

    get_latest_before = latest_before
    sample = latest_before

    def between(self, start_ns, end_ns, include_end=True):
        start,end = int(start_ns),int(end_ns)
        if end < start:
            raise ValueError('interval end precedes start')
        return [x.state for x in self.buffer.snapshot()
                if (start <= x.t_ns <= end if include_end else start <= x.t_ns < end)]

    interval = between

    def snapshot(self):
        return [item.state for item in self.buffer.snapshot()]

    def __len__(self):
        return len(self.buffer)

    def reward_stats(self, start_ns, end_ns):
        rows = self.between(start_ns,end_ns)
        if not rows:
            raise ValueError('action interval contains no RL reward samples')
        return summarize_rewards(rows)


def summarize_rewards(states):
    """Sample mean, never a sum: long actions must not earn more by sampling."""
    rows = list(states)
    if not rows:
        raise ValueError('cannot summarize an empty reward interval')
    components = []
    for row in rows:
        reward = row.get('reward') if isinstance(row,dict) else row.reward
        if reward is None:
            raise ValueError('state has no reward; configure RLStateBuilder reward calculator')
        record = reward if isinstance(reward,dict) else reward.as_dict()
        components.append(record)
    totals = [float(x.get('reward',x.get('reward_total'))) for x in components]
    if not all(math.isfinite(v) for v in totals):
        raise ValueError('reward interval contains NaN/Inf')
    result = {'reward_sample_count':len(rows),'reward_mean':statistics.fmean(totals),
              'reward_min':min(totals),'reward_max':max(totals)}
    for name in ('r_depth','r_heading','r_pitch','r_roll'):
        result['mean_'+name] = statistics.fmean(float(x[name]) for x in components)
    return result


class RLStateSyncWorker:
    """Absolute monotonic schedule, finite joins, exceptions delivered to main.

    Loggers are existing asynchronous JsonlLogger instances. No inference or
    training runs in this worker, and no work runs in ServoExecutor's thread.
    """
    def __init__(self, builder, buffer=None, rate_hz=30., stop_event=None, logger=None,
                 clock_ns=time.monotonic_ns, *, failure_queue=None, sync_logger=None,
                 raw_loggers=None, wait_fn=None):
        rate_hz = float(rate_hz)
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError('state sample frequency must be positive and finite')
        self.builder = builder
        self.buffer = buffer if buffer is not None else RLStateBuffer()
        self.period_ns = max(1,round(1e9/rate_hz))
        self.stop_event = stop_event or threading.Event()
        self.logger,self.sync_logger,self.raw_loggers = logger,sync_logger,raw_loggers
        self.clock_ns = clock_ns
        self.failure_queue = failure_queue
        self.wait_fn = wait_fn or (lambda event,seconds:event.wait(max(0.,seconds)))
        self.thread = None
        self.error = None
        self.last_error = None
        self.skipped_ticks = 0
        self.sample_count = 0
        self.last_update_t_ns = None

    @property
    def skipped_tick_count(self):
        return self.skipped_ticks

    def start(self):
        if self.is_alive():
            return
        if self.error is not None:
            self.raise_if_failed()
        self.thread = threading.Thread(target=self._run_guarded,name='rl-state-sync',daemon=True)
        self.thread.start()

    def join(self, timeout=5.):
        if self.thread is not None:
            self.thread.join(timeout)

    def stop(self, timeout=5.):
        self.stop_event.set()
        self.join(timeout)
        if self.is_alive():
            raise RuntimeError('RLStateSyncWorker did not stop within join timeout')

    def is_alive(self):
        return bool(self.thread and self.thread.is_alive())

    def raise_if_failed(self):
        if self.error is not None:
            raise RuntimeError(f'RLStateSyncWorker failed: {self.last_error}') from self.error

    def sample_once(self, scheduled_t_ns):
        state = self.builder.build(int(scheduled_t_ns))
        self.buffer.append(state)
        if self.logger is not None and self.logger.write(state.to_dict()) is False:
            raise RuntimeError('RL state log queue overflow')
        if self.sync_logger is not None:
            # The exact snapshot used by build(), without taking another sensor
            # sample whose content could differ despite having the same query.
            snapshot = getattr(self.builder,'last_synchronized_sample',None)
            if snapshot is not None and self.sync_logger.write(snapshot) is False:
                raise RuntimeError('synchronized sensor log queue overflow')
        if self.raw_loggers is not None:
            sync = self.builder.synchronizer
            if sync is not None and hasattr(sync,'buffers'):
                self.raw_loggers.write_from_buffers(sync.buffers)
        self.sample_count += 1
        self.last_update_t_ns = int(scheduled_t_ns)
        return state

    def _run_guarded(self):
        try:
            self._run()
        except BaseException as exc:
            self.error = exc
            self.last_error = f'{type(exc).__name__}: {exc}'
            if self.failure_queue is not None:
                try:
                    self.failure_queue.put_nowait({'fault_source':'RLStateSyncWorker',
                        'fault_code':getattr(exc,'fault_code','RL_STATE_WORKER_FAILED'),
                        'message':self.last_error,'traceback':traceback.format_exc(),
                        't_ns':self.clock_ns()})
                except queue.Full:
                    pass
            self.stop_event.set()

    def _run(self):
        start = self.clock_ns()
        tick = 0
        while not self.stop_event.is_set():
            due = start+tick*self.period_ns
            now = self.clock_ns()
            # Event.wait 在不同平台可能提前返回；只有到达计划时刻才允许采样。
            while now < due and not self.stop_event.is_set():
                self.wait_fn(self.stop_event,(due-now)/1e9)
                now = self.clock_ns()
            if self.stop_event.is_set():
                break
            latest_due = max(0,(now-start)//self.period_ns)
            if latest_due > tick:
                self.skipped_ticks += int(latest_due-tick)
                tick = int(latest_due)
                due = start+tick*self.period_ns
            self.sample_once(due)
            tick += 1

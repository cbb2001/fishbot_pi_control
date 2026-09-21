"""One bounded learner queue; both policies are always trained as a pair."""

from __future__ import annotations
import queue, threading, time, traceback, copy
from dataclasses import dataclass


@dataclass(frozen=True)
class PairUpdateJob:
    tail_rollout: object
    fin_rollout: object
    source_policy_version: int


@dataclass(frozen=True)
class PairUpdateResult:
    source_policy_version: int
    tail_metrics: dict | None = None
    fin_metrics: dict | None = None
    error: str | None = None


class PPOUpdateWorker:
    def __init__(self, agents, autostart=True):
        self.agents = agents
        self.jobs = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.active = False
        self.tail_candidate_ready = False
        self.fin_candidate_ready = False
        self.thread = None
        self._pair_pending = False
        self.checkpoint_requests = queue.Queue(maxsize=1)
        self.checkpoint_error = None
        self.last_checkpoint = None
        self._update_error = None
        self._error_observed = False
        if autostart:
            self.start()

    @property
    def busy(self):
        return self._pair_pending

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.thread = threading.Thread(
            target=self._run, name="PPOUpdateWorker", daemon=True
        )
        self.thread.start()

    def submit_pair(self, tail_rollout, fin_rollout, source_policy_version):
        with self._lock:
            if self._stop.is_set():
                raise RuntimeError("PPO worker 已停止")
            if self.busy:
                return False
            for name, rollout in (("tail", tail_rollout), ("fin", fin_rollout)):
                if (
                    rollout.source_policy_version != source_policy_version
                    or self.agents[name].policy_version != source_policy_version
                ):
                    raise ValueError("冻结 rollout policy 版本错误")
                if len(rollout) < self.agents[name].config.min_transitions_per_update:
                    raise ValueError("两个 agent 都必须满足最小样本量")
            self._pair_pending = True
            self.tail_candidate_ready = self.fin_candidate_ready = False
            self.jobs.put_nowait(
                PairUpdateJob(tail_rollout, fin_rollout, source_policy_version)
            )
            return True

    def request_checkpoint(self, directory, metadata=None):
        # Captured weights/optimizer refer to immutable published objects. Large torch
        # serialization and all filesystem work happen only in this learner thread.
        request = (
            directory,
            self.agents["tail"].checkpoint_state(),
            self.agents["fin"].checkpoint_state(),
            copy.deepcopy(metadata or {}),
        )
        try:
            self.checkpoint_requests.get_nowait()
        except queue.Empty:
            pass
        self.checkpoint_requests.put_nowait(request)

    def _save_pending_checkpoint(self):
        try:
            request = self.checkpoint_requests.get_nowait()
        except queue.Empty:
            return
        try:
            from .rl_ppo_agent_20260914 import save_checkpoint_pair

            self.last_checkpoint = save_checkpoint_pair(*request)
        except Exception:
            self.checkpoint_error = traceback.format_exc()

    def _run(self):
        while (
            not self._stop.is_set()
            or not self.jobs.empty()
            or not self.checkpoint_requests.empty()
        ):
            self._save_pending_checkpoint()
            try:
                job = self.jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            self.active = True
            try:
                start = time.monotonic()
                tail = self.agents["tail"].update(job.tail_rollout)
                self.tail_candidate_ready = tail is not None
                if tail is not None:
                    tail["update_duration_s"] = time.monotonic() - start
                start = time.monotonic()
                fin = self.agents["fin"].update(job.fin_rollout)
                self.fin_candidate_ready = fin is not None
                if fin is not None:
                    fin["update_duration_s"] = time.monotonic() - start
                self.results.put_nowait(
                    PairUpdateResult(job.source_policy_version, tail, fin)
                )
            except Exception:
                self._update_error = traceback.format_exc()
                self.results.put_nowait(
                    PairUpdateResult(
                        job.source_policy_version, error=self._update_error
                    )
                )
            finally:
                self.active = False
                self.jobs.task_done()

    def poll(self):
        if self.checkpoint_error:
            self._error_observed = True
            raise RuntimeError(f"checkpoint 后台保存失败:\n{self.checkpoint_error}")
        if (
            self.thread is not None
            and not self.thread.is_alive()
            and not self._stop.is_set()
        ):
            raise RuntimeError("PPO worker 意外退出")
        try:
            r = self.results.get_nowait()
            self._pair_pending = False
            if r.error:
                self._error_observed = True
            return r
        except queue.Empty:
            return None

    def drain_results(self):
        r = self.poll()
        return [] if r is None else [r]

    def stop(self, wait=True):
        self._stop.set()
        if wait and self.thread:
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                raise RuntimeError("PPO worker 退出超时")
            error = self.checkpoint_error or self._update_error
            if error and not self._error_observed:
                self._error_observed = True
                raise RuntimeError(f"PPO worker 退出时存在未处理故障:\n{error}")

    close = stop


@dataclass(frozen=True)
class FinUpdateResult:
    source_policy_version: int
    metrics: dict | None = None
    error: str | None = None
    cancelled: bool = False


class FinPPOUpdateWorker:
    """One Fin learner with cooperative cancellation between minibatches.

    The runner holds the final commanded pose while busy. This worker never
    publishes policies, moves actuators, or touches checkpoint files.
    """

    def __init__(self, agent):
        self.agent = agent
        self.jobs = queue.Queue(maxsize=1)
        self.results = queue.Queue(maxsize=1)
        self.cancel_event = threading.Event()
        self.stop_event = threading.Event()
        self.busy = False
        self.active = False
        self.started_t = None
        self._error = None
        self._error_observed = False
        self.thread = threading.Thread(
            target=self._run, name="FinPPOUpdateWorker", daemon=False
        )
        self.thread.start()

    def submit(self, rollout):
        if self.stop_event.is_set():
            raise RuntimeError("Fin PPO worker 已关闭")
        if self.busy:
            return False
        if rollout.source_policy_version != self.agent.policy_version:
            raise ValueError("Fin rollout 来源版本不一致")
        self.cancel_event.clear()
        self.busy = True
        self.started_t = time.monotonic()
        self.jobs.put_nowait(rollout)
        return True

    def cancel(self):
        self.cancel_event.set()

    def _clear_candidate(self):
        self.agent.candidate_model = None
        self.agent.candidate_normalizer = None
        self.agent.candidate_optimizer_state = None
        self.agent.candidate_policy_version = None

    def _run(self):
        while not self.stop_event.is_set() or not self.jobs.empty():
            try:
                rollout = self.jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            self.active = True
            started = time.monotonic()
            try:
                metrics = self.agent.update(rollout, cancel_event=self.cancel_event)
                if metrics is None:
                    raise RuntimeError("Fin PPO rollout 未达到最小样本量")
                if self.cancel_event.is_set():
                    self._clear_candidate()
                    result = FinUpdateResult(
                        rollout.source_policy_version, cancelled=True
                    )
                else:
                    metrics = dict(
                        metrics, update_duration_s=time.monotonic() - started
                    )
                    result = FinUpdateResult(
                        rollout.source_policy_version, metrics=metrics
                    )
            except BaseException:
                self._clear_candidate()
                cancelled = self.cancel_event.is_set()
                self._error = None if cancelled else traceback.format_exc()
                result = FinUpdateResult(
                    rollout.source_policy_version,
                    error=self._error,
                    cancelled=cancelled,
                )
            finally:
                self.active = False
                self.jobs.task_done()
            self.results.put_nowait(result)

    def poll(self):
        if not self.thread.is_alive() and not self.stop_event.is_set():
            raise RuntimeError("Fin PPO worker 意外退出")
        try:
            result = self.results.get_nowait()
        except queue.Empty:
            return None
        self.busy = False
        if result.error:
            self._error_observed = True
        return result

    def close(self, timeout_s=10.0):
        self.cancel_event.set()
        self.stop_event.set()
        self.thread.join(timeout=float(timeout_s))
        if self.thread.is_alive():
            raise TimeoutError("Fin PPO worker 未在退出时限内停止")
        self._clear_candidate()
        self.busy = False
        if self._error and not self._error_observed:
            self._error_observed = True
            raise RuntimeError(f"Fin PPO 后台训练失败:\n{self._error}")

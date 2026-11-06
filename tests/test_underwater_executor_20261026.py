import pytest
import threading
import time

from control.runtime.rl_actions_20260914 import FinAction, TailAction
from control.runtime.rl_action_scheduler_20260914 import RLActionScheduler
from control.runtime.rl_servo_executor_20260914 import RLServoExecutor, DryRunRLServoController
from control.runtime.rl_servo_state_tracker_20260914 import RLServoStateTracker


def executor():
    centers = {1:85., 2:95., 3:95., 4:111., 5:90., 6:143., 7:90.}
    limits = {i:(0., 270.) for i in centers}
    now = [0]
    scheduler = RLActionScheduler(centers, limits=limits)
    ex = RLServoExecutor(DryRunRLServoController(), scheduler,
                         RLServoStateTracker(centers, limits), clock_ns=lambda:now[0])
    return ex, scheduler, now


def test_tail_center_preserves_active_and_queued_fin_and_blocks_tail_until_resume():
    ex, scheduler, now = executor()
    ex.submit('tail', TailAction(10., .3))
    fin_id = ex.submit('fin', FinAction(53., .3, 0, 0))
    ex._consume_requests()
    queued_fin = ex.submit('fin', FinAction(-53., .3, 1, 1))
    ex.submit('tail', TailAction(-10., .3))
    now[0] = 100_000_000
    hold = ex.request_hold(agents=('tail',), center=True, block=True)
    with pytest.raises(RuntimeError): ex.submit('tail', TailAction(10., .3))
    ex._consume_requests()
    assert hold.is_set() and hold.failure is None
    assert scheduler.active['fin'].request_id == fin_id
    assert scheduler.active['tail'] is None
    assert scheduler.committed_angles[1] == 85.
    assert [r.request_id for r in ex._pending] == [queued_fin]
    now[0] = 300_000_000
    ex.tick_once()
    assert ex.drain_completed()[0].request_id == fin_id
    ex.resume_agents(('tail',))
    ex.submit('tail', TailAction(10., .3))


def test_terminal_center_seals_submission_and_cancels_every_queued_action():
    ex, scheduler, now = executor()
    ex.submit('tail', TailAction(10., .3))
    ex.submit('fin', FinAction(53., .3, 0, 0))
    ex._consume_requests()
    ex.submit('tail', TailAction(-10., .3))
    ticket = ex.request_center_stop()
    with pytest.raises(RuntimeError): ex.submit('fin', FinAction(-53., .3, 1, 1))
    with pytest.raises(RuntimeError): ex.resume_agents(('tail',))
    ex._consume_requests()
    assert ticket.is_set() and ticket.failure is None
    assert len(ticket.cancelled) == 3
    assert not ex._pending
    assert ex.stop_event.is_set()
    assert all(item is None for item in scheduler.active.values())
    assert ticket.angles_deg == scheduler.centers


class StopRecordingController(DryRunRLServoController):
    def __init__(self):
        super().__init__()
        self.stopped=threading.Event()
        self.stop_t_ns=None
        self.stopped_channels=None

    def stop_all(self,channels=None):
        self.stop_t_ns=time.monotonic_ns()
        self.stopped_channels=channels
        self.stopped.set()


def terminal_executor(hold_s):
    ex,scheduler,_=executor()
    controller=StopRecordingController()
    ex=RLServoExecutor(controller,scheduler,ex.tracker,
        config={'terminal_center_hold_s':hold_s},
        servo_id_to_channel={i:i-1 for i in scheduler.centers})
    return ex,controller,scheduler


def test_terminal_center_keeps_pwm_active_until_hold_finishes():
    ex,controller,scheduler=terminal_executor(.15)
    ex.start()
    try:
        assert ex.wait_ready(1)
        ex.submit('tail',TailAction(10.,.3))
        ex.completion_queue.get(timeout=1)
        ticket=ex.request_center_stop()
        assert ticket.wait(1) and ticket.failure is None
        assert not controller.stopped.wait(.04)
        with pytest.raises(RuntimeError): ex.submit('tail',TailAction(10.,.3))
        ex.join(1)
        assert not ex.is_alive() and ex.failure is None
        assert controller.stopped_channels==list(range(7))
        assert controller.stop_t_ns-ticket.ack_t_ns >= 150_000_000
        assert ex.terminal_center_hold_completed_t_ns >= ticket.ack_t_ns+150_000_000
        assert ex.pwm_disabled_t_ns >= ex.terminal_center_hold_completed_t_ns
        assert list(controller.commands)[-7:]==[{i-1:scheduler.centers[i]} for i in range(1,8)]
    finally:
        ex.request_emergency_stop();ex.join(1)


@pytest.mark.parametrize('shutdown',['emergency','external'])
def test_terminal_hold_is_interruptible_without_additional_angle_writes(shutdown):
    ex,controller,_=terminal_executor(1.)
    ex.start()
    try:
        assert ex.wait_ready(1)
        ticket=ex.request_center_stop()
        assert ticket.wait(1) and ticket.failure is None
        before=list(controller.commands)
        if shutdown=='emergency': ex.request_emergency_stop()
        else: ex.shutdown_event.set()
        ex.join(.5)
        assert not ex.is_alive() and controller.stopped.is_set()
        assert ex.terminal_center_hold_completed_t_ns is None
        assert controller.stop_t_ns-ticket.ack_t_ns < 1_000_000_000
        assert list(controller.commands)==before
    finally:
        ex.request_emergency_stop();ex.join(1)

import queue
import threading
import time
import unittest

from control.runtime.rl_actions_20260914 import FinAction, TailAction
from control.runtime.rl_action_scheduler_20260914 import RLActionScheduler
from control.runtime.rl_servo_executor_20260914 import RLServoExecutor, DryRunRLServoController
from control.runtime.rl_servo_state_tracker_20260914 import RLServoStateTracker


CENTERS = {1:85.,2:95.,3:95.,4:111.,5:90.,6:143.,7:90.}
LIMITS = {i:(0.,270.) for i in CENTERS}


class RecordingController:
    """无硬件 I/O 边界；记录通道、线程和时间，并可模拟总线失败。"""
    def __init__(self, fail_channel=None):
        self.construct_thread = threading.get_ident()
        self.calls = []
        self.stops = []
        self.fail_channel = fail_channel
        self.failed = False

    def write_angle(self, channel, angle):
        if channel == self.fail_channel and not self.failed:
            self.failed = True
            raise OSError('injected bus failure')
        self.calls.append((threading.get_ident(), time.monotonic_ns(), channel, angle))

    def stop_all(self, channels=None):
        self.stops.append((threading.get_ident(), tuple(channels or ())))


def make_executor(controller, **kwargs):
    scheduler = RLActionScheduler(CENTERS, limits=LIMITS)
    tracker = RLServoStateTracker(CENTERS, LIMITS)
    ex = RLServoExecutor(controller, scheduler, tracker,
                         {'command_hz':100.,'safe_recenter_s':.06},
                         active_servo_ids=(4,5,6,7), **kwargs)
    return ex, scheduler, tracker


class RLServoExecutorTests(unittest.TestCase):
    def test_fin_only_tick_and_completion_ack_preserve_context(self):
        now = [0]
        controller = RecordingController()
        ex, scheduler, tracker = make_executor(controller, clock_ns=lambda: now[0])
        item = scheduler.schedule('fin', FinAction(53,.3,0,1), start_t_ns=0)
        item.request_id, item.context = 8, {'episode':2}
        now[0] = 300_000_000
        ex.tick_once(100_000_000)  # 落后 tick 应按真实时间到终点。
        self.assertEqual({v[2] for v in controller.calls}, {4,5,6,7})
        completed = ex.drain_completed()
        self.assertEqual(len(completed), 1)
        self.assertEqual((completed[0].request_id, completed[0].context), (8, {'episode':2}))
        self.assertEqual(completed[0].completion_t_ns, 300_000_000)
        self.assertEqual(tracker.latest_commanded_angles()[4], 164.)
        ex.tick_once()
        self.assertEqual(controller.calls[-4][3], 164.)
        with self.assertRaises(ValueError): ex.submit('tail', TailAction(10,.3))

    def test_partial_write_failure_records_successful_channels(self):
        controller = RecordingController(fail_channel=5)
        ex, _, tracker = make_executor(controller)
        with self.assertRaises(OSError): ex.tick_once()
        pose = tracker.get_servo_pose_at(time.monotonic_ns())
        self.assertEqual(pose.commanded_angle_deg[4], 111.)
        self.assertIsNone(pose.commanded_angle_deg[5])
        self.assertEqual(ex.last_successful_servo_ids, (4,))

    def test_factory_writes_recenter_and_pwm_stop_share_one_thread(self):
        controllers = []
        def factory():
            controller = RecordingController()
            controllers.append(controller)
            return controller
        ex, _, _ = make_executor(factory)
        ex.start()
        self.assertTrue(ex.wait_ready(1))
        ex.submit('fin', FinAction(53,.3,0,0), context={'episode':4})
        completed = ex.completion_queue.get(timeout=1)
        self.assertEqual(completed.context, {'episode':4})
        ex.stop(timeout=1)
        self.assertFalse(ex.is_alive())
        c = controllers[0]
        self.assertEqual({v[0] for v in c.calls} | {v[0] for v in c.stops}, {c.construct_thread})
        self.assertNotEqual(c.construct_thread, threading.get_ident())
        self.assertEqual({v[2] for v in c.calls}, {4,5,6,7})
        angles = [v[3] for v in c.calls if v[2] == 4]
        self.assertTrue(any(111 < a < 164 for a in angles[angles.index(164)+1:]))
        self.assertEqual(angles[-1], 111.)

    def test_hold_ack_freezes_active_action_and_rejects_ambiguous_resume(self):
        controller = RecordingController()
        ex, scheduler, tracker = make_executor(controller)
        ex.start()
        try:
            self.assertTrue(ex.wait_ready(1))
            rid = ex.submit('fin', FinAction(53,.6,0,0), context={'episode':9})
            ex.started_queue.get(timeout=1)
            time.sleep(.08)
            hold = ex.request_hold()
            self.assertTrue(hold.wait(1))
            self.assertIsNone(hold.failure)
            self.assertEqual(hold.cancelled[0].request_id, rid)
            self.assertEqual(hold.cancelled[0].context, {'episode':9})
            angle = hold.angles_deg[4]
            self.assertGreater(angle, 111.)
            self.assertLess(angle, 164.)
            time.sleep(.04)
            self.assertEqual(tracker.latest_commanded_angles()[4], angle)
            self.assertIsNone(scheduler.active['fin'])
            self.assertTrue(ex.is_alive())
            self.assertEqual(ex.drain_completed(), [])
            # 未完成的行程不能当作完整 action 更新交替状态，需新会话回中。
            with self.assertRaisesRegex(RuntimeError, '重新启动'):
                scheduler.schedule('fin', FinAction(-53,.3,0,0))
        finally:
            ex.stop(timeout=1)

    def test_fault_skips_recenter_and_disables_keep_pwm(self):
        controllers = []
        def factory():
            c = RecordingController(fail_channel=5)
            controllers.append(c)
            return c
        failure_queue = queue.Queue()
        ex, _, _ = make_executor(factory, keep_pwm=True, failure_queue=failure_queue)
        ex.start()
        ex.join(1)
        self.assertFalse(ex.is_alive())
        self.assertIsNotNone(ex.failure)
        c = controllers[0]
        self.assertTrue(c.stops)
        self.assertEqual({v[2] for v in c.calls}, {4})
        self.assertEqual(failure_queue.get_nowait()['successful_servo_ids'], [4])

    def test_emergency_stop_skips_recenter_even_with_keep_pwm(self):
        c = RecordingController()
        ex, _, _ = make_executor(c, keep_pwm=True)
        ex.start()
        try:
            self.assertTrue(ex.wait_ready(1))
            ex.submit('fin', FinAction(53, .3, 0, 0))
            ex.completion_queue.get(timeout=1)
            ex.request_emergency_stop()
            ex.stop(timeout=1)
            self.assertFalse(ex.is_alive())
            self.assertEqual([x[3] for x in c.calls if x[2] == 4][-1], 164.)
            self.assertEqual(c.stops[-1][1], (4, 5, 6, 7))
        finally:
            ex.stop(timeout=1)

    def test_dry_run_keeps_bounded_command_history(self):
        c = DryRunRLServoController(max_commands=3)
        for value in range(10): c.write_angle(4, value)
        self.assertEqual(len(c.commands), 3)
        self.assertEqual(c.commands[-1], {4:9.})

    def test_failed_pwm_disable_still_attempts_other_fin_channels(self):
        class Controller(RecordingController):
            def stop(self, channel):
                self.stops.append((threading.get_ident(), channel))
                if channel == 5:
                    raise OSError('injected disable failure')
        c = Controller()
        ex, _, _ = make_executor(c)
        ex.start()
        self.assertTrue(ex.wait_ready(1))
        ex.request_emergency_stop()
        ex.stop(timeout=1)
        self.assertEqual([x[1] for x in c.stops], [4, 5, 6, 7])
        self.assertIsNotNone(ex.failure)

    def test_first_recenter_error_does_not_retry_angle_writes(self):
        c = RecordingController()
        ex, _, _ = make_executor(c)
        ex.start()
        self.assertTrue(ex.wait_ready(1))
        ex.submit('fin', FinAction(53, .3, 0, 0))
        ex.completion_queue.get(timeout=1)
        # 调用 stop 后第一个回中写入失败；不得继续发送角度。
        attempted = []
        def fail(*args):
            attempted.append(args)
            raise OSError('recenter error')
        c.write_angle = fail
        ex.stop(timeout=1)
        self.assertIsNotNone(ex.failure)
        self.assertTrue(c.stops)
        self.assertEqual(len(attempted), 1)


if __name__ == '__main__': unittest.main()

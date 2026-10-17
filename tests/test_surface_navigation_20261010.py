"""水面导航：实测报文、四基站几何、角度环绕、PI 和仅尾鳍输出。"""
import math
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from control.runtime import surface_navigation_20261010 as nav
from control.runtime.rl_actions_20260914 import build_calibration
from control.runtime.rl_servo_executor_20260914 import DryRunRLServoController, RLServoExecutor
from control.runtime.rl_servo_state_tracker_20260914 import RLServoStateTracker
from control.runtime.rl_servo_executor_20260914 import RLHoldAcknowledgement


REPORT = (
    b'LO=[no solution]\r\n'
    b'mc 00 00001dfe 0000bfbc 0000d292 00005e51 4705 c7 002826cf t0:0 0000\r\n'
    b'$KT0,7.68,49.08,53.91,24.15,null,null,null,null,LO=[no solution]\r\n'
)


@pytest.mark.parametrize('center_hold', [False, True])
def test_hold_writes_center_immediately_only_when_enabled(robot, center_hold):
    calibration = build_calibration(robot)
    scheduler = nav.NavigationScheduler(calibration)
    tracker = RLServoStateTracker(calibration)
    hardware = DryRunRLServoController()
    executor = RLServoExecutor(hardware, scheduler, tracker,
        config={'hold_at_center': center_hold}, clock_ns=lambda: 1_000_000_000,
        active_servo_ids=(1, 2, 3), servo_id_to_channel={1: 0, 2: 1, 3: 2})
    scheduler.schedule('tail', nav.NavigationTailAction(-20., 2.), start_t_ns=0)
    ticket = RLHoldAcknowledgement()
    executor._hold(ticket)
    # 在摆动中途，回中模式直接写三路中位，不先写冻结角、不生成回中轨迹。
    expected = {1:85., 2:95., 3:95.} if center_hold else {1:75., 2:85., 3:85.}
    assert list(hardware.commands) == [{sid-1: angle} for sid, angle in expected.items()]
    assert ticket.is_set() and ticket.failure is None
    assert ticket.angles_deg == expected
    assert scheduler.active['tail'] is None
    # 后续恢复从已确认的中位连续启动，不沿用中断前的角度。
    trajectory = scheduler.schedule('tail', nav.NavigationTailAction(15., .3), start_t_ns=1_000_000_000).trajectory
    assert trajectory.start_angles_deg == expected


@pytest.fixture
def robot():
    return yaml.safe_load((Path(__file__).resolve().parents[1] / 'config/robot.yaml').read_text(encoding='utf-8'))


@pytest.fixture
def cfg():
    return nav.NavigationConfig(L=4., W=3., x_d=3., y_d=1., theta_0=10., theta_00=12.)


def test_real_report_split_at_every_byte_and_no_kt_duplicate():
    decoder = nav.UWBStreamDecoder()
    frames = []
    for byte in REPORT:
        frames.extend(decoder.feed(bytes([byte])))
    assert len(frames) == 1
    assert frames[0].ranges_m == pytest.approx((7.678, 49.084, 53.906, 24.145))
    assert decoder.feed(REPORT) == []  # 同一 mc 测量的重复上报不得触发新动作。


def test_decoder_recovers_after_invalid_frame_and_accepts_next_measurement():
    decoder = nav.UWBStreamDecoder()
    invalid = REPORT.replace(b'00001dfe', b'ffffffff')
    assert decoder.feed(invalid) == []
    assert len(decoder.feed(REPORT)) == 1
    assert len(decoder.feed(REPORT.replace(b'002826cf', b'002826d0'))) == 1
    assert decoder.feed(REPORT.replace(b't0:0', b't1:0')) == []
    decoder.feed(b'x' * 20000)
    assert len(decoder.feed(b'\r\n' + REPORT.replace(b'002826cf', b'002826d1'))) == 1


@pytest.mark.parametrize('x,y,h', [(0., 0., 2.), (4., 3., 2.), (1., 2., 0.), (1., 2., 5.)])
def test_four_ranges_recover_xy_independent_of_common_height(x, y, h):
    ranges = tuple(math.sqrt((x-a)**2 + (y-b)**2 + h*h)
                   for a, b in ((0, 0), (4, 0), (4, 3), (0, 3)))
    assert nav.position_from_ranges(ranges, 4., 3.) == pytest.approx((x, y))


def test_bad_geometry_and_invalid_measurement_rejected():
    for ranges, length, width in [((1, 2, 3, 4), 0, 3), ((1, math.nan, 3, 4), 4, 3),
                                  ((1, 2, 3), 4, 3), ((-1, 2, 3, 4), 4, 3)]:
        with pytest.raises(ValueError):
            nav.position_from_ranges(ranges, length, width)


@pytest.mark.parametrize('x,y', [(0., 0.), (4., 3.), (1., 2.)])
def test_known_anchor_height_recovers_surface_position(x, y):
    ranges = tuple(math.sqrt((x-a)**2+(y-b)**2+1.93**2)
                   for a, b in ((0, 0), (4, 0), (4, 3), (0, 3)))
    assert nav.position_from_ranges(ranges, 4., 3., H=1.93) == pytest.approx((x, y))


def test_known_height_does_not_reject_by_residual_but_checks_minimum_range():
    ranges = tuple(math.sqrt((1-a)**2+(2-b)**2+5.**2)
                   for a, b in ((0, 0), (4, 0), (4, 3), (0, 3)))
    assert nav.position_from_ranges(ranges, 4., 3.) == pytest.approx((1., 2.))
    assert nav.position_from_ranges(ranges, 4., 3., H=1.93) == pytest.approx((1., 2.))
    with pytest.raises(ValueError, match='高度'):
        nav.position_from_ranges((1., 3., 4., 3.), 4., 3., H=1.93)


@pytest.mark.parametrize('limit', [.001, .5])
@pytest.mark.parametrize('ranges,length,width,expected', [
    ((2.612, 50.805, 58.554, 29.136), 51.5, 26., (.731, -3.245)),
    ((2.762, 49.680, 55.041, 25.313), 50., 26.5, (.754, 1.981)),
    ((3.222, 65.528, 54.581, 25.524), 50., 26.5, (-8.056, 19.606)),
])
def test_inconsistent_ranges_return_computed_coordinates(ranges, length, width, expected, limit):
    # 实测日志中的大残差帧也须返回计算解；旧残差阈值不再阻止定位。
    assert nav.position_from_ranges(ranges, length, width, range_residual_limit_m=limit,
                                    H=1.93) == pytest.approx(expected, abs=.001)


@pytest.mark.parametrize('x,y', [(-.1, 1.), (-1., 1.), (4.1, 1.), (5., 1.),
                                  (1., -.1), (1., -1.), (1., 3.1), (1., 4.)])
@pytest.mark.parametrize('height', [None, 1.93])
@pytest.mark.parametrize('tolerance', [0., .25])
def test_outside_position_is_preserved_without_rejection_or_clipping(x, y, height, tolerance):
    # 覆盖四条边附近和明显越界的解；旧容差参数不能改变计算坐标。
    ranges = tuple(math.sqrt((x-a)**2+(y-b)**2+1.93**2)
                   for a, b in ((0, 0), (4, 0), (4, 3), (0, 3)))
    assert nav.position_from_ranges(ranges, 4., 3., tolerance, H=height) == pytest.approx((x, y))


@pytest.mark.parametrize('height', [-1., float('nan'), float('inf')])
def test_height_validation(cfg, height):
    with pytest.raises(ValueError):
        replace(cfg, H=height)


def test_height_yaml_and_cli_override(cfg):
    robot = {'surface_navigation_20261010': dict(L=4., W=3., x_d=1., y_d=2., H=1.93)}
    assert nav.NavigationConfig.from_robot(robot).H == 1.93
    assert nav.NavigationConfig.from_robot(robot, H=0.).H == 0.


def test_imu_calibration_and_filter_across_wrap():
    imu = nav.CalibratedIMU(tau_s=.1, min_samples=4)
    rows = [dict(yaw_deg=y, roll_deg=y, pitch_deg=10.) for y in (179., -179., 179., -179.)]
    imu.calibrate(rows)
    a = imu.update(dict(yaw_deg=-179., roll_deg=-179., pitch_deg=12.), 1_000_000_000)
    assert a == pytest.approx(dict(yaw_deg=1., roll_deg=1., pitch_deg=2.))
    b = imu.update(dict(yaw_deg=179., roll_deg=179., pitch_deg=10.), 1_010_000_000)
    assert 0 < b['yaw_deg'] < 1
    assert 0 < b['pitch_deg'] < 2
    assert imu.update(dict(yaw_deg=50., roll_deg=50., pitch_deg=50.), 1_010_000_000) == b


def test_filter_does_not_average_179_and_minus179_to_zero():
    imu = nav.CalibratedIMU(tau_s=.1, min_samples=2)
    imu.calibrate([dict(yaw_deg=0., roll_deg=0., pitch_deg=0.)] * 2)
    imu.update(dict(yaw_deg=179., roll_deg=179., pitch_deg=0.), 0)
    value = imu.update(dict(yaw_deg=-179., roll_deg=-179., pitch_deg=0.), 100_000_000)
    assert abs(value['yaw_deg']) > 178.


def test_imu_rejects_moving_or_invalid_calibration():
    imu = nav.CalibratedIMU(min_samples=2)
    for rows in [[], [dict(yaw_deg=y, roll_deg=0, pitch_deg=0) for y in (0, 80)],
                 [dict(yaw_deg=0, roll_deg=0, pitch_deg=100)] * 2]:
        with pytest.raises(ValueError):
            imu.calibrate(rows)


@pytest.mark.parametrize('yaw,left,right', [(0., -10., 12.), (-90., -20., 12.), (90., -10., 22.)])
def test_pi_direction_and_bias_formula(cfg, robot, yaw, left, right):
    policy = nav.NavigationPolicy(cfg, build_calibration(robot))
    decision = policy.decide(1., 1., yaw, 1_000_000_000)
    assert decision.theta_left == pytest.approx(left)
    assert decision.theta_right == pytest.approx(right)


@pytest.mark.parametrize('yaw,output,left,right', [(0., 0., -10., 12.),
                                                 (-90., .5, -20., 12.), (90., -.5, -10., 22.)])
def test_signed_pi_output_drives_absolute_amplitude_formula(cfg, robot, yaw, output, left, right):
    policy = nav.NavigationPolicy(cfg, build_calibration(robot))
    decision = policy.decide(1., 1., yaw, 1_000_000_000)
    assert decision.yaw_d_deg == pytest.approx(0.)
    assert decision.yaw_error_deg == pytest.approx(-yaw)
    assert decision.pi_output == pytest.approx(output)
    assert decision.effort == pytest.approx(abs(output))
    assert decision.theta_left == pytest.approx(left)
    assert decision.theta_right == pytest.approx(right)


def test_negative_pi_integral_uses_positive_half_stroke_limit(cfg, robot):
    robot['servo']['channels'][2]['max_angle'] = 109.  # 中位95°，正半程最多+14°。
    policy = nav.NavigationPolicy(replace(cfg, kp=0., ki=1.), build_calibration(robot))
    policy.decide(1., 1., 90., 0)
    decision = policy.decide(1., 1., 90., 1_000_000_000)
    assert decision.theta_left == -10.
    assert decision.theta_right == 14.
    assert policy.integral == pytest.approx(-.1)  # -(14-12)/(30-10)，带符号积分仍需防饱和。
    assert decision.pi_output == pytest.approx(-.1)
    reversed_error = policy.decide(1., 1., -90., 1_100_000_000)
    assert reversed_error.effort == 0.  # 转向改变后清除历史积分。


@pytest.mark.parametrize('yaw,output,left,right', [(-90., 1., -30., 12.), (90., -1., -10., 30.)])
def test_signed_pi_saturates_before_absolute_amplitude(cfg, robot, yaw, output, left, right):
    policy = nav.NavigationPolicy(replace(cfg, kp=100.), build_calibration(robot))
    decision = policy.decide(1., 1., yaw, 0)
    assert decision.pi_output == output and decision.effort == 1.
    assert decision.theta_left == left and decision.theta_right == right


def test_behind_target_uses_atan2_and_shortest_error(cfg, robot):
    policy = nav.NavigationPolicy(replace(cfg, x_d=0., y_d=1.), build_calibration(robot))
    decision = policy.decide(1., 1., -179., 0)
    assert decision.yaw_error_deg == pytest.approx(-1.)
    assert decision.theta_left == -10.
    assert decision.theta_right > 12.


def test_integral_actual_sample_interval_reset_and_antiwindup(cfg, robot):
    policy = nav.NavigationPolicy(replace(cfg, kp=0., ki=1.), build_calibration(robot))
    assert policy.decide(1, 1, -90, 0).effort == 0
    assert policy.decide(1, 1, -90, 500_000_000).effort == pytest.approx(.25)
    assert policy.decide(1, 1, -90, 500_000_000).effort == pytest.approx(.25)
    assert policy.decide(1, 1, -90, 10_000_000_000).effort == pytest.approx(.25)  # 长断流不积分。
    for index in range(1, 30):
        assert 0 <= policy.decide(1, 1, -90, 10_000_000_000 + index*100_000_000).effort <= 1
    assert policy.decide(1, 1, 90, 13_000_000_000).effort == 0  # 换向不继承旧积分。
    assert policy.decide(1, 1, 0, 13_100_000_000).theta_right == 12.


def test_output_respects_theta_max_and_all_three_physical_limits(cfg, robot):
    robot['servo']['channels'][1]['min_angle'] = 88.
    robot['servo']['channels'][2]['max_angle'] = 103.
    policy = nav.NavigationPolicy(replace(cfg, kp=100), build_calibration(robot))
    left = policy.decide(1, 1, -90, 0)
    right = policy.decide(1, 1, 90, 1_000_000_000)
    assert left.theta_left == -7.
    assert right.theta_right == 8.
    policy = nav.NavigationPolicy(replace(cfg, kp=100), build_calibration(yaml.safe_load(
        (Path(__file__).resolve().parents[1] / 'config/robot.yaml').read_text(encoding='utf-8'))))
    assert policy.decide(1, 1, 90, 0).theta_right == 30.


def test_stop_depends_only_on_distance_regardless_of_heading(cfg, robot):
    policy = nav.NavigationPolicy(cfg, build_calibration(robot))
    assert not policy.decide(1, 1, 0, 0).reached
    assert policy.decide(2.99, 1, 90, 1).reached
    assert policy.decide(2.99, 1, 179, 1).reached
    assert policy.decide(2.99, 1, 0, 2).reached
    assert policy.decide(3, 1, 150, 3).reached  # 重合点无目标航向，定义误差为0。
    boundary = nav.NavigationPolicy(replace(cfg, delta_s=.25), build_calibration(robot))
    assert not boundary.decide(2.75, 1, 0, 4).reached  # 等于阈值仍应导航。


@pytest.mark.parametrize('change', [dict(L=0), dict(W=None), dict(x_d=5), dict(y_d=-1),
    dict(theta_max=31), dict(theta_0=30), dict(theta_00=30), dict(theta_0=-1),
    dict(t_0=-1), dict(t1=-1), dict(t2=0), dict(ki=-1), dict(kp=math.nan),
    dict(delta_yaw=0), dict(delta_s=0), dict(imu_filter_tau_s=-1)])
def test_invalid_config_rejected_before_hardware(cfg, change):
    with pytest.raises(ValueError):
        replace(cfg, **change)


@pytest.mark.parametrize('duration', [0., .03])
def test_tail_scheduler_zero_duration_and_no_pectoral_writes(robot, duration):
    calibration = build_calibration(robot)
    scheduler = nav.NavigationScheduler(calibration)
    tracker = RLServoStateTracker(calibration)
    hardware = DryRunRLServoController()
    now = [1_000_000_000]
    executor = RLServoExecutor(hardware, scheduler, tracker, clock_ns=lambda: now[0],
                               active_servo_ids=(1, 2, 3), servo_id_to_channel={1:0, 2:1, 3:2})
    for theta in (-15., 17.):
        scheduler.schedule('tail', nav.NavigationTailAction(theta, duration), start_t_ns=now[0])
        now[0] += round(duration*1e9)
        executor.tick_once()
        completed = executor.drain_completed()
        assert len(completed) == 1 and completed[0].endpoint_written
        assert tracker.latest_commanded_angles()[1] == 85 + theta
    assert all(set(row).issubset({0, 1, 2}) for row in hardware.commands)

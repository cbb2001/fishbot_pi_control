from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from _bootstrap import add_project_root

PROJECT_ROOT = add_project_root()

from control.safety import ensure_not_windows_hardware_run  # noqa: E402


STEREO_MODES: dict[str, tuple[int, int]] = {
    "2560x960": (2560, 960),  # left 1280x960 + right 1280x960
    "2560x720": (2560, 720),  # left 1280x720 + right 1280x720
    "1280x480": (1280, 480),  # left 640x480  + right 640x480
    "640x240": (640, 240),  # left 320x240  + right 320x240
}
DEFAULT_STEREO_MODE = "1280x480"
DEFAULT_CAPTURE_DIR = PROJECT_ROOT / "captures"


def load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit(
            "OpenCV is required to read USB camera frames.\n"
            "Install it in the Raspberry Pi system Python environment, for example:\n"
            "  sudo apt update && sudo apt install -y python3-opencv"
        ) from exc
    return cv2


def parse_device(value: str) -> int | str:
    value = value.strip()
    if value.isdigit():
        return int(value)
    if value.startswith("/dev/video"):
        suffix = value.removeprefix("/dev/video")
        if suffix.isdigit():
            return int(suffix)
    return value


def list_video_devices() -> list[Path]:
    return sorted(Path("/dev").glob("video*"))


def fourcc_to_text(value: float) -> str:
    code = int(value)
    if code <= 0:
        return "unknown"
    chars = []
    for shift in (0, 8, 16, 24):
        char = chr((code >> shift) & 0xFF)
        chars.append(char if char.isprintable() else "?")
    return "".join(chars)


def split_stereo_frame(frame: Any) -> tuple[Any, Any]:
    height, width = frame.shape[:2]
    if width % 2 != 0:
        raise ValueError(f"Stereo frame width must be even, got {width}.")
    mid = width // 2
    left = frame[:, :mid].copy()
    right = frame[:, mid:].copy()
    if left.shape[:2] != right.shape[:2]:
        raise ValueError(f"Stereo split failed: left={left.shape}, right={right.shape}.")
    if height <= 0 or mid <= 0:
        raise ValueError(f"Invalid stereo frame shape: {frame.shape}.")
    return left, right


def save_image(cv2: Any, path: Path, image: Any, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise SystemExit(f"Failed to save {label} image to: {path}")
    print(f"Saved {label} image to: {path}")


def split_image_paths(path: Path) -> tuple[Path, Path]:
    suffix = path.suffix or ".jpg"
    stem = path.stem if path.suffix else path.name
    return (
        path.with_name(f"{stem}_left{suffix}"),
        path.with_name(f"{stem}_right{suffix}"),
    )


def numbered_image_path(path: Path, frame_number: int) -> Path:
    suffix = path.suffix or ".jpg"
    stem = path.stem if path.suffix else path.name
    return path.with_name(f"{stem}_{frame_number:06d}{suffix}")


def save_stereo_images(
    cv2: Any,
    path: Path,
    frame: Any,
    left: Any | None,
    right: Any | None,
    *,
    mono: bool,
    no_save_split: bool,
) -> None:
    save_image(cv2, path, frame, "full stereo")
    if mono or no_save_split:
        return
    if left is None or right is None:
        raise SystemExit("No split stereo frame was captured, so left/right images were not saved.")
    left_path, right_path = split_image_paths(path)
    save_image(cv2, left_path, left, "left")
    save_image(cv2, right_path, right, "right")


def default_capture_filename(mode: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"stereo_{mode}_{timestamp}.jpg"


def resolve_save_frame_path(path: Path | None, mode: str) -> Path:
    if path is None:
        return DEFAULT_CAPTURE_DIR / default_capture_filename(mode)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if path.suffix:
        return path

    return path / default_capture_filename(mode)


def open_camera(cv2: Any, device: int | str, width: int, height: int, fps: float, fourcc: str) -> Any:
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(device)

    if not cap.isOpened():
        raise SystemExit(
            f"Could not open USB camera device: {device!r}. "
            "Check that the camera is plugged in and visible as /dev/video0."
        )

    fourcc = fourcc.strip().upper()
    if fourcc:
        if len(fourcc) != 4:
            raise SystemExit(f"--fourcc must be a 4-character code, got: {fourcc!r}")
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)

    return cap


def main() -> None:
    ensure_not_windows_hardware_run()

    mode_list = ", ".join(STEREO_MODES)
    parser = argparse.ArgumentParser(description="Read side-by-side stereo frames from a USB camera using OpenCV.")
    parser.add_argument("--device", default="/dev/video0", help="Camera index or device path. Default: /dev/video0.")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to read. Use 0 to run until Ctrl+C.")
    parser.add_argument(
        "--mode",
        choices=sorted(STEREO_MODES),
        default=DEFAULT_STEREO_MODE,
        help=f"Requested side-by-side stereo mode. Supported: {mode_list}.",
    )
    parser.add_argument("--width", type=int, default=0, help="Override requested frame width.")
    parser.add_argument("--height", type=int, default=0, help="Override requested frame height.")
    parser.add_argument("--fps", type=float, default=0.0, help="Requested camera FPS. Use 0 to keep camera default.")
    parser.add_argument("--fourcc", default="MJPG", help="Requested camera pixel format. Default: MJPG.")
    parser.add_argument("--log-every", type=int, default=30, help="Print status every N frames.")
    parser.add_argument("--display", action="store_true", help="Show live image window. Requires a desktop/display.")
    parser.add_argument("--display-split", action="store_true", help="Show left and right windows when --display is used.")
    parser.add_argument("--mono", action="store_true", help="Do not split the frame into left/right images.")
    parser.add_argument(
        "--save-frame",
        type=Path,
        help=(
            "Save the last full side-by-side frame to this image path. "
            "Relative paths are resolved under the project root. "
            "If omitted, saves to captures/stereo_<mode>_<timestamp>.jpg."
        ),
    )
    parser.add_argument("--no-save", action="store_true", help="Read frames without saving images.")
    parser.add_argument(
        "--save-all",
        action="store_true",
        help=(
            "Save every successfully read frame. "
            "Without this option, only the last frame is saved after reading finishes."
        ),
    )
    parser.add_argument(
        "--no-save-split",
        action="store_true",
        help="When --save-frame is used, do not also save *_left and *_right images.",
    )
    args = parser.parse_args()

    if args.frames < 0:
        parser.error("--frames must be 0 or a positive integer.")
    if args.log_every <= 0:
        parser.error("--log-every must be a positive integer.")
    if (args.width == 0) != (args.height == 0):
        parser.error("--width and --height must be set together.")
    if args.no_save and args.save_all:
        parser.error("--no-save and --save-all cannot be used together.")
    if args.no_save and args.save_frame is not None:
        parser.error("--save-frame cannot be used with --no-save.")

    cv2 = load_cv2()
    device = parse_device(args.device)
    requested_width, requested_height = STEREO_MODES[args.mode]
    if args.width > 0 and args.height > 0:
        requested_width, requested_height = args.width, args.height
    save_frame_path = None if args.no_save else resolve_save_frame_path(args.save_frame, args.mode)

    detected_devices = list_video_devices()
    if detected_devices:
        print("Detected video devices: " + ", ".join(str(path) for path in detected_devices))
    else:
        print("No /dev/video* devices were found before opening the camera.")

    cap = open_camera(cv2, device, requested_width, requested_height, args.fps, args.fourcc)
    last_frame = None
    last_left = None
    last_right = None
    start_time = time.monotonic()
    last_log_time = start_time
    frame_count = 0
    saved_frame_count = 0

    try:
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = fourcc_to_text(cap.get(cv2.CAP_PROP_FOURCC))
        print(
            f"Camera opened: device={device!r} "
            f"requested={requested_width}x{requested_height} "
            f"resolution={actual_width}x{actual_height} "
            f"reported_fps={actual_fps:.2f} fourcc={fourcc}"
        )
        if not args.mono and actual_width % 2 == 0:
            print(f"Stereo split: full={actual_width}x{actual_height}, left/right={actual_width // 2}x{actual_height}")

        while args.frames == 0 or frame_count < args.frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise SystemExit(
                    f"Camera opened but frame read failed after {frame_count} frame(s). "
                    "Check USB bandwidth, camera permissions, and whether another process is using it."
                )

            last_frame = frame
            if not args.mono:
                try:
                    last_left, last_right = split_stereo_frame(frame)
                except ValueError as exc:
                    raise SystemExit(str(exc)) from exc

            frame_count += 1
            if save_frame_path is not None and args.save_all:
                frame_path = numbered_image_path(save_frame_path, frame_count)
                save_stereo_images(
                    cv2,
                    frame_path,
                    frame,
                    last_left,
                    last_right,
                    mono=args.mono,
                    no_save_split=args.no_save_split,
                )
                saved_frame_count += 1

            if frame_count == 1 or frame_count % args.log_every == 0:
                now = time.monotonic()
                elapsed = max(now - start_time, 1e-9)
                recent_elapsed = max(now - last_log_time, 1e-9)
                recent_fps = args.log_every / recent_elapsed if frame_count > 1 else 0.0
                timestamp = datetime.now().isoformat(timespec="seconds")
                stereo_text = ""
                if last_left is not None and last_right is not None:
                    stereo_text = f" left_shape={last_left.shape} right_shape={last_right.shape}"
                print(
                    f"time={timestamp} "
                    f"frames={frame_count} "
                    f"full_shape={frame.shape}"
                    f"{stereo_text} "
                    f"avg_fps={frame_count / elapsed:.2f} "
                    f"recent_fps={recent_fps:.2f}"
                )
                last_log_time = now

            if args.display:
                cv2.imshow("USB Camera Test", frame)
                if args.display_split and last_left is not None and last_right is not None:
                    cv2.imshow("USB Camera Left", last_left)
                    cv2.imshow("USB Camera Right", last_right)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("Display stopped by user.")
                    break

    except KeyboardInterrupt:
        print("USB camera test stopped by user.")
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()

    if save_frame_path is not None and not args.save_all:
        if last_frame is None:
            raise SystemExit("No frame was captured, so nothing was saved.")
        save_stereo_images(
            cv2,
            save_frame_path,
            last_frame,
            last_left,
            last_right,
            mono=args.mono,
            no_save_split=args.no_save_split,
        )
        saved_frame_count = 1

    print(f"USB camera test finished. Frames read: {frame_count}, saved frame groups: {saved_frame_count}")


if __name__ == "__main__":
    main()

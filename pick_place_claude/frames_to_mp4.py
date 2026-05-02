#!/usr/bin/env python3
"""
Convert a directory of PNG frames into an MP4 video.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def _open_video_writer(*, cv2_module, output_path: Path, fps: float, width: int, height: int):
    codec_candidates = ["avc1", "H264", "mp4v"]
    for codec in codec_candidates:
        writer = cv2_module.VideoWriter(
            str(output_path),
            cv2_module.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise RuntimeError(f"Failed to open MP4 writer for {output_path} with codecs {codec_candidates}")


def _encode_with_ffmpeg(*, frames_dir: Path, output_path: Path, fps: float) -> int:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not installed")

    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"No PNG frames found in {frames_dir}")

    pattern = frames_dir / "frame_%06d.png"
    if not pattern.exists():
        raise RuntimeError(f"Expected sequential frame pattern {pattern}")

    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        f"{fps}",
        "-i",
        str(pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return len(frame_paths)


def encode_png_sequence_to_mp4(*, frames_dir: Path, output_path: Path, fps: float) -> int:
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        raise RuntimeError(f"No PNG frames found in {frames_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        return _encode_with_ffmpeg(frames_dir=frames_dir, output_path=output_path, fps=fps)
    except Exception:
        # Fall back to OpenCV on machines without ffmpeg or without libx264.
        pass

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to encode MP4 on this machine") from exc

    first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")

    height, width = first_frame.shape[:2]

    writer, _codec = _open_video_writer(cv2_module=cv2, output_path=output_path, fps=fps, width=width, height=height)

    written = 0
    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            if frame.shape[:2] != (height, width):
                raise RuntimeError(
                    f"Frame size mismatch for {frame_path}: "
                    f"expected {(height, width)}, got {frame.shape[:2]}"
                )
            writer.write(frame)
            written += 1
    finally:
        writer.release()

    return written


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PNG frames into an MP4 video.")
    parser.add_argument("--frames_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=12.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    frame_count = encode_png_sequence_to_mp4(frames_dir=args.frames_dir, output_path=args.output, fps=args.fps)
    print(f"[frames_to_mp4] output={args.output}")
    print(f"[frames_to_mp4] frames={frame_count}")
    print(f"[frames_to_mp4] fps={args.fps}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Batch-extract frames from every video in a folder with ffmpeg.

Frames are sampled every --i seconds within an optional [--s, --e] time window,
rescaled to a 2:1 aspect ratio (scale=iw:iw/2, equirectangular panorama), and
saved as lossless PNGs named "<video_name>_%08d.png" under the output root.
Time inputs accept hh-mm-ss / mm-ss or plain seconds; -1 means video start/end.
"""
import os
import argparse
import subprocess
from tqdm import tqdm

VIDEO_EXTENSIONS = {'.mp4', '.webm', '.mkv', '.avi', '.flv', '.mov'}


def parse_time(t_str):
    """Parse a time string into seconds.

    - "-1"              -> -1 (sentinel: video start / end);
    - "hh-mm-ss"/"mm-ss" -> total seconds;
    - anything else      -> float seconds.
    """
    t_str = t_str.strip()
    if t_str == "-1":
        return -1
    if '-' in t_str:
        parts = t_str.split('-')
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        raise argparse.ArgumentTypeError(
            "Bad time format: expected hh-mm-ss or mm-ss.")
    try:
        return float(t_str)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Cannot parse time: use hh-mm-ss / mm-ss or plain seconds.")


def sec_to_hhmmss(seconds):
    """Format seconds as an ffmpeg-friendly hh:mm:ss.ss string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def probe_duration(input_path):
    """Return the video duration in seconds via ffprobe, or None on failure."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", input_path],
            capture_output=True, text=True, check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, OSError) as e:
        print(f"Failed to probe duration of {input_path}: {e}")
        return None


def extract_frames_ffmpeg(input_path, output_root, start_time, end_time, interval):
    """Extract frames from one video with ffmpeg.

    - start_time / end_time == -1 -> resolved to 0 / full duration via ffprobe.
    - One frame every `interval` seconds (fps = 1 / interval).
    - Frames are rescaled to a 2:1 aspect ratio (scale=iw:iw/2) and written as
      lossless PNGs "<video_name>_%08d.png" under output_root.
    """
    video_id = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_root, exist_ok=True)

    if start_time == -1 or end_time == -1:
        duration = probe_duration(input_path)
        if duration is None:
            return
        if start_time == -1:
            start_time = 0
        if end_time == -1 or end_time > duration:
            end_time = duration

    output_pattern = os.path.join(output_root, f"{video_id}_%08d.png")
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", sec_to_hhmmss(start_time),
        "-to", sec_to_hhmmss(end_time),
        "-i", input_path,
        "-vf", f"fps={1 / interval},scale=iw:iw/2",
        output_pattern,
    ]

    print(f"Processing: {input_path}")
    subprocess.run(command, check=True)
    print(f"Saved frames to: {output_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from every video in a folder (one frame "
                    "every --i seconds), rescale to a 2:1 aspect ratio and "
                    "save as lossless PNGs. Times accept hh-mm-ss or seconds."
    )
    parser.add_argument("--d", type=str, default="Video/",
                        help="folder containing the input videos")
    parser.add_argument("--o", type=str,
                        default="Frames/",
                        help="output root directory for the extracted frames")
    parser.add_argument("--s", type=parse_time, default="-1",
                        help="start time (hh-mm-ss or seconds); -1 = video start")
    parser.add_argument("--e", type=parse_time, default="-1",
                        help="end time (hh-mm-ss or seconds); -1 = video end")
    parser.add_argument("--i", type=float, default=5.0,
                        help="sampling interval in seconds (one frame every N s)")
    parser.add_argument("--video_start", type=int, default=0,
                        help="index of the first video to process (sorted order)")
    parser.add_argument("--video_num", type=int, default=-1,
                        help="number of videos to process; -1 = all remaining")
    args = parser.parse_args()

    if not os.path.isdir(args.d):
        print(f"Video folder does not exist: {args.d}")
        return

    video_files = sorted(
        f for f in os.listdir(args.d)
        if os.path.isfile(os.path.join(args.d, f))
        and os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    )
    if args.video_num != -1:
        video_files = video_files[args.video_start:args.video_start + args.video_num]
    else:
        video_files = video_files[args.video_start:]
    if not video_files:
        print("No videos to process.")
        return

    for file in tqdm(video_files, desc="Processing videos"):
        try:
            extract_frames_ffmpeg(os.path.join(args.d, file),
                                  args.o, args.s, args.e, args.i)
        except subprocess.CalledProcessError as e:
            print(f"ffmpeg failed on {file}: {e}")


if __name__ == "__main__":
    main()

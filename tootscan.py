#!/usr/bin/env python3
"""
TootScan 💨

Because manually searching hundreds of hours of video for farts and burps
would be unreasonable.

TootScan recursively scans your local video collection, uses YAMNet to detect
AudioSet's "Fart" and "Burping, eructation" classes, merges nearby detections,
and automatically extracts a short video around every glorious event.

Everything runs locally. No cloud. No API. Your gas stays private.

Requirements:
    pip install tensorflow tensorflow-hub numpy
    ffmpeg must be available on PATH.

Example:
    python tootscan.py "D:\\Medal\\Clips" -o "D:\\TootScan\\output"

Even ChatGPT said:
    "For once, we put AI in a really important scientific mission."
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

# Keep TensorFlow startup noise under control. Set this before importing TF.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - friendly CLI failure
    raise SystemExit("Missing dependency: numpy. Run: pip install numpy") from exc

try:
    import tensorflow as tf
    import tensorflow_hub as hub
except ImportError as exc:  # pragma: no cover - friendly CLI failure
    raise SystemExit(
        "Missing TensorFlow dependencies. Run: pip install tensorflow tensorflow-hub"
    ) from exc


MODEL_URL = "https://tfhub.dev/google/yamnet/1"
MODEL_WINDOW_SECONDS = 0.96
MODEL_HOP_SECONDS = 0.48
SAMPLE_RATE = 16_000
CACHE_VERSION = 1
SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".m4v"}
TARGET_CLASSES = {
    "fart": "Fart",
    "burp": "Burping, eructation",
}


@dataclass
class FrameDetection:
    label: str
    center: float
    score: float


@dataclass
class Event:
    label: str
    start: float
    end: float
    peak_time: float
    confidence: float


@dataclass
class ExportedEvent:
    source: str
    label: str
    event_start: float
    event_end: float
    peak_time: float
    confidence: float
    clip_start: float
    clip_end: float
    output: str | None
    cached: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect Fart / Burping sounds in local video clips with YAMNet "
            "and export every matching passage."
        )
    )
    parser.add_argument("input", type=Path, help="Root directory containing TootScan clips")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("tootscan_output"),
        help="Output directory (default: ./tootscan_output)",
    )
    parser.add_argument(
        "--fart-threshold",
        type=float,
        default=0.30,
        help="YAMNet score required for Fart (default: 0.30)",
    )
    parser.add_argument(
        "--burp-threshold",
        type=float,
        default=0.30,
        help="YAMNet score required for Burping, eructation (default: 0.30)",
    )
    parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.80,
        help="Merge same-type detections separated by at most N seconds (default: 0.80)",
    )
    parser.add_argument(
        "--pre",
        type=float,
        default=3.0,
        help="Seconds to include before an event (default: 3.0)",
    )
    parser.add_argument(
        "--post",
        type=float,
        default=4.0,
        help="Seconds to include after an event (default: 4.0)",
    )
    parser.add_argument(
        "--audio-stream",
        type=int,
        default=0,
        help=(
            "Audio stream index to analyze (default: 0). Useful if TootScan stores "
            "microphone/game audio as separate streams."
        ),
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=60.0,
        help="Inference chunk size, in seconds (default: 60)",
    )
    parser.add_argument(
        "--fast-cut",
        action="store_true",
        help=(
            "Use FFmpeg stream-copy for exports. Much faster, but cut start times can "
            "be keyframe-limited. Default re-encodes exported snippets for accurate cuts."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Detect and write detections.json, but do not export video snippets",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore the analysis cache and overwrite existing exported snippets",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.is_dir():
        raise SystemExit(f"Input directory does not exist: {args.input}")
    if args.audio_stream < 0:
        raise SystemExit("--audio-stream must be >= 0")
    if args.chunk_seconds < 2.0:
        raise SystemExit("--chunk-seconds must be >= 2")
    if args.pre < 0 or args.post < 0 or args.merge_gap < 0:
        raise SystemExit("--pre, --post and --merge-gap must be >= 0")
    for name in ("fart_threshold", "burp_threshold"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be between 0 and 1")


def ensure_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit(
            "FFmpeg was not found on PATH. On Windows you can install it with:\n"
            "  winget install Gyan.FFmpeg\n"
            "Then reopen your terminal."
        )
    return ffmpeg


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def discover_videos(root: Path, output: Path) -> list[Path]:
    videos: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        # Prevent exports from becoming inputs when output lives inside input.
        if is_relative_to(path, output):
            continue
        videos.append(path)
    return sorted(videos, key=lambda p: str(p).lower())


def load_yamnet():
    print("Loading YAMNet (the first run may download the model)...")
    model = hub.load(MODEL_URL)

    class_map = model.class_map_path().numpy()
    if isinstance(class_map, bytes):
        class_map = class_map.decode("utf-8")

    names: list[str] = []
    with tf.io.gfile.GFile(class_map) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            names.append(row["display_name"])

    indices: dict[str, int] = {}
    for short_name, display_name in TARGET_CLASSES.items():
        try:
            indices[short_name] = names.index(display_name)
        except ValueError as exc:
            raise RuntimeError(f"YAMNet class not found: {display_name}") from exc

    print(
        "YAMNet ready: "
        + ", ".join(f"{TARGET_CLASSES[k]}=#{v}" for k, v in indices.items())
    )
    return model, indices


def run_command(command: list[str], *, capture_stderr: bool = True) -> None:
    result = subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_stderr else None,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or "").strip()
        raise RuntimeError(detail or f"Command failed with exit code {result.returncode}")


def extract_analysis_wav(
    ffmpeg: str, source: Path, destination: Path, audio_stream: int
) -> None:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        f"0:a:{audio_stream}",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(destination),
    ]
    run_command(command)


def analyze_wav(
    wav_path: Path,
    model,
    class_indices: dict[str, int],
    fart_threshold: float,
    burp_threshold: float,
    chunk_seconds: float,
) -> tuple[list[FrameDetection], float]:
    thresholds = {
        "fart": fart_threshold,
        "burp": burp_threshold,
    }
    detections: list[FrameDetection] = []

    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        total_frames = wav_file.getnframes()

        if channels != 1 or sample_width != 2 or sample_rate != SAMPLE_RATE:
            raise RuntimeError(
                "Unexpected WAV format after FFmpeg conversion: "
                f"channels={channels}, width={sample_width}, rate={sample_rate}"
            )

        duration = total_frames / SAMPLE_RATE
        chunk_frames = max(int(chunk_seconds * SAMPLE_RATE), SAMPLE_RATE)
        # Read one model window beyond each chunk boundary. The duplicate predictions
        # are intentionally merged later; this prevents missing a sound on a boundary.
        overlap_frames = int(MODEL_WINDOW_SECONDS * SAMPLE_RATE)

        start_frame = 0
        while start_frame < total_frames:
            wav_file.setpos(start_frame)
            frames_to_read = min(
                chunk_frames + overlap_frames,
                total_frames - start_frame,
            )
            raw = wav_file.readframes(frames_to_read)
            waveform = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0

            if waveform.size == 0:
                break

            scores, _, _ = model(tf.convert_to_tensor(waveform, dtype=tf.float32))
            scores_np = scores.numpy()
            base_time = start_frame / SAMPLE_RATE

            for short_name, class_index in class_indices.items():
                class_scores = scores_np[:, class_index]
                threshold = thresholds[short_name]
                matching = np.flatnonzero(class_scores >= threshold)

                for frame_index in matching:
                    # YAMNet's score is for a 0.96s patch; represent it by its center.
                    center = (
                        base_time
                        + float(frame_index) * MODEL_HOP_SECONDS
                        + MODEL_WINDOW_SECONDS / 2.0
                    )
                    center = min(center, duration)
                    detections.append(
                        FrameDetection(
                            label=short_name,
                            center=center,
                            score=float(class_scores[frame_index]),
                        )
                    )

            start_frame += chunk_frames

    return detections, duration


def merge_detections(
    detections: Iterable[FrameDetection], merge_gap: float
) -> list[Event]:
    events: list[Event] = []

    for label in TARGET_CLASSES:
        label_detections = sorted(
            (d for d in detections if d.label == label), key=lambda d: d.center
        )
        current: Event | None = None

        for detection in label_detections:
            patch_start = max(0.0, detection.center - MODEL_WINDOW_SECONDS / 2.0)
            patch_end = detection.center + MODEL_WINDOW_SECONDS / 2.0

            if current is None:
                current = Event(
                    label=label,
                    start=patch_start,
                    end=patch_end,
                    peak_time=detection.center,
                    confidence=detection.score,
                )
                continue

            if patch_start <= current.end + merge_gap:
                current.end = max(current.end, patch_end)
                if detection.score > current.confidence:
                    current.confidence = detection.score
                    current.peak_time = detection.center
            else:
                events.append(current)
                current = Event(
                    label=label,
                    start=patch_start,
                    end=patch_end,
                    peak_time=detection.center,
                    confidence=detection.score,
                )

        if current is not None:
            events.append(current)

    return sorted(events, key=lambda e: (e.start, e.label))


def safe_stem(value: str, max_length: int = 90) -> str:
    cleaned = "".join(c if c.isalnum() or c in "-_. " else "_" for c in value)
    cleaned = "_".join(cleaned.split())
    return (cleaned or "clip")[:max_length]


def short_path_hash(relative_path: str) -> str:
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:8]


def event_output_path(
    output_root: Path,
    source: Path,
    source_relative: str,
    event: Event,
    event_index: int,
) -> Path:
    label_dir = output_root / ("farts" if event.label == "fart" else "burps")
    label_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_stem(source.stem)
    digest = short_path_hash(source_relative)
    return label_dir / (
        f"{stem}__{digest}__{event.label}_{event_index:03d}"
        f"__{event.peak_time:.2f}s__{event.confidence:.2f}.mp4"
    )


def export_snippet(
    ffmpeg: str,
    source: Path,
    destination: Path,
    start: float,
    end: float,
    fast_cut: bool,
) -> None:
    duration = max(0.05, end - start)

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0?",
        "-map",
        "0:a:0?",
    ]

    if fast_cut:
        command += [
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
        ]
    else:
        command += [
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
        ]

    command.append(str(destination))
    run_command(command)


def file_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def detection_config(args: argparse.Namespace) -> dict[str, object]:
    # pre/post and cut mode do not affect detection, so changing them can safely reuse
    # the expensive audio-analysis cache.
    return {
        "model": MODEL_URL,
        "fart_threshold": args.fart_threshold,
        "burp_threshold": args.burp_threshold,
        "merge_gap": args.merge_gap,
        "audio_stream": args.audio_stream,
        "chunk_seconds": args.chunk_seconds,
    }


def load_cache(cache_path: Path, config: dict[str, object]) -> dict:
    empty = {
        "version": CACHE_VERSION,
        "detection_config": config,
        "files": {},
    }
    if not cache_path.exists():
        return empty

    try:
        with cache_path.open("r", encoding="utf-8") as handle:
            cache = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return empty

    if (
        cache.get("version") != CACHE_VERSION
        or cache.get("detection_config") != config
        or not isinstance(cache.get("files"), dict)
    ):
        return empty
    return cache


def save_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    validate_args(args)

    input_root = args.input.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    ffmpeg = ensure_ffmpeg()
    videos = discover_videos(input_root, output_root)
    if not videos:
        print(f"No supported video files found under: {input_root}")
        return 0

    print(f"Input : {input_root}")
    print(f"Output: {output_root}")
    print(f"Found {len(videos)} video(s).")
    print(
        f"Thresholds: fart={args.fart_threshold:.2f}, "
        f"burp={args.burp_threshold:.2f} | pre={args.pre:.1f}s, post={args.post:.1f}s"
    )

    config = detection_config(args)
    cache_path = output_root / ".tootscan_cache.json"
    cache = load_cache(cache_path, config)

    # Delay model loading until we actually have at least one uncached file.
    model = None
    class_indices = None

    manifest: list[ExportedEvent] = []
    total_farts = 0
    total_burps = 0
    failed_files: list[dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="tootscan-") as temp_dir_name:
        temp_dir = Path(temp_dir_name)

        for video_index, source in enumerate(videos, start=1):
            relative = source.relative_to(input_root).as_posix()
            fingerprint = file_fingerprint(source)
            cached_entry = cache["files"].get(relative)
            use_cache = (
                not args.force
                and isinstance(cached_entry, dict)
                and cached_entry.get("fingerprint") == fingerprint
                and isinstance(cached_entry.get("events"), list)
                and isinstance(cached_entry.get("duration"), (int, float))
            )

            print(f"\n[{video_index}/{len(videos)}] {relative}")

            try:
                if use_cache:
                    events = [Event(**item) for item in cached_entry["events"]]
                    duration = float(cached_entry["duration"])
                    print(f"  cache hit -> {len(events)} event(s)")
                else:
                    if model is None or class_indices is None:
                        model, class_indices = load_yamnet()

                    temp_wav = temp_dir / f"clip_{video_index:06d}.wav"
                    extract_analysis_wav(ffmpeg, source, temp_wav, args.audio_stream)
                    frame_detections, duration = analyze_wav(
                        temp_wav,
                        model,
                        class_indices,
                        args.fart_threshold,
                        args.burp_threshold,
                        args.chunk_seconds,
                    )
                    events = merge_detections(frame_detections, args.merge_gap)
                    temp_wav.unlink(missing_ok=True)

                    cache["files"][relative] = {
                        "fingerprint": fingerprint,
                        "duration": duration,
                        "events": [asdict(event) for event in events],
                    }
                    save_json_atomic(cache_path, cache)
                    print(f"  analyzed -> {len(events)} event(s)")

                per_label_index = {"fart": 0, "burp": 0}

                if not events:
                    print("  nothing detected")
                    continue

                for event in events:
                    per_label_index[event.label] += 1
                    if event.label == "fart":
                        total_farts += 1
                    else:
                        total_burps += 1

                    clip_start = max(0.0, event.start - args.pre)
                    clip_end = min(duration, event.end + args.post)
                    destination = event_output_path(
                        output_root,
                        source,
                        relative,
                        event,
                        per_label_index[event.label],
                    )

                    exported: str | None = None
                    if not args.dry_run:
                        if args.force or not destination.exists():
                            export_snippet(
                                ffmpeg,
                                source,
                                destination,
                                clip_start,
                                clip_end,
                                args.fast_cut,
                            )
                        exported = str(destination)

                    print(
                        f"  ✓ {event.label.upper():4s} @ {event.peak_time:8.2f}s "
                        f"score={event.confidence:.3f} "
                        f"[{clip_start:.2f}s -> {clip_end:.2f}s]"
                    )

                    manifest.append(
                        ExportedEvent(
                            source=str(source),
                            label=event.label,
                            event_start=event.start,
                            event_end=event.end,
                            peak_time=event.peak_time,
                            confidence=event.confidence,
                            clip_start=clip_start,
                            clip_end=clip_end,
                            output=exported,
                            cached=use_cache,
                        )
                    )

            except KeyboardInterrupt:
                print("\nInterrupted by user. Cache saved for completed files.")
                save_json_atomic(cache_path, cache)
                return 130
            except Exception as exc:  # keep scanning the rest of the library
                message = str(exc).strip() or exc.__class__.__name__
                print(f"  ERROR: {message}")
                failed_files.append({"source": str(source), "error": message})

    manifest_path = output_root / "detections.json"
    manifest_payload = {
        "input": str(input_root),
        "output": str(output_root),
        "model": MODEL_URL,
        "settings": {
            **config,
            "pre": args.pre,
            "post": args.post,
            "fast_cut": args.fast_cut,
            "dry_run": args.dry_run,
        },
        "summary": {
            "videos_scanned": len(videos),
            "farts": total_farts,
            "burps": total_burps,
            "events": len(manifest),
            "failures": len(failed_files),
        },
        "events": [asdict(event) for event in manifest],
        "failures": failed_files,
    }
    save_json_atomic(manifest_path, manifest_payload)
    save_json_atomic(cache_path, cache)

    print("\n" + "=" * 56)
    print(f"Videos scanned : {len(videos)}")
    print(f"Farts detected : {total_farts}")
    print(f"Burps detected : {total_burps}")
    print(f"Failures        : {len(failed_files)}")
    print(f"Manifest        : {manifest_path}")
    if not args.dry_run:
        print(f"Exports         : {output_root}")
    print("=" * 56)

    if failed_files:
        print("Some files failed; details are recorded in detections.json.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

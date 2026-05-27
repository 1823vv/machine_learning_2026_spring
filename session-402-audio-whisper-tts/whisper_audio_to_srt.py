import os
import glob
from pathlib import Path
import torch
import argparse
from faster_whisper import WhisperModel

import sys
import platform

# Fix encoding issues on Windows
if platform.system() == "Windows":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer)
    os.environ["PYTHONIOENCODING"] = "utf-8"

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid warnings
os.environ["USE_TORCH"] = "1"  # Force use of PyTorch
os.environ["USE_TF"] = "0"  # Disable TensorFlow

MODEL_MAP = {
    1: "tiny",        # ~75 MB, fastest, least accurate
    2: "base",        # ~142 MB
    3: "small",       # ~466 MB
    4: "medium",      # ~1.5 GB
    5: "large-v1",    # ~3 GB
    6: "large-v2",    # ~3 GB (improved v1)
    7: "large-v3",    # ~3 GB (most accurate, recommended)
}

def get_video_files(directory="./"):
    """Get all video files in the specified directory."""
    video_extensions = ["*.mp4", "*.mkv", "*.webm", "*.flv"]
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(directory, ext)))
    return video_files

def transcribe_video(video_path, model_size):
    """Transcribe video using Whisper model and return SRT content."""
    print(f"Transcribing {video_path} with model '{model_size}'...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if torch.cuda.is_available() else "int8"

    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    segments, _ = model.transcribe(
        video_path, language="en", task="transcribe", vad_filter=True
    )

    srt_content = []
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment.start)
        end = format_timestamp(segment.end)
        text = segment.text.strip()
        srt_content.append(f"{i}\n{start} --> {end}\n{text}\n\n")
    return srt_content

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"

def main(directory="./", model_choice=4, overwrite=False):
    model_size = MODEL_MAP.get(model_choice, "medium")

    video_files = get_video_files(directory)
    if not video_files:
        print(f"No video files found in {directory}.")
        return

    print(f"Found {len(video_files)} video file(s).")

    for video_path in video_files:
        video_filename = Path(video_path).stem
        srt_path = os.path.join(directory, f"{video_filename}.srt")

        if os.path.exists(srt_path) and not overwrite:
            print(f"Skipping {video_path} (SRT already exists). Use --overwrite-existing-srt-files to regenerate.")
            continue

        srt_content = transcribe_video(video_path, model_size)

        with open(srt_path, "w", encoding="utf-8") as f:
            f.writelines(srt_content)

        print(f"Subtitles saved to {srt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe videos to English SRT")
    parser.add_argument(
        "directory",
        nargs="?",
        default="./audio-or-video-files-to-transcribe",
        help="Directory containing video files (default: current directory)",
    )
    parser.add_argument(
        "--model",
        type=int,
        choices=[1, 2, 3, 4, 5, 6, 7],
        default=4,
        help="Model size: 1=tiny, 2=base, 3=small, 4=medium, 5=large-v1, 6=large-v2, 7=large-v3 (default: 4)",
    )
    parser.add_argument(
        "--overwrite-existing-srt-files",
        action="store_true",
        help="If set, existing .srt files will be overwritten",
    )
    args = parser.parse_args()

    main(args.directory, args.model, args.overwrite_existing_srt_files)

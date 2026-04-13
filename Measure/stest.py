import subprocess
import json
import os
from urllib.parse import urlparse


def is_url(path):
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https")


def get_video_metadata(path_or_url):
    """
    Works for both:
    - Local file: /path/to/video.mp4
    - URL: https://example.com/video.mp4
    """

    # Check local file exists
    if not is_url(path_or_url) and not os.path.exists(path_or_url):
        raise FileNotFoundError(f"File not found: {path_or_url}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path_or_url
    ]

    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print("Error running ffprobe:")
        print(result.stderr.decode())
        return

    data = json.loads(result.stdout)

    # Get video stream
    video_stream = next(
        (s for s in data["streams"] if s["codec_type"] == "video"),
        None
    )

    if not video_stream:
        print("No video stream found")
        return

    # Extract metadata
    width = video_stream.get("width")
    height = video_stream.get("height")

    fps_str = video_stream.get("r_frame_rate", "0/1")
    try:
        fps = eval(fps_str)
    except:
        fps = 0

    duration = float(data["format"].get("duration", 0))
    size_bytes = int(data["format"].get("size", 0))

    metadata = {
        "resolution": f"{width}x{height}",
        "width": width,
        "height": height,
        "fps": fps,
        "duration_sec": round(duration, 2),
        "size_mb": round(size_bytes / (1024 * 1024), 2),
        "format": data["format"].get("format_name"),
    }

    return metadata


# -------------------
# Example usage
# -------------------

if __name__ == "__main__":
    source = input("Enter video path or URL: ").strip()

    meta = get_video_metadata(source)

    if meta:
        print("\nVideo Metadata:")
        for k, v in meta.items():
            print(f"{k}: {v}")
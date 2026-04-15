"""
Parse app.log and produce a CSV summary with one row per task execution.

Usage:
    python parse_logs.py                          # uses Measure/logs/app.log
    python parse_logs.py path/to/app.log          # custom log path
    python parse_logs.py --out results.csv        # custom output path
    python parse_logs.py --since "2026-04-15 10:00"   # only logs after this time
    python parse_logs.py --since "10:00"              # today at 10:00
    python parse_logs.py --since "2026-04-15"          # from start of that day
"""

import re
import csv
import sys
import os
from datetime import datetime
from collections import defaultdict

LOG_FILE = os.path.join(os.path.dirname(__file__), "logs", "app.log")
OUT_FILE = os.path.join(os.path.dirname(__file__), "logs" ,"task_summary.csv")

TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S,%f"

# Matches:  INFO 2026-04-15 11:59:29,298 [ThreadPoolExecutor-1_0] task process_plank plank started: video=77 test=vmK617LE assessment=2ybjG2ND
# Also handles old format without [thread]
LINE_RE = re.compile(
    r"^(?P<level>\w+)\s+"
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+)\s+"
    r"(?:\[(?P<thread>[^\]]+)\]\s+)?"
    r"\S+\s+"                          # module
    r"\S+\s+"                          # funcName
    r"(?P<msg>.+)$"
)

STARTED_RE = re.compile(
    r"(?P<task_label>[\w_/]+)\s+started:\s+video=(?P<video_id>\d+)"
    r"(?:\s+test=(?P<test_id>\S+))?"
    r"(?:\s+assessment=(?P<assessment_id>\S+))?"
)

DONE_RE = re.compile(
    r"(?P<task_label>[\w_/]+)\s+done:\s+video=(?P<video_id>\d+)"
    r"(?:\s+distance=(?P<distance>[\d.]+)m)?"
    r"(?:\s+duration=(?P<duration>[\d.]+)s)?"
)

FAILED_RE = re.compile(
    r"(?P<task_label>[\w_/]+)\s+failed:\s+video=(?P<video_id>\d+)"
    r"(?:\s+[\u2014-]\s*(?P<error_msg>.+))?"
)

ERROR_RE = re.compile(
    r"(?P<task_label>[\w_/]+):\s+(?P<error_msg>.+?)\s*(?:video=(?P<video_id>\d+))?"
)


def parse_since(value):
    """Parse --since value into a datetime. Falls back to today 00:00 on bad/future input."""
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    now = datetime.now()

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%H:%M:%S",
        "%H:%M",
    ]

    dt = None
    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt)
            # Time-only formats: attach today's date
            if fmt in ("%H:%M:%S", "%H:%M"):
                parsed = parsed.replace(year=now.year, month=now.month, day=now.day)
            dt = parsed
            break
        except ValueError:
            continue

    if dt is None:
        print(f'[warning] Could not parse --since "{value}", using today\'s logs.')
        return today_start

    if dt > now:
        print(f'[warning] --since time {dt} is in the future, using today\'s logs.')
        return today_start

    return dt


def parse_ts(ts_str):
    return datetime.strptime(ts_str, TIMESTAMP_FMT)


def main():
    args = sys.argv[1:]
    log_path = LOG_FILE
    out_path = OUT_FILE
    since_dt = None

    i = 0
    while i < len(args):
        if args[i] == "--out" and i + 1 < len(args):
            out_path = args[i + 1]
            i += 2
        elif args[i] == "--since" and i + 1 < len(args):
            since_dt = parse_since(args[i + 1])
            i += 2
        else:
            log_path = args[i]
            i += 1

    if since_dt:
        print(f"Filtering logs from: {since_dt}")

    # key: (video_id, thread) -> dict of start info
    in_flight = {}
    rows = []

    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            m = LINE_RE.match(line)
            if not m:
                continue

            ts = parse_ts(m.group("ts"))
            if since_dt and ts < since_dt:
                continue
            thread = m.group("thread") or "unknown"
            msg = m.group("msg")

            sm = STARTED_RE.search(msg)
            if sm:
                key = (sm.group("video_id"), thread)
                in_flight[key] = {
                    "video_id": sm.group("video_id"),
                    "task_label": sm.group("task_label"),
                    "test_id": sm.group("test_id") or "",
                    "assessment_id": sm.group("assessment_id") or "",
                    "thread": thread,
                    "start_ts": ts,
                }
                continue

            dm = DONE_RE.search(msg)
            if dm:
                vid = dm.group("video_id")
                key = (vid, thread)
                entry = in_flight.pop(key, None)
                if entry:
                    elapsed = (ts - entry["start_ts"]).total_seconds()
                    rows.append({
                        "video_id": vid,
                        "task": entry["task_label"],
                        "test_id": entry["test_id"],
                        "assessment_id": entry["assessment_id"],
                        "thread": thread,
                        "started_at": entry["start_ts"].strftime(TIMESTAMP_FMT),
                        "finished_at": ts.strftime(TIMESTAMP_FMT),
                        "duration_s": round(elapsed, 3),
                        "result_distance_m": dm.group("distance") or "",
                        "result_duration_s": dm.group("duration") or "",
                        "status": "done",
                        "error_msg": "",
                    })
                continue

            fm = FAILED_RE.search(msg)
            if fm:
                vid = fm.group("video_id")
                key = (vid, thread)
                entry = in_flight.pop(key, None)
                error_msg = (fm.group("error_msg") or "").strip()
                if entry:
                    elapsed = (ts - entry["start_ts"]).total_seconds()
                    rows.append({
                        "video_id": vid,
                        "task": entry["task_label"],
                        "test_id": entry["test_id"],
                        "assessment_id": entry["assessment_id"],
                        "thread": thread,
                        "started_at": entry["start_ts"].strftime(TIMESTAMP_FMT),
                        "finished_at": ts.strftime(TIMESTAMP_FMT),
                        "duration_s": round(elapsed, 3),
                        "result_distance_m": "",
                        "result_duration_s": "",
                        "status": "failed",
                        "error_msg": error_msg,
                    })
                else:
                    # Failed with no matching started line
                    rows.append({
                        "video_id": vid or "",
                        "task": fm.group("task_label"),
                        "test_id": "",
                        "assessment_id": "",
                        "thread": thread,
                        "started_at": "",
                        "finished_at": ts.strftime(TIMESTAMP_FMT),
                        "duration_s": "",
                        "result_distance_m": "",
                        "result_duration_s": "",
                        "status": "failed",
                        "error_msg": error_msg,
                    })

    # Anything still in-flight at EOF = incomplete
    for entry in in_flight.values():
        rows.append({
            "video_id": entry["video_id"],
            "task": entry["task_label"],
            "test_id": entry["test_id"],
            "assessment_id": entry["assessment_id"],
            "thread": entry["thread"],
            "started_at": entry["start_ts"].strftime(TIMESTAMP_FMT),
            "finished_at": "",
            "duration_s": "",
            "result_distance_m": "",
            "result_duration_s": "",
            "status": "incomplete",
            "error_msg": "",
        })

    rows.sort(key=lambda r: r["started_at"])

    # Deduplicate: for the same video_id keep only the latest run
    latest = {}
    for row in rows:
        vid = row["video_id"]
        if vid not in latest or row["started_at"] > latest[vid]["started_at"]:
            latest[vid] = row
    rows = sorted(latest.values(), key=lambda r: r["started_at"])

    fieldnames = [
        "video_id", "task", "test_id", "assessment_id", "thread",
        "started_at", "finished_at", "duration_s",
        "result_distance_m", "result_duration_s", "status", "error_msg",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows -> {out_path}")

    # Print a quick summary to terminal
    done = [r for r in rows if r["status"] == "done"]
    failed = [r for r in rows if r["status"] == "failed"]
    incomplete = [r for r in rows if r["status"] == "incomplete"]
    durations = [r["duration_s"] for r in done if r["duration_s"] != ""]
    avg = sum(durations) / len(durations) if durations else 0

    print(f"  done={len(done)}  failed={len(failed)}  incomplete={len(incomplete)}")
    if durations:
        print(f"  avg duration: {avg:.1f}s  min: {min(durations):.1f}s  max: {max(durations):.1f}s")


if __name__ == "__main__":
    main()

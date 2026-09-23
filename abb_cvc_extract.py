#!/usr/bin/env python3
"""
Extract time-series data from Aussie Broadband CVC capacity graph images.

Outputs InfluxDB line protocol to stdout, or writes directly to InfluxDB.

Usage:
    # Extract a single POI
    python3 abb_cvc_extract.py --poi peakhurst --date 2026-03-24

    # Extract from a local image
    python3 abb_cvc_extract.py /tmp/peakhurst_cvc.png

    # Extract specific POIs and write to InfluxDB
    python3 abb_cvc_extract.py --poi peakhurst --poi peakhurstlink2 \\
        --yesterday --write-influxdb

    # Extract all discovered POIs and write to InfluxDB (CronJob mode)
    python3 abb_cvc_extract.py --discover --yesterday --write-influxdb

    # List all discovered POI slugs
    python3 abb_cvc_extract.py --discover-list

    # Output as CSV
    python3 abb_cvc_extract.py --poi peakhurst --date 2026-03-24 --format csv

    # Pipe to InfluxDB manually
    python3 abb_cvc_extract.py --poi peakhurst --date 2026-03-24 | curl -s \\
        "$INFLUXDB_URL/api/v2/write?org=$INFLUXDB_ORG&bucket=abb-cvc&precision=s" \\
        -H "Authorization: Token $INFLUXDB_TOKEN" \\
        --data-binary @-

Environment variables (for --write-influxdb):
    INFLUXDB_URL     InfluxDB base URL (e.g. https://influx.example.com)
    INFLUXDB_ORG     InfluxDB organisation ID
    INFLUXDB_BUCKET  InfluxDB bucket name
    INFLUXDB_TOKEN   InfluxDB API token
    LOG_LEVEL        Logging level (default: INFO)
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

ABB_CVC_URL = "https://cvcs.aussiebroadband.com.au/{poi}.png"
ABB_CVC_PAGE = "https://www.aussiebroadband.com.au/network/cvc-graphs/"
DEFAULT_CACHE_DIR = Path(os.environ.get("ABB_CVC_CACHE_DIR", ".cache/abb-cvc-extract"))
AEDT = timezone(timedelta(hours=11))
AEST = timezone(timedelta(hours=10))


# ---------------------------------------------------------------------------
# Structured JSON logging
# ---------------------------------------------------------------------------


class JSONFormatter(logging.Formatter):
    """Emit one JSON object per log line for Loki/Vector/Promtail ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        entry: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat().replace("+00:00", "Z"),
            "level": record.levelname.lower(),
            "msg": record.getMessage(),
        }
        # Merge any structured fields passed via `extra=`
        for key in (
            "poi",
            "date",
            "points",
            "bytes",
            "status",
            "scale_max",
            "capacity",
            "gridlines",
            "x_labels",
            "download",
            "upload",
            "interval",
            "total_pois",
            "total_points",
            "failed_pois",
            "missing_vars",
            "error",
        ):
            val = getattr(record, key, None)
            if val is not None:
                entry[key] = val
        if record.exc_info and record.exc_info[0] is not None:
            entry["error"] = "".join(traceback.format_exception(*record.exc_info)).rstrip()
        return json.dumps(entry, default=str)


def setup_logging() -> logging.Logger:
    """Configure structured JSON logging to stderr."""
    logger = logging.getLogger("abb_cvc_extract")
    if logger.handlers:
        return logger
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    return logger


log = setup_logging()


# ---------------------------------------------------------------------------
# POI discovery
# ---------------------------------------------------------------------------


def discover_pois() -> list[dict[str, str]]:
    """Discover all CVC POIs from the ABB website.

    Scrapes the CVC graphs page and extracts POI slugs and display names
    from the embedded Nuxt payload.

    Returns a list of dicts: [{"slug": "peakhurst", "name": "Peakhurst"}, ...]
    """
    req = urllib.request.Request(
        ABB_CVC_PAGE,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.aussiebroadband.com.au/",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")

    # The Nuxt SSR payload escapes URL slashes as \u002F, but older/static
    # captures may contain literal slashes. Normalize before matching so
    # discovery keeps working across either representation.
    html = html.replace(r"\u002F", "/")

    # The Nuxt SSR payload contains entries like:
    #   "https://cvcs.aussiebroadband.com.au/peakhurst.png","peakhurst","Peakhurst"
    pattern = re.compile(
        r'cvcs\.aussiebroadband\.com\.au/([a-z0-9]+)\.png","([a-z0-9]+)","([^"]+)"'
    )

    pois = []
    seen = set()
    for match in pattern.finditer(html):
        slug = match.group(2)
        name = match.group(3)
        if slug not in seen:
            seen.add(slug)
            pois.append({"slug": slug, "name": name})

    pois.sort(key=lambda p: p["slug"])
    return pois


# ---------------------------------------------------------------------------
# Image downloading
# ---------------------------------------------------------------------------


def _cache_paths(cache_dir: Path, poi: str) -> tuple[Path, Path]:
    """Return image and metadata cache paths for a POI."""
    safe_poi = poi.lower().replace("/", "_")
    return cache_dir / "images" / f"{safe_poi}.png", cache_dir / "images" / f"{safe_poi}.json"


def download_image(poi: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Download CVC graph image for a POI, reusing cached images when unchanged."""
    url = ABB_CVC_URL.format(poi=poi.lower())
    image_path, meta_path = _cache_paths(cache_dir, poi)
    image_path.parent.mkdir(parents=True, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": ABB_CVC_PAGE,
    }
    if meta_path.exists() and image_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("etag"):
                headers["If-None-Match"] = meta["etag"]
            if meta.get("last_modified"):
                headers["If-Modified-Since"] = meta["last_modified"]
        except (OSError, json.JSONDecodeError):
            pass

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            image_path.write_bytes(resp.read())
            meta_path.write_text(
                json.dumps(
                    {
                        "url": url,
                        "etag": resp.headers.get("ETag"),
                        "last_modified": resp.headers.get("Last-Modified"),
                        "fetched_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                    },
                    sort_keys=True,
                )
            )
            log.debug("Downloaded image", extra={"poi": poi, "status": resp.status})
    except urllib.error.HTTPError as e:
        if e.code == 304 and image_path.exists():
            log.debug("Using cached image", extra={"poi": poi})
            return image_path
        log.error("Failed to download image", extra={"poi": poi, "error": str(e)})
        sys.exit(1)
    return image_path


# ---------------------------------------------------------------------------
# Graph structure detection
# ---------------------------------------------------------------------------


def find_gridlines_y(rgb: np.ndarray) -> list[int]:
    """Find horizontal gridlines by scanning for uniform gray rows.

    ABB CVC graphs always have 5 gridlines at equal spacing. If only 4 are
    detected (common when the 0 Mbps gridline is obscured by the black
    download line), the 5th is inferred from the spacing of the other 4.
    """
    h, w = rgb.shape[:2]
    gridlines = []
    for y in range(40, h - 40):
        row = rgb[y, 80 : w - 40, :]
        gray = (
            (np.abs(row[:, 0].astype(int) - row[:, 1].astype(int)) < 12)
            & (np.abs(row[:, 1].astype(int) - row[:, 2].astype(int)) < 12)
            & (row[:, 0] > 185)
            & (row[:, 0] < 248)
        )
        if np.sum(gray) > (w * 0.7) and (not gridlines or y - gridlines[-1] > 10):
            gridlines.append(y)

    # Infer the missing 5th (bottom / 0 Mbps) gridline from even spacing
    if len(gridlines) == 4:
        spacings = [gridlines[i + 1] - gridlines[i] for i in range(3)]
        avg_spacing = round(sum(spacings) / len(spacings))
        inferred = gridlines[-1] + avg_spacing
        gridlines.append(inferred)
        log.debug(
            "Inferred 5th gridline", extra={"gridline_y": inferred, "spacing_px": avg_spacing}
        )

    return gridlines


def find_label_centers_x(rgb: np.ndarray) -> list[int]:
    """Find X-axis label center positions from text in bottom margin."""
    h = rgb.shape[0]
    bottom = rgb[h - 50 : h - 20, :, :]
    text_mask = (bottom[:, :, 0] < 100) & (bottom[:, :, 1] < 100) & (bottom[:, :, 2] < 100)
    text_cols = sorted(set(np.where(text_mask)[1]))

    clusters: list[list[int]] = []
    for x in text_cols:
        if not clusters or x - clusters[-1][-1] > 8:
            clusters.append([x])
        else:
            clusters[-1].append(x)

    return [(c[0] + c[-1]) // 2 for c in clusters]


# ---------------------------------------------------------------------------
# Line extraction
# ---------------------------------------------------------------------------


def _largest_cluster_median(matches: np.ndarray) -> int:
    """Return median row offset for the largest near-contiguous cluster."""
    split_points = np.flatnonzero(np.diff(matches) > 3) + 1
    clusters = np.split(matches, split_points)
    largest = max(clusters, key=len)
    return int(np.median(largest))


def extract_line(
    rgb: np.ndarray,
    x_left: int,
    x_right: int,
    color_mask_fn,
    y_search_top: int = 50,
    y_search_bot: int = 260,
) -> list[tuple[int, int]]:
    """Extract a colored line's y-coordinate at each x position.

    Returns list of (x, y) tuples. Uses column-wise scanning with
    cluster-based detection to find the line and reject stray pixels
    (e.g. title text, axis labels, or other artifacts).
    """
    region = rgb[y_search_top:y_search_bot, x_left : x_right + 1, :]
    mask = color_mask_fn(region)
    if not isinstance(mask, np.ndarray) or mask.shape != region.shape[:2]:
        # Backwards-compatible path for callers whose mask functions only
        # understand a single column shaped as ``(height, channels)``.
        mask = np.empty(region.shape[:2], dtype=bool)
        for x_offset in range(region.shape[1]):
            mask[:, x_offset] = color_mask_fn(region[:, x_offset, :])

    points = []
    for x_offset in range(mask.shape[1]):
        matches = np.flatnonzero(mask[:, x_offset])
        if len(matches) > 0:
            y = _largest_cluster_median(matches) + y_search_top
            points.append((x_left + x_offset, y))
    return points


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


def pixel_to_mbps(y: int, gridlines: list[int], mbps_values: list[float]) -> float:
    """Convert pixel y-coordinate to Mbps using gridline calibration."""
    if len(gridlines) < 2:
        return 0.0

    # Linear interpolation between gridlines
    for i in range(len(gridlines) - 1):
        if gridlines[i] <= y <= gridlines[i + 1]:
            frac = (y - gridlines[i]) / (gridlines[i + 1] - gridlines[i])
            return mbps_values[i] + frac * (mbps_values[i + 1] - mbps_values[i])

    # Extrapolate if outside gridline range
    if y < gridlines[0]:
        px_per_mbps = (gridlines[1] - gridlines[0]) / (mbps_values[1] - mbps_values[0])
        return mbps_values[0] + (y - gridlines[0]) / px_per_mbps
    else:
        px_per_mbps = (gridlines[-1] - gridlines[-2]) / (mbps_values[-1] - mbps_values[-2])
        return mbps_values[-1] + (y - gridlines[-1]) / px_per_mbps


def pixel_to_timestamp(x: int, x_labels: list[int], date: datetime) -> datetime:
    """Convert pixel x-coordinate to timestamp using label positions.

    x_labels correspond to 00:00, 02:00, 04:00, ..., 24:00 (13 labels).
    """
    if len(x_labels) < 2:
        return date

    hours_per_label = 2.0
    # Find which segment we're in
    for i in range(len(x_labels) - 1):
        if x_labels[i] <= x <= x_labels[i + 1]:
            frac = (x - x_labels[i]) / (x_labels[i + 1] - x_labels[i])
            hours = (i + frac) * hours_per_label
            return date + timedelta(hours=hours)

    # Extrapolate
    if x < x_labels[0]:
        frac = (x - x_labels[0]) / (x_labels[1] - x_labels[0])
        hours = frac * hours_per_label
    else:
        frac = (x - x_labels[-2]) / (x_labels[-1] - x_labels[-2])
        hours = ((len(x_labels) - 2) + frac) * hours_per_label

    return date + timedelta(hours=max(0, min(24, hours)))


# ---------------------------------------------------------------------------
# Scale detection
# ---------------------------------------------------------------------------


def detect_mbps_scale(gridlines: list[int], rgb: np.ndarray) -> list[float]:
    """Detect the Mbps values for each gridline from Y-axis label positions.

    ABB CVC graphs have 5 evenly-spaced gridlines. The scale is always a
    multiple of 2650 Mbps (i.e., max = N * 2650, gridline step = max / 4).
    Common scales: 2650, 5300, 10600, 15900, 21200.

    Detection strategy:
    1. Measure the pixel width of the top Y-axis label text.
    2. Compare to the second label to determine digit count (4 vs 5+).
    3. Use the blue capacity line position to narrow down the exact scale.
    """
    n = len(gridlines)
    if n < 4:
        log.warning("Insufficient gridlines, using default scale", extra={"gridlines": n})
        step = 10600 / 4
        return [10600 - i * step for i in range(n)]

    # --- Step 1: Measure Y-axis label widths to determine digit count ---
    left = rgb[:, 0:70, :]
    text_mask = (left[:, :, 0] < 100) & (left[:, :, 1] < 100) & (left[:, :, 2] < 100)

    rows = sorted(set(np.where(text_mask)[0]))
    row_clusters: list[list[int]] = []
    for y in rows:
        if not row_clusters or y - row_clusters[-1][-1] > 5:
            row_clusters.append([y])
        else:
            row_clusters[-1].append(y)

    label_widths = []
    for c in row_clusters:
        label_text = text_mask[c[0] : c[-1] + 1, :]
        cols_with_text = np.where(np.any(label_text, axis=0))[0]
        if len(cols_with_text) > 0:
            label_widths.append(int(cols_with_text[-1] - cols_with_text[0]))
        else:
            label_widths.append(0)

    # Determine digit count of each label from width.
    digit_counts: list[int] = []
    if len(label_widths) >= 4:
        median_4digit = float(np.median(sorted(label_widths[:4])[1:3]))
        for w_px in label_widths[:4]:
            digit_counts.append(5 if w_px > median_4digit * 1.05 else 4)

    # Use the digit pattern to determine the scale
    if digit_counts == [5, 4, 4, 4]:
        candidates = [10600]
    elif digit_counts == [5, 5, 4, 4]:
        candidates = [15900]
    elif digit_counts == [5, 5, 5, 4]:
        candidates = [21200]
    elif digit_counts == [5, 5, 5, 5]:
        candidates = [26500]
    elif all(d == 4 for d in digit_counts[:4]):
        candidates = [2650, 5300, 7950]
    else:
        candidates = [10600]  # safe default

    # --- Step 2: Use blue line position to determine exact scale ---
    blue_mask_arr = (rgb[:, :, 0] < 110) & (rgb[:, :, 1] > 140) & (rgb[:, :, 2] > 180)
    blue_ys = np.where(blue_mask_arr)[0]

    if len(blue_ys) > 0:
        blue_y = int(np.median(blue_ys))
        total_px = gridlines[-1] - gridlines[0]
        blue_frac = (blue_y - gridlines[0]) / total_px

        best = None
        best_err = float("inf")
        for max_mbps in candidates:
            blue_mbps = max_mbps * (1 - blue_frac)
            rounded = round(blue_mbps / 50) * 50
            err = abs(blue_mbps - rounded)
            if err < best_err:
                best = (max_mbps, rounded)
                best_err = err

        if best:
            max_mbps, capacity = best
            log.debug(
                "Detected scale",
                extra={"scale_max": max_mbps, "capacity": capacity},
            )
            return [max_mbps - i * (max_mbps / 4) for i in range(n)]

    # Fallback: use label width heuristic alone
    max_mbps = candidates[0]
    log.warning("Using fallback scale", extra={"scale_max": max_mbps})
    return [max_mbps - i * (max_mbps / 4) for i in range(n)]


# ---------------------------------------------------------------------------
# Graph extraction
# ---------------------------------------------------------------------------


def extract_graph(image_path: Path, poi: str, date_str: str | None = None) -> list[dict]:
    """Extract all time-series data from a CVC graph image."""
    img = cv2.imread(str(image_path))
    if img is None:
        log.error("Cannot read image", extra={"poi": poi, "error": str(image_path)})
        sys.exit(1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _h, w = img.shape[:2]

    # --- Calibration ---
    gridlines = find_gridlines_y(rgb)
    x_labels = find_label_centers_x(rgb)
    mbps_values = detect_mbps_scale(gridlines, rgb)

    log.debug(
        "Graph calibration",
        extra={
            "poi": poi,
            "gridlines": gridlines,
            "x_labels": len(x_labels),
            "scale_max": mbps_values[0],
        },
    )

    # Parse date
    if date_str:
        base_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=AEDT)
    else:
        base_date = datetime.now(AEDT).replace(hour=0, minute=0, second=0, microsecond=0)
        log.info(
            "No date specified, using today",
            extra={"poi": poi, "date": base_date.strftime("%Y-%m-%d")},
        )

    # --- Extract lines ---
    x_left = x_labels[0] if x_labels else 70
    x_right = x_labels[-1] if x_labels else w - 30

    # Color masks
    def black_mask(col):
        return (col[..., 0] < 55) & (col[..., 1] < 55) & (col[..., 2] < 55)

    def green_mask(col):
        return (
            (col[..., 1] > 100)
            & (col[..., 0] < 160)
            & (col[..., 2] < 100)
            & (col[..., 1] > col[..., 0])
        )

    def blue_mask(col):
        return (col[..., 0] < 110) & (col[..., 1] > 140) & (col[..., 2] > 180)

    # Constrain search to within gridline bounds (with small margin).
    y_top = gridlines[0] - 5
    y_bot = gridlines[-1] + 10

    black_points = extract_line(rgb, x_left, x_right, black_mask, y_top, y_bot)
    green_points = extract_line(rgb, x_left, x_right, green_mask, gridlines[-3], y_bot)
    blue_points = extract_line(rgb, x_left, x_right, blue_mask, y_top, y_bot)

    log.debug(
        "Line extraction complete",
        extra={
            "poi": poi,
            "download": len(black_points),
            "upload": len(green_points),
            "capacity": len(blue_points),
        },
    )

    # --- Convert to time-series ---
    results = []

    for x, y in black_points:
        ts = pixel_to_timestamp(x, x_labels, base_date)
        mbps = pixel_to_mbps(y, gridlines, mbps_values)
        results.append(
            {
                "ts": ts,
                "measurement": "abb_cvc",
                "tags": {"poi": poi, "metric": "download"},
                "value": max(0, mbps),
            }
        )

    for x, y in green_points:
        ts = pixel_to_timestamp(x, x_labels, base_date)
        mbps = pixel_to_mbps(y, gridlines, mbps_values)
        results.append(
            {
                "ts": ts,
                "measurement": "abb_cvc",
                "tags": {"poi": poi, "metric": "upload"},
                "value": max(0, mbps),
            }
        )

    for x, y in blue_points:
        ts = pixel_to_timestamp(x, x_labels, base_date)
        mbps = pixel_to_mbps(y, gridlines, mbps_values)
        results.append(
            {
                "ts": ts,
                "measurement": "abb_cvc",
                "tags": {"poi": poi, "metric": "capacity"},
                "value": max(0, mbps),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def downsample(points: list[dict], interval_seconds: int = 60) -> list[dict]:
    """Downsample points to one per interval by averaging."""
    if not points:
        return []

    buckets: dict[tuple[str, int], tuple[float, int, dict]] = {}
    for p in points:
        metric = p["tags"]["metric"]
        bucket_ts = int(p["ts"].timestamp()) // interval_seconds * interval_seconds
        key = (metric, bucket_ts)
        value_sum, count, representative = buckets.get(key, (0.0, 0, p))
        buckets[key] = (value_sum + p["value"], count + 1, representative)

    result = []
    for (_metric, bucket_ts), (value_sum, count, representative_point) in sorted(buckets.items()):
        representative = representative_point.copy()
        representative["ts"] = datetime.fromtimestamp(bucket_ts, tz=AEDT)
        representative["value"] = value_sum / count
        result.append(representative)

    return result


def to_line_protocol(points: list[dict]) -> str:
    """Convert extracted data to InfluxDB line protocol."""
    lines = []
    for p in sorted(points, key=lambda x: (x["tags"]["metric"], x["ts"])):
        tags = ",".join(f"{k}={v}" for k, v in sorted(p["tags"].items()))
        ts_unix = int(p["ts"].timestamp())
        lines.append(f"{p['measurement']},{tags} value={p['value']:.1f} {ts_unix}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# InfluxDB writer
# ---------------------------------------------------------------------------


def write_influxdb(points: list[dict]) -> None:
    """Write points to InfluxDB using the v2 write API.

    Reads connection details from environment variables:
        INFLUXDB_URL, INFLUXDB_ORG, INFLUXDB_BUCKET, INFLUXDB_TOKEN
    """
    required = ("INFLUXDB_URL", "INFLUXDB_ORG", "INFLUXDB_BUCKET", "INFLUXDB_TOKEN")
    missing = [k for k in required if not os.environ.get(k)]
    if missing:
        log.error("Missing InfluxDB environment variables", extra={"missing_vars": missing})
        sys.exit(1)

    url = os.environ["INFLUXDB_URL"]
    org = os.environ["INFLUXDB_ORG"]
    bucket = os.environ["INFLUXDB_BUCKET"]
    token = os.environ["INFLUXDB_TOKEN"]

    data = to_line_protocol(points).encode()
    if not data:
        log.info("No data to write")
        return

    write_url = f"{url}/api/v2/write?org={org}&bucket={bucket}&precision=s"
    req = urllib.request.Request(
        write_url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Token {token}",
            "Content-Type": "text/plain",
        },
    )
    try:
        resp = urllib.request.urlopen(req)
        log.info(
            "Wrote to InfluxDB",
            extra={"bytes": len(data), "points": len(points), "status": resp.status},
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        log.error(
            "InfluxDB write failed",
            extra={"status": e.code, "error": body},
        )
        sys.exit(1)


def yesterday_date() -> str:
    """Return yesterday's date in YYYY-MM-DD format (AEST)."""
    now_aest = datetime.now(AEDT)
    yesterday = now_aest - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")


def process_poi(
    slug: str,
    date: str | None,
    interval: int,
    cache_dir: Path,
    local_image: Path | None = None,
) -> list[dict]:
    """Download/cache, extract, and downsample a single POI."""
    image_path = local_image or download_image(slug, cache_dir)
    points = extract_graph(image_path, slug, date)
    return downsample(points, interval)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Extract time-series data from ABB CVC graph images"
    )
    parser.add_argument("image", nargs="?", help="Path to CVC graph image")
    parser.add_argument(
        "--poi",
        action="append",
        help="POI name to process (can be repeated). Downloads from ABB.",
    )
    parser.add_argument(
        "--all-links",
        action="store_true",
        help="Also download link2, link3, etc. variants for each --poi",
    )
    parser.add_argument("--date", help="Date for timestamps (YYYY-MM-DD)")
    parser.add_argument(
        "--yesterday",
        action="store_true",
        help="Use yesterday's date (AEST). Shorthand for CronJob usage.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Downsample interval in seconds (default: 300)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0,
        help="Delay in seconds between POI downloads (default: 0). "
        "Use with --discover to spread load, e.g. --delay 10.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of POIs to download/extract in parallel (default: 1).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Image cache directory (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--format",
        choices=["influx", "csv"],
        default="influx",
        help="Output format (default: influx line protocol)",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Discover and process all POIs from ABB website.",
    )
    parser.add_argument(
        "--discover-list",
        action="store_true",
        help="Print all discovered POI slugs (one per line) and exit.",
    )
    parser.add_argument(
        "--write-influxdb",
        action="store_true",
        help="Write results directly to InfluxDB (requires env vars).",
    )
    parser.add_argument(
        "--influxdb-batch-size",
        type=int,
        default=50_000,
        help="Maximum points per InfluxDB write batch (default: 50000).",
    )
    args = parser.parse_args()

    # Resolve date
    if args.yesterday:
        args.date = yesterday_date()
        log.info("Using yesterday's date", extra={"date": args.date})

    # --- Discovery: just list slugs ---
    if args.discover_list:
        pois = discover_pois()
        for p in pois:
            print(p["slug"])
        return

    # --- Build list of POI slugs to process ---
    poi_slugs: list[str] = []
    local_image: Path | None = None

    if args.image:
        local_image = Path(args.image)
        poi_slugs = [(args.poi[0] if args.poi else None) or local_image.stem]

    elif args.discover:
        pois = discover_pois()
        log.info("Discovered POIs", extra={"total_pois": len(pois)})
        if not pois:
            log.error("No POIs discovered")
            sys.exit(1)
        poi_slugs = [p["slug"] for p in pois]

    elif args.poi:
        for poi_name in args.poi:
            poi_slugs.append(poi_name)
            if args.all_links:
                for i in range(2, 10):
                    poi_slugs.append(f"{poi_name}link{i}")
    else:
        parser.error("provide an image path, --poi, --discover, or --discover-list")

    # --- Extract and output ---
    all_points: list[dict] = []
    pending_influx_points: list[dict] = []
    total_points = 0
    total_pois = 0
    failed_pois = 0
    workers = max(1, args.workers)

    def handle_points(slug: str, points: list[dict]) -> None:
        nonlocal total_points, total_pois, pending_influx_points
        total_points += len(points)
        total_pois += 1
        log.info(
            "Extracted POI",
            extra={"poi": slug, "points": len(points), "interval": args.interval},
        )

        if args.write_influxdb:
            pending_influx_points.extend(points)
            while len(pending_influx_points) >= args.influxdb_batch_size:
                batch = pending_influx_points[: args.influxdb_batch_size]
                del pending_influx_points[: args.influxdb_batch_size]
                write_influxdb(batch)
        else:
            all_points.extend(points)

    if workers == 1:
        for i, slug in enumerate(poi_slugs):
            # Rate-limit downloads to be polite to ABB servers.
            if i > 0 and args.delay > 0 and not local_image:
                time.sleep(args.delay)
            try:
                points = process_poi(slug, args.date, args.interval, args.cache_dir, local_image)
                handle_points(slug, points)
            except SystemExit:
                failed_pois += 1
            except Exception:
                log.exception("Extraction failed", extra={"poi": slug})
                failed_pois += 1
    else:
        if local_image and len(poi_slugs) > 1:
            log.warning("Parallel workers ignored for a single local image")
            workers = 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {}
            for i, slug in enumerate(poi_slugs):
                if i > 0 and args.delay > 0 and not local_image:
                    time.sleep(args.delay)
                future = executor.submit(
                    process_poi,
                    slug,
                    args.date,
                    args.interval,
                    args.cache_dir,
                    local_image,
                )
                futures[future] = slug

            for future in concurrent.futures.as_completed(futures):
                slug = futures[future]
                try:
                    handle_points(slug, future.result())
                except SystemExit:
                    failed_pois += 1
                except Exception:
                    log.exception("Extraction failed", extra={"poi": slug})
                    failed_pois += 1

    if args.write_influxdb and pending_influx_points:
        write_influxdb(pending_influx_points)

    # Flush accumulated points for stdout modes.
    if not args.write_influxdb:
        if args.format == "csv":
            _output_csv(all_points)
        else:
            print(to_line_protocol(all_points))

    log.info(
        "Run complete",
        extra={"total_pois": total_pois, "total_points": total_points, "failed_pois": failed_pois},
    )


def _output_csv(points: list[dict]) -> None:
    """Output points as CSV to stdout."""
    print("timestamp,poi,metric,value_mbps")
    for p in sorted(points, key=lambda x: (x["tags"]["poi"], x["tags"]["metric"], x["ts"])):
        print(f"{p['ts'].isoformat()},{p['tags']['poi']},{p['tags']['metric']},{p['value']:.1f}")


if __name__ == "__main__":
    main()

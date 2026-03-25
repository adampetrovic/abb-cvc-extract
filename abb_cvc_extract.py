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
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

ABB_CVC_URL = "https://cvcs.aussiebroadband.com.au/{poi}.png"
ABB_CVC_PAGE = "https://www.aussiebroadband.com.au/network/cvc-graphs/"
AEDT = timezone(timedelta(hours=11))
AEST = timezone(timedelta(hours=10))


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


def download_image(poi: str) -> Path:
    """Download CVC graph image for a POI."""
    url = ABB_CVC_URL.format(poi=poi.lower())
    tmp = Path(tempfile.mktemp(suffix=".png"))
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": ABB_CVC_PAGE,
        },
    )
    try:
        with urllib.request.urlopen(req) as resp:
            tmp.write_bytes(resp.read())
    except urllib.error.HTTPError as e:
        print(f"error: failed to download {url}: {e}", file=sys.stderr)
        sys.exit(1)
    return tmp


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
        print(
            f"info: inferred 5th gridline at y={inferred} (spacing={avg_spacing}px)",
            file=sys.stderr,
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
    points = []
    for x in range(x_left, x_right + 1):
        col = rgb[y_search_top:y_search_bot, x, :]
        mask = color_mask_fn(col)
        matches = np.where(mask)[0]
        if len(matches) > 0:
            # Find the largest contiguous cluster of matching pixels.
            # This rejects stray pixels from title text or other artifacts
            # that would otherwise skew the median.
            clusters: list[list[int]] = []
            current = [matches[0]]
            for i in range(1, len(matches)):
                if matches[i] - matches[i - 1] <= 3:  # allow small gaps (anti-aliasing)
                    current.append(matches[i])
                else:
                    clusters.append(current)
                    current = [matches[i]]
            clusters.append(current)

            # Use the largest cluster (the actual line)
            largest = max(clusters, key=len)
            y = int(np.median(largest)) + y_search_top
            points.append((x, y))
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
        print(
            f"warning: expected 5 gridlines, found {n}. Using default scale.",
            file=sys.stderr,
        )
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

    print(
        f"info: label digit pattern={digit_counts}, candidates={candidates}",
        file=sys.stderr,
    )

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
            print(
                f"info: detected scale max={max_mbps} Mbps, "
                f"CVC capacity={capacity} Mbps (blue y={blue_y}, "
                f"digits={digit_counts}, "
                f"widths={label_widths[:3]})",
                file=sys.stderr,
            )
            return [max_mbps - i * (max_mbps / 4) for i in range(n)]

    # Fallback: use label width heuristic alone
    max_mbps = candidates[0]
    print(f"warning: using fallback scale max={max_mbps} Mbps", file=sys.stderr)
    return [max_mbps - i * (max_mbps / 4) for i in range(n)]


# ---------------------------------------------------------------------------
# Graph extraction
# ---------------------------------------------------------------------------


def extract_graph(image_path: Path, poi: str, date_str: str | None = None) -> list[dict]:
    """Extract all time-series data from a CVC graph image."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"error: cannot read image {image_path}", file=sys.stderr)
        sys.exit(1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _h, w = img.shape[:2]

    # --- Calibration ---
    gridlines = find_gridlines_y(rgb)
    x_labels = find_label_centers_x(rgb)
    mbps_values = detect_mbps_scale(gridlines, rgb)

    print(
        f"info: gridlines={gridlines}, x_labels={len(x_labels)}, "
        f"scale={mbps_values[0]:.0f}-{mbps_values[-1]:.0f} Mbps",
        file=sys.stderr,
    )

    # Parse date
    if date_str:
        base_date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=AEDT)
    else:
        base_date = datetime.now(AEDT).replace(hour=0, minute=0, second=0, microsecond=0)
        print(
            f"info: no date specified, using {base_date.strftime('%Y-%m-%d')}",
            file=sys.stderr,
        )

    # --- Extract lines ---
    x_left = x_labels[0] if x_labels else 70
    x_right = x_labels[-1] if x_labels else w - 30

    # Color masks
    def black_mask(col):
        return (col[:, 0] < 55) & (col[:, 1] < 55) & (col[:, 2] < 55)

    def green_mask(col):
        return (col[:, 1] > 100) & (col[:, 0] < 160) & (col[:, 2] < 100) & (col[:, 1] > col[:, 0])

    def blue_mask(col):
        return (col[:, 0] < 110) & (col[:, 1] > 140) & (col[:, 2] > 180)

    # Constrain search to within gridline bounds (with small margin).
    # Using gridlines[0] avoids title text and other artifacts above the plot area.
    y_top = gridlines[0] - 5
    y_bot = gridlines[-1] + 10

    black_points = extract_line(rgb, x_left, x_right, black_mask, y_top, y_bot)
    green_points = extract_line(rgb, x_left, x_right, green_mask, gridlines[-3], y_bot)
    blue_points = extract_line(rgb, x_left, x_right, blue_mask, y_top, y_bot)

    print(
        f"info: extracted {len(black_points)} download, "
        f"{len(green_points)} upload, {len(blue_points)} capacity points",
        file=sys.stderr,
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

    buckets: dict[tuple[str, int], list[dict]] = {}
    for p in points:
        metric = p["tags"]["metric"]
        bucket_ts = int(p["ts"].timestamp()) // interval_seconds * interval_seconds
        key = (metric, bucket_ts)
        buckets.setdefault(key, []).append(p)

    result = []
    for (_metric, bucket_ts), group in sorted(buckets.items()):
        avg_val = float(np.mean([p["value"] for p in group]))
        representative = group[0].copy()
        representative["ts"] = datetime.fromtimestamp(bucket_ts, tz=AEDT)
        representative["value"] = avg_val
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
        print(
            f"error: missing environment variables for --write-influxdb: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    url = os.environ["INFLUXDB_URL"]
    org = os.environ["INFLUXDB_ORG"]
    bucket = os.environ["INFLUXDB_BUCKET"]
    token = os.environ["INFLUXDB_TOKEN"]

    data = to_line_protocol(points).encode()
    if not data:
        print("info: no data to write", file=sys.stderr)
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
        print(
            f"info: wrote {len(data)} bytes ({len(points)} points) to InfluxDB, HTTP {resp.status}",
            file=sys.stderr,
        )
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        print(
            f"error: InfluxDB write failed: HTTP {e.code} {e.reason}\n{body}",
            file=sys.stderr,
        )
        sys.exit(1)


def yesterday_date() -> str:
    """Return yesterday's date in YYYY-MM-DD format (AEST)."""
    now_aest = datetime.now(AEDT)
    yesterday = now_aest - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")


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
    args = parser.parse_args()

    # Resolve date
    if args.yesterday:
        args.date = yesterday_date()
        print(f"info: using yesterday's date: {args.date}", file=sys.stderr)

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
        print(f"info: discovered {len(pois)} POIs", file=sys.stderr)
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
    # When writing to InfluxDB, flush per-POI to keep memory bounded
    # and avoid a single massive write (540 POIs x ~2500 points = ~1.4M points).
    # For stdout modes, accumulate for a single output.
    all_points: list[dict] = []
    total_points = 0
    total_pois = 0
    failed_pois = 0

    for i, slug in enumerate(poi_slugs):
        # Rate-limit downloads to be polite to ABB servers
        if i > 0 and args.delay > 0 and not local_image:
            time.sleep(args.delay)

        # Download or use local image
        if local_image:
            image_path = local_image
        else:
            try:
                image_path = download_image(slug)
            except SystemExit:
                failed_pois += 1
                continue

        try:
            points = extract_graph(image_path, slug, args.date)
            points = downsample(points, args.interval)
            total_points += len(points)
            total_pois += 1
            print(
                f"info: {slug}: {len(points)} points after {args.interval}s downsample",
                file=sys.stderr,
            )

            if args.write_influxdb and points:
                write_influxdb(points)
            else:
                all_points.extend(points)
        except Exception as e:
            print(f"warning: {slug}: extraction failed: {e}", file=sys.stderr)
            failed_pois += 1
        finally:
            if not local_image:
                image_path.unlink(missing_ok=True)

    # Flush accumulated points for stdout modes
    if not args.write_influxdb:
        if args.format == "csv":
            _output_csv(all_points)
        else:
            print(to_line_protocol(all_points))

    print(
        f"info: done — {total_pois} POIs, {total_points} points"
        + (f", {failed_pois} failed" if failed_pois else ""),
        file=sys.stderr,
    )


def _output_csv(points: list[dict]) -> None:
    """Output points as CSV to stdout."""
    print("timestamp,poi,metric,value_mbps")
    for p in sorted(points, key=lambda x: (x["tags"]["poi"], x["tags"]["metric"], x["ts"])):
        print(f"{p['ts'].isoformat()},{p['tags']['poi']},{p['tags']['metric']},{p['value']:.1f}")


if __name__ == "__main__":
    main()

# abb-cvc-extract

Extract time-series data from [Aussie Broadband](https://www.aussiebroadband.com.au/) CVC capacity graph images using computer vision (OpenCV + NumPy).

ABB publishes CVC utilisation graphs as PNG images at `https://cvcs.aussiebroadband.com.au/{poi}.png` — there is no API or raw data available. This tool extracts the download (black), upload (green), and capacity (blue) lines from these graphs and outputs structured data as InfluxDB line protocol or CSV.

## Usage

### Docker (recommended)

```bash
# Extract a POI and output InfluxDB line protocol
docker run --rm ghcr.io/adampetrovic/abb-cvc-extract --poi peakhurst --date 2026-03-24

# Output as CSV
docker run --rm ghcr.io/adampetrovic/abb-cvc-extract --poi peakhurst --date 2026-03-24 --format csv

# Extract from a local image
docker run --rm -v /tmp:/tmp ghcr.io/adampetrovic/abb-cvc-extract /tmp/peakhurst.png --poi peakhurst

# Write directly to InfluxDB
docker run --rm ghcr.io/adampetrovic/abb-cvc-extract --poi peakhurst --date 2026-03-24 | \
  curl -s "$INFLUXDB_URL/api/v2/write?org=$ORG&bucket=abb-cvc&precision=s" \
    -H "Authorization: Token $TOKEN" \
    --data-binary @-
```

### Python

```bash
pip install opencv-python-headless numpy
python3 abb_cvc_extract.py --poi peakhurst --date 2026-03-24
```

### Options

| Flag | Description |
|---|---|
| `image` | Path to a local CVC graph PNG (positional) |
| `--poi NAME` | POI name — downloads from ABB if no local image given |
| `--all-links` | Also download link2, link3, … variants |
| `--date YYYY-MM-DD` | Date for timestamps (default: today AEST) |
| `--interval SECS` | Downsample interval in seconds (default: 60) |
| `--format influx\|csv` | Output format (default: influx line protocol) |

## How it works

1. **Download** the CVC graph PNG for a given POI
2. **Detect gridlines** — horizontal gray rows in the plot area
3. **Detect X-axis labels** — text clusters in the bottom margin (00:00–24:00)
4. **Detect Y-axis scale** — label pixel widths determine digit count → known ABB scale (2650, 5300, 10600, 15900 Mbps)
5. **Extract lines** — column-wise scanning with colour masks and cluster-based detection to reject artifacts
6. **Calibrate** — convert pixel coordinates to timestamps and Mbps values
7. **Downsample** — average to 60-second intervals (~860 points per line)
8. **Output** — InfluxDB line protocol or CSV

## Container images

Published to GHCR on every release:

```
ghcr.io/adampetrovic/abb-cvc-extract:latest
ghcr.io/adampetrovic/abb-cvc-extract:1
ghcr.io/adampetrovic/abb-cvc-extract:1.0
ghcr.io/adampetrovic/abb-cvc-extract:1.0.0
```

## License

MIT

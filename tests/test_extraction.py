"""Tests for CVC graph extraction against real fixture images.

Each fixture was captured on 2026-03-24 from ABB's CVC graph page.
The assertions encode the known-correct extraction values so that any
future refactoring can be validated against these baselines.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from abb_cvc_extract import (
    detect_mbps_scale,
    discover_pois,
    downsample,
    extract_graph,
    extract_line,
    find_gridlines_y,
    find_label_centers_x,
    main,
    pixel_to_mbps,
    pixel_to_timestamp,
    to_line_protocol,
    write_influxdb,
    yesterday_date,
)

AEDT = timezone(timedelta(hours=11))
DATE = "2026-03-24"


def _load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    assert img is not None, f"Failed to load image: {path}"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Gridline detection
# ---------------------------------------------------------------------------


class TestFindGridlines:
    """Test gridline detection across fixture images."""

    def test_peakhurst_5_gridlines(self, peakhurst_image):
        rgb = _load_rgb(peakhurst_image)
        gridlines = find_gridlines_y(rgb)
        assert len(gridlines) == 5
        assert gridlines == [90, 127, 165, 202, 240]

    def test_keysborough_5_gridlines(self, keysborough_image):
        rgb = _load_rgb(keysborough_image)
        gridlines = find_gridlines_y(rgb)
        assert len(gridlines) == 5
        assert gridlines == [90, 127, 165, 202, 240]

    def test_katanning_infers_5th_gridline(self, katanning_image):
        """Katanning has near-zero traffic — the bottom gridline is obscured
        by the black download line. The function should infer it."""
        rgb = _load_rgb(katanning_image)
        gridlines = find_gridlines_y(rgb)
        assert len(gridlines) == 5
        # Inferred from 37px spacing: 202 + 37 = 239
        assert gridlines[:4] == [90, 127, 165, 202]
        assert gridlines[4] in (239, 240)  # allow ±1px rounding

    def test_campbelltownlink9_infers_5th_gridline(self, campbelltownlink9_image):
        """Empty graph — all labels show 0 Mbps. Still infers 5 gridlines."""
        rgb = _load_rgb(campbelltownlink9_image)
        gridlines = find_gridlines_y(rgb)
        assert len(gridlines) == 5

    def test_gridline_even_spacing(self, peakhurst_image):
        """Gridlines should be approximately evenly spaced."""
        rgb = _load_rgb(peakhurst_image)
        gridlines = find_gridlines_y(rgb)
        spacings = [gridlines[i + 1] - gridlines[i] for i in range(len(gridlines) - 1)]
        assert max(spacings) - min(spacings) <= 2


# ---------------------------------------------------------------------------
# X-axis label detection
# ---------------------------------------------------------------------------


class TestFindLabelsX:
    """Test X-axis hour label detection."""

    def test_peakhurst_13_labels(self, peakhurst_image):
        """ABB graphs have 13 x-axis labels (00:00 to 24:00, every 2 hours)."""
        rgb = _load_rgb(peakhurst_image)
        labels = find_label_centers_x(rgb)
        assert len(labels) == 13

    @pytest.mark.parametrize(
        "fixture",
        ["peakhurst", "woolloongabba", "keysborough", "katanning", "campbelltownlink9"],
    )
    def test_all_fixtures_13_labels(self, fixtures_dir, fixture):
        rgb = _load_rgb(fixtures_dir / f"{fixture}.png")
        labels = find_label_centers_x(rgb)
        assert len(labels) == 13

    def test_labels_span_plot_area(self, peakhurst_image):
        """First label should be near x=70, last near x=930 for 960px images."""
        rgb = _load_rgb(peakhurst_image)
        labels = find_label_centers_x(rgb)
        assert 50 < labels[0] < 90
        assert 910 < labels[-1] < 945


# ---------------------------------------------------------------------------
# Scale detection
# ---------------------------------------------------------------------------


class TestDetectScale:
    """Test Y-axis scale detection."""

    @pytest.mark.parametrize(
        "fixture",
        ["peakhurst", "woolloongabba", "keysborough"],
    )
    def test_10600_scale(self, fixtures_dir, fixture):
        rgb = _load_rgb(fixtures_dir / f"{fixture}.png")
        gridlines = find_gridlines_y(rgb)
        scale = detect_mbps_scale(gridlines, rgb)
        assert scale[0] == 10600
        assert scale[-1] == 0

    def test_katanning_10600_scale(self, katanning_image):
        """Katanning has 10600 scale despite low traffic."""
        rgb = _load_rgb(katanning_image)
        gridlines = find_gridlines_y(rgb)
        scale = detect_mbps_scale(gridlines, rgb)
        assert scale[0] == 10600
        assert scale[-1] == 0

    def test_scale_values_evenly_spaced(self, peakhurst_image):
        rgb = _load_rgb(peakhurst_image)
        gridlines = find_gridlines_y(rgb)
        scale = detect_mbps_scale(gridlines, rgb)
        step = scale[0] - scale[1]
        for i in range(len(scale) - 1):
            assert scale[i] - scale[i + 1] == pytest.approx(step)


# ---------------------------------------------------------------------------
# Line extraction
# ---------------------------------------------------------------------------


class TestExtractLine:
    """Test colored line extraction from graph images."""

    def test_peakhurst_download_points(self, peakhurst_image):
        rgb = _load_rgb(peakhurst_image)
        gridlines = find_gridlines_y(rgb)
        x_labels = find_label_centers_x(rgb)

        def black_mask(col):
            return (col[:, 0] < 55) & (col[:, 1] < 55) & (col[:, 2] < 55)

        points = extract_line(
            rgb,
            x_labels[0],
            x_labels[-1],
            black_mask,
            gridlines[0] - 5,
            gridlines[-1] + 10,
        )
        assert len(points) == 859

    def test_peakhurst_upload_points(self, peakhurst_image):
        rgb = _load_rgb(peakhurst_image)
        gridlines = find_gridlines_y(rgb)
        x_labels = find_label_centers_x(rgb)

        def green_mask(col):
            return (
                (col[:, 1] > 100) & (col[:, 0] < 160) & (col[:, 2] < 100) & (col[:, 1] > col[:, 0])
            )

        points = extract_line(
            rgb,
            x_labels[0],
            x_labels[-1],
            green_mask,
            gridlines[-3],
            gridlines[-1] + 10,
        )
        assert len(points) == 859

    def test_katanning_no_upload(self, katanning_image):
        """Katanning has near-zero traffic — upload line may be undetectable."""
        rgb = _load_rgb(katanning_image)
        gridlines = find_gridlines_y(rgb)
        x_labels = find_label_centers_x(rgb)

        def green_mask(col):
            return (
                (col[:, 1] > 100) & (col[:, 0] < 160) & (col[:, 2] < 100) & (col[:, 1] > col[:, 0])
            )

        points = extract_line(
            rgb,
            x_labels[0],
            x_labels[-1],
            green_mask,
            gridlines[-3],
            gridlines[-1] + 10,
        )
        assert len(points) == 0


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------


class TestExtractGraph:
    """End-to-end extraction tests with known-good values."""

    def test_peakhurst_full(self, peakhurst_image):
        results = extract_graph(peakhurst_image, "peakhurst", DATE)
        dl = [r for r in results if r["tags"]["metric"] == "download"]
        ul = [r for r in results if r["tags"]["metric"] == "upload"]
        cap = [r for r in results if r["tags"]["metric"] == "capacity"]

        assert len(dl) == 859
        assert len(ul) == 859
        assert len(cap) == 859

        dl_values = [r["value"] for r in dl]
        assert max(dl_values) == pytest.approx(7531.6, abs=100)
        assert min(dl_values) == pytest.approx(1464.5, abs=200)

        # Capacity should be ~10027 Mbps (constant)
        cap_values = set(round(r["value"], 0) for r in cap)
        assert len(cap_values) <= 3  # near-constant
        assert max(r["value"] for r in cap) == pytest.approx(10027, abs=200)

    def test_woolloongabba_full(self, woolloongabba_image):
        results = extract_graph(woolloongabba_image, "woolloongabba", DATE)
        dl = [r for r in results if r["tags"]["metric"] == "download"]

        assert len(dl) == 859
        dl_values = [r["value"] for r in dl]
        assert max(dl_values) == pytest.approx(5370, abs=200)
        assert 1500 < sum(dl_values) / len(dl_values) < 3000  # mean ~2205

    def test_keysborough_full(self, keysborough_image):
        results = extract_graph(keysborough_image, "keysborough", DATE)
        dl = [r for r in results if r["tags"]["metric"] == "download"]

        assert len(dl) == 859
        dl_values = [r["value"] for r in dl]
        assert max(dl_values) == pytest.approx(6137, abs=200)
        assert 1800 < sum(dl_values) / len(dl_values) < 3300

    def test_katanning_near_zero_traffic(self, katanning_image):
        results = extract_graph(katanning_image, "katanning", DATE)
        dl = [r for r in results if r["tags"]["metric"] == "download"]
        cap = [r for r in results if r["tags"]["metric"] == "capacity"]

        # Download should be extracted (at or near 0)
        assert len(dl) == 859
        assert all(r["value"] < 200 for r in dl)

        # Capacity should still be detected
        assert len(cap) == 859
        assert max(r["value"] for r in cap) == pytest.approx(10027, abs=200)

    def test_campbelltownlink9_empty_graph(self, campbelltownlink9_image):
        """Completely empty graph — zero on all axes."""
        results = extract_graph(campbelltownlink9_image, "campbelltownlink9", DATE)
        dl = [r for r in results if r["tags"]["metric"] == "download"]
        ul = [r for r in results if r["tags"]["metric"] == "upload"]

        # No meaningful download or upload data
        assert len(dl) == 0
        assert len(ul) == 0

    def test_no_spikes_above_scale(self, peakhurst_image):
        """Regression test: no download values should exceed the scale max."""
        results = extract_graph(peakhurst_image, "peakhurst", DATE)
        dl = [r for r in results if r["tags"]["metric"] == "download"]
        assert all(r["value"] <= 10600 for r in dl)

    def test_no_negative_values(self, peakhurst_image):
        results = extract_graph(peakhurst_image, "peakhurst", DATE)
        assert all(r["value"] >= 0 for r in results)

    def test_timestamps_span_24h(self, peakhurst_image):
        """Extracted timestamps should span approximately 00:00-24:00."""
        results = extract_graph(peakhurst_image, "peakhurst", DATE)
        dl = [r for r in results if r["tags"]["metric"] == "download"]
        ts_sorted = sorted(r["ts"] for r in dl)
        span_hours = (ts_sorted[-1] - ts_sorted[0]).total_seconds() / 3600
        assert 22 < span_hours <= 24

    def test_tags_correct(self, peakhurst_image):
        results = extract_graph(peakhurst_image, "peakhurst", DATE)
        for r in results:
            assert r["measurement"] == "abb_cvc"
            assert r["tags"]["poi"] == "peakhurst"
            assert r["tags"]["metric"] in ("download", "upload", "capacity")


# ---------------------------------------------------------------------------
# Downsample
# ---------------------------------------------------------------------------


class TestDownsample:
    def test_reduces_point_count(self, peakhurst_image):
        results = extract_graph(peakhurst_image, "peakhurst", DATE)
        dl = [r for r in results if r["tags"]["metric"] == "download"]
        downsampled = downsample(dl, 120)  # 2-minute buckets
        assert len(downsampled) < len(dl)
        assert len(downsampled) > 0

    def test_preserves_metric_separation(self, peakhurst_image):
        results = extract_graph(peakhurst_image, "peakhurst", DATE)
        downsampled = downsample(results, 60)
        metrics = {r["tags"]["metric"] for r in downsampled}
        assert metrics == {"download", "upload", "capacity"}


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


class TestLineProtocol:
    def test_format(self, peakhurst_image):
        results = extract_graph(peakhurst_image, "peakhurst", DATE)
        output = to_line_protocol(results[:3])
        lines = output.strip().split("\n")
        for line in lines:
            # InfluxDB line protocol: measurement,tags field=value timestamp
            assert line.startswith("abb_cvc,")
            assert "value=" in line
            parts = line.split(" ")
            assert len(parts) == 3
            # Timestamp should be unix seconds
            assert parts[2].isdigit()


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------


class TestPixelToMbps:
    def test_at_gridline(self):
        gridlines = [90, 127, 165, 202, 240]
        mbps_values = [10600, 7950, 5300, 2650, 0]
        assert pixel_to_mbps(90, gridlines, mbps_values) == pytest.approx(10600)
        assert pixel_to_mbps(240, gridlines, mbps_values) == pytest.approx(0)
        assert pixel_to_mbps(165, gridlines, mbps_values) == pytest.approx(5300)

    def test_between_gridlines(self):
        gridlines = [90, 127, 165, 202, 240]
        mbps_values = [10600, 7950, 5300, 2650, 0]
        mid = (90 + 127) // 2  # ~108
        result = pixel_to_mbps(mid, gridlines, mbps_values)
        assert 8000 < result < 10600


class TestPixelToTimestamp:
    def test_at_label(self):
        labels = list(range(70, 931, 72))  # 13 labels, ~72px apart
        base = datetime(2026, 3, 24, tzinfo=AEDT)
        ts = pixel_to_timestamp(labels[6], labels, base)
        expected = base + timedelta(hours=12)
        assert ts == expected

    def test_between_labels(self):
        labels = list(range(70, 931, 72))
        base = datetime(2026, 3, 24, tzinfo=AEDT)
        mid = (labels[0] + labels[1]) // 2
        ts = pixel_to_timestamp(mid, labels, base)
        assert base < ts < base + timedelta(hours=2)


# ---------------------------------------------------------------------------
# Yesterday date helper
# ---------------------------------------------------------------------------


class TestYesterdayDate:
    def test_returns_valid_date(self):
        result = yesterday_date()
        # Should be a valid YYYY-MM-DD string
        parsed = datetime.strptime(result, "%Y-%m-%d")
        assert parsed is not None

    def test_is_before_today(self):
        result = yesterday_date()
        parsed = datetime.strptime(result, "%Y-%m-%d").replace(tzinfo=AEDT)
        now = datetime.now(AEDT).replace(hour=0, minute=0, second=0, microsecond=0)
        assert parsed < now


# ---------------------------------------------------------------------------
# InfluxDB writer
# ---------------------------------------------------------------------------


class TestWriteInfluxdb:
    def test_missing_env_vars(self, peakhurst_image, monkeypatch):
        """Should exit with error when env vars are missing."""
        monkeypatch.delenv("INFLUXDB_URL", raising=False)
        monkeypatch.delenv("INFLUXDB_ORG", raising=False)
        monkeypatch.delenv("INFLUXDB_BUCKET", raising=False)
        monkeypatch.delenv("INFLUXDB_TOKEN", raising=False)

        results = extract_graph(peakhurst_image, "peakhurst", DATE)
        with pytest.raises(SystemExit):
            write_influxdb(results)

    def test_empty_points_no_error(self, monkeypatch):
        """Writing empty points should not raise."""
        monkeypatch.setenv("INFLUXDB_URL", "http://localhost:8086")
        monkeypatch.setenv("INFLUXDB_ORG", "test")
        monkeypatch.setenv("INFLUXDB_BUCKET", "test")
        monkeypatch.setenv("INFLUXDB_TOKEN", "test")
        # Empty list — should return without making a request
        write_influxdb([])

    def test_partial_env_vars_reports_missing(self, monkeypatch):
        """Should report which specific vars are missing."""
        monkeypatch.setenv("INFLUXDB_URL", "http://localhost:8086")
        monkeypatch.delenv("INFLUXDB_ORG", raising=False)
        monkeypatch.delenv("INFLUXDB_BUCKET", raising=False)
        monkeypatch.setenv("INFLUXDB_TOKEN", "test")
        with pytest.raises(SystemExit):
            write_influxdb([{"ts": datetime.now(AEDT), "measurement": "x", "tags": {}, "value": 1}])


# ---------------------------------------------------------------------------
# Downsample intervals
# ---------------------------------------------------------------------------


class TestDownsampleIntervals:
    """Test different downsample intervals including the new 300s default."""

    def test_300s_default_interval(self, peakhurst_image):
        """Default 5-min downsample should produce ~289 points per metric."""
        results = extract_graph(peakhurst_image, "peakhurst", DATE)
        dl = [r for r in results if r["tags"]["metric"] == "download"]
        downsampled = downsample(dl, 300)
        # 24h / 5min = 288 buckets, but coverage may be slightly more
        assert 280 <= len(downsampled) <= 295

    def test_60s_vs_300s_point_count(self, peakhurst_image):
        """300s interval should produce significantly fewer points than 60s."""
        results = extract_graph(peakhurst_image, "peakhurst", DATE)
        dl = [r for r in results if r["tags"]["metric"] == "download"]
        ds_60 = downsample(dl, 60)
        ds_300 = downsample(dl, 300)
        # Raw points are ~1.67min apart so 60s doesn't reduce much,
        # but 300s should be roughly 3x fewer
        assert len(ds_60) > len(ds_300) * 2

    def test_300s_preserves_value_range(self, peakhurst_image):
        """Downsampling should preserve approximate min/max values."""
        results = extract_graph(peakhurst_image, "peakhurst", DATE)
        dl = [r for r in results if r["tags"]["metric"] == "download"]
        raw_max = max(r["value"] for r in dl)
        raw_min = min(r["value"] for r in dl)
        ds = downsample(dl, 300)
        ds_max = max(r["value"] for r in ds)
        ds_min = min(r["value"] for r in ds)
        # Averages dampen extremes slightly, but should be close
        assert ds_max == pytest.approx(raw_max, rel=0.15)
        assert ds_min == pytest.approx(raw_min, rel=0.3)

    def test_300s_timestamps_aligned(self, peakhurst_image):
        """Downsampled timestamps should be aligned to 5-minute boundaries."""
        results = extract_graph(peakhurst_image, "peakhurst", DATE)
        dl = [r for r in results if r["tags"]["metric"] == "download"]
        ds = downsample(dl, 300)
        for r in ds:
            assert int(r["ts"].timestamp()) % 300 == 0


# ---------------------------------------------------------------------------
# POI discovery
# ---------------------------------------------------------------------------


class TestDiscoverPois:
    """Test POI discovery with mocked HTTP responses."""

    SAMPLE_HTML = (
        '"https://cvcs.aussiebroadband.com.au/peakhurst.png","peakhurst","Peakhurst",'
        '"https://cvcs.aussiebroadband.com.au/peakhurstlink2.png","peakhurstlink2","Peakhurst (Link 2)",'
        '"https://cvcs.aussiebroadband.com.au/darwin.png","darwin","Darwin",'
    )

    def test_parses_pois_from_html(self):
        """discover_pois should extract slug and name from Nuxt payload."""
        with patch("abb_cvc_extract.urllib.request.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = self.SAMPLE_HTML.encode()

            pois = discover_pois()

        slugs = [p["slug"] for p in pois]
        assert "peakhurst" in slugs
        assert "peakhurstlink2" in slugs
        assert "darwin" in slugs

    def test_returns_sorted_by_slug(self):
        with patch("abb_cvc_extract.urllib.request.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = self.SAMPLE_HTML.encode()

            pois = discover_pois()

        slugs = [p["slug"] for p in pois]
        assert slugs == sorted(slugs)

    def test_deduplicates_slugs(self):
        duped_html = self.SAMPLE_HTML + self.SAMPLE_HTML  # same POIs twice
        with patch("abb_cvc_extract.urllib.request.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = duped_html.encode()

            pois = discover_pois()

        slugs = [p["slug"] for p in pois]
        assert len(slugs) == len(set(slugs))

    def test_includes_display_name(self):
        with patch("abb_cvc_extract.urllib.request.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = self.SAMPLE_HTML.encode()

            pois = discover_pois()

        by_slug = {p["slug"]: p["name"] for p in pois}
        assert by_slug["peakhurst"] == "Peakhurst"
        assert by_slug["peakhurstlink2"] == "Peakhurst (Link 2)"

    def test_empty_page_returns_empty(self):
        with patch("abb_cvc_extract.urllib.request.urlopen") as mock_urlopen:
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = b"<html>no data</html>"

            pois = discover_pois()

        assert pois == []


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


class TestCLI:
    """Test the main() CLI entrypoint with various argument combinations."""

    def test_single_poi_influx_output(self, peakhurst_image, capsys):
        """--poi with a local image should produce influx line protocol on stdout."""
        with patch(
            "sys.argv", ["prog", str(peakhurst_image), "--poi", "peakhurst", "--date", DATE]
        ):
            main()

        captured = capsys.readouterr()
        lines = [line for line in captured.out.strip().split("\n") if line]
        assert len(lines) > 0
        assert all(line.startswith("abb_cvc,") for line in lines)

    def test_csv_format(self, peakhurst_image, capsys):
        with patch(
            "sys.argv",
            ["prog", str(peakhurst_image), "--poi", "peakhurst", "--date", DATE, "--format", "csv"],
        ):
            main()

        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        assert lines[0] == "timestamp,poi,metric,value_mbps"
        assert len(lines) > 100

    def test_custom_interval(self, peakhurst_image, capsys):
        """--interval 600 should produce fewer points than default 300."""
        with patch(
            "sys.argv",
            [
                "prog",
                str(peakhurst_image),
                "--poi",
                "peakhurst",
                "--date",
                DATE,
                "--interval",
                "600",
                "--format",
                "csv",
            ],
        ):
            main()

        captured = capsys.readouterr()
        lines = [
            line
            for line in captured.out.strip().split("\n")
            if line and not line.startswith("timestamp")
        ]
        dl_lines = [line for line in lines if "download" in line]
        # 24h / 10min = ~144 points
        assert 130 <= len(dl_lines) <= 150

    def test_discover_list(self, capsys):
        """--discover-list should print slugs, one per line."""
        sample_html = (
            '"https://cvcs.aussiebroadband.com.au/alpha.png","alpha","Alpha",'
            '"https://cvcs.aussiebroadband.com.au/beta.png","beta","Beta",'
        )
        with (
            patch("sys.argv", ["prog", "--discover-list"]),
            patch("abb_cvc_extract.urllib.request.urlopen") as mock_urlopen,
        ):
            mock_resp = mock_urlopen.return_value.__enter__.return_value
            mock_resp.read.return_value = sample_html.encode()
            main()

        captured = capsys.readouterr()
        slugs = captured.out.strip().split("\n")
        assert slugs == ["alpha", "beta"]

    def test_write_influxdb_per_poi(self, peakhurst_image, monkeypatch):
        """--write-influxdb should call write_influxdb once per POI."""
        monkeypatch.setenv("INFLUXDB_URL", "http://localhost:8086")
        monkeypatch.setenv("INFLUXDB_ORG", "test")
        monkeypatch.setenv("INFLUXDB_BUCKET", "test")
        monkeypatch.setenv("INFLUXDB_TOKEN", "test")

        calls = []
        with (
            patch(
                "sys.argv",
                [
                    "prog",
                    str(peakhurst_image),
                    "--poi",
                    "peakhurst",
                    "--date",
                    DATE,
                    "--write-influxdb",
                ],
            ),
            patch("abb_cvc_extract.write_influxdb", side_effect=lambda pts: calls.append(len(pts))),
        ):
            main()

        assert len(calls) == 1
        assert calls[0] > 0

    def test_no_args_exits(self):
        """No arguments should exit with error."""
        with patch("sys.argv", ["prog"]), pytest.raises(SystemExit):
            main()

    def test_yesterday_flag(self, peakhurst_image, capsys):
        """--yesterday should resolve to a date string and not crash."""
        with patch(
            "sys.argv",
            [
                "prog",
                str(peakhurst_image),
                "--poi",
                "peakhurst",
                "--yesterday",
                "--format",
                "csv",
            ],
        ):
            main()

        captured = capsys.readouterr()
        assert "timestamp,poi,metric,value_mbps" in captured.out

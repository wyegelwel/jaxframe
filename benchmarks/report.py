#!/usr/bin/env python3
"""
JaxFrame benchmark report generator.

Reads JSON from benchmarks/results/ and produces a self-contained HTML report
with inline SVG charts showing performance as a function of data size.

Usage:
    uv run python benchmarks/report.py                    # use latest.json
    uv run python benchmarks/report.py results/20260427.json  # specific file
"""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# SVG chart generation
# ---------------------------------------------------------------------------

# Chart dimensions
CHART_W = 600
CHART_H = 350
MARGIN = {"top": 40, "right": 150, "bottom": 60, "left": 80}
PLOT_W = CHART_W - MARGIN["left"] - MARGIN["right"]
PLOT_H = CHART_H - MARGIN["top"] - MARGIN["bottom"]

COLORS = {
    "pandas": "#2196F3",  # blue
    "jaxframe": "#FF9800",  # orange
    "jit": "#4CAF50",  # green
    "raw_jnp": "#9C27B0",  # purple
}

LINE_LABELS = {
    "pandas": "pandas",
    "jaxframe": "jaxframe (eager)",
    "jit": "jaxframe (JIT)",
    "raw_jnp": "raw jnp (JIT)",
}


def _log_scale(val, vmin, vmax):
    """Map val to 0-1 on log scale."""
    if val <= 0 or vmin <= 0 or vmax <= 0:
        return 0
    log_val = math.log10(val)
    log_min = math.log10(vmin)
    log_max = math.log10(vmax)
    if log_max == log_min:
        return 0.5
    return (log_val - log_min) / (log_max - log_min)


def _nice_ticks(vmin, vmax, max_ticks=6):
    """Generate nice log-scale tick values."""
    if vmin <= 0:
        vmin = 0.001
    if vmax <= vmin:
        vmax = vmin * 10
    log_min = math.floor(math.log10(vmin))
    log_max = math.ceil(math.log10(vmax))
    ticks = []
    for exp in range(log_min, log_max + 1):
        base = 10**exp
        for mult in [1, 2, 5]:
            val = base * mult
            if vmin * 0.9 <= val <= vmax * 1.1:
                ticks.append(val)
    # Thin out if too many
    while len(ticks) > max_ticks:
        ticks = ticks[::2]
    return ticks


def _format_number(n):
    """Format large numbers compactly."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.0f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(int(n))


def _format_time(ms):
    """Format time value compactly."""
    if ms >= 100:
        return f"{ms:.0f}"
    if ms >= 1:
        return f"{ms:.1f}"
    if ms >= 0.01:
        return f"{ms:.2f}"
    return f"{ms:.3f}"


def make_svg_chart(title, x_label, series_data, x_is_log=True):
    """
    Generate an SVG chart string.

    series_data: dict of {series_name: [(x, y), ...]}
        series_name is one of: "pandas", "jaxframe", "jit", "raw_jnp"
    """
    # Collect all x and y values
    all_x = []
    all_y = []
    for pts in series_data.values():
        for x, y in pts:
            if y is not None and y > 0:
                all_x.append(x)
                all_y.append(y)

    if not all_x or not all_y:
        return (
            f'<svg width="{CHART_W}" height="{CHART_H}">'
            '<text x="50%" y="50%" text-anchor="middle">No data</text></svg>'
        )

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Add padding to y range
    y_min = y_min * 0.5
    y_max = y_max * 2

    def px(val):
        t = _log_scale(val, x_min, x_max) if x_is_log else (val - x_min) / max(x_max - x_min, 1)
        return MARGIN["left"] + t * PLOT_W

    def py(val):
        t = _log_scale(val, y_min, y_max)
        return MARGIN["top"] + PLOT_H - t * PLOT_H  # flip Y

    lines = []
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CHART_W}" height="{CHART_H}" '
        f'font-family="system-ui, -apple-system, sans-serif" font-size="11">'
    )

    # Background
    lines.append(f'<rect width="{CHART_W}" height="{CHART_H}" fill="white"/>')

    # Grid lines (Y axis)
    y_ticks = _nice_ticks(y_min, y_max)
    for tick in y_ticks:
        y_pos = py(tick)
        if MARGIN["top"] <= y_pos <= MARGIN["top"] + PLOT_H:
            lines.append(
                f'<line x1="{MARGIN["left"]}" y1="{y_pos:.1f}" '
                f'x2="{MARGIN["left"] + PLOT_W}" y2="{y_pos:.1f}" '
                f'stroke="#e0e0e0" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{MARGIN["left"] - 8}" y="{y_pos + 4:.1f}" '
                f'text-anchor="end" fill="#666" font-size="10">{_format_time(tick)}ms</text>'
            )

    # Grid lines (X axis)
    x_ticks = _nice_ticks(x_min, x_max) if x_is_log else sorted(set(all_x))
    for tick in x_ticks:
        x_pos = px(tick)
        if MARGIN["left"] <= x_pos <= MARGIN["left"] + PLOT_W:
            lines.append(
                f'<line x1="{x_pos:.1f}" y1="{MARGIN["top"]}" '
                f'x2="{x_pos:.1f}" y2="{MARGIN["top"] + PLOT_H}" '
                f'stroke="#e0e0e0" stroke-width="1"/>'
            )
            lines.append(
                f'<text x="{x_pos:.1f}" y="{MARGIN["top"] + PLOT_H + 16}" '
                f'text-anchor="middle" fill="#666" font-size="10">{_format_number(tick)}</text>'
            )

    # Plot area border
    lines.append(
        f'<rect x="{MARGIN["left"]}" y="{MARGIN["top"]}" '
        f'width="{PLOT_W}" height="{PLOT_H}" fill="none" stroke="#ccc" stroke-width="1"/>'
    )

    # Data lines
    for series_name, pts in series_data.items():
        color = COLORS.get(series_name, "#999")
        valid_pts = [(x, y) for x, y in pts if y is not None and y > 0]
        if not valid_pts:
            continue

        # Sort by x
        valid_pts.sort(key=lambda p: p[0])

        # Line
        path_parts = []
        for i, (x, y) in enumerate(valid_pts):
            cmd = "M" if i == 0 else "L"
            path_parts.append(f"{cmd}{px(x):.1f},{py(y):.1f}")
        lines.append(
            f'<path d="{" ".join(path_parts)}" fill="none" '
            f'stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>'
        )

        # Data points
        for x, y in valid_pts:
            lines.append(
                f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" '
                f'fill="{color}" stroke="white" stroke-width="1.5"/>'
            )

    # Legend
    legend_x = MARGIN["left"] + PLOT_W + 12
    legend_y = MARGIN["top"] + 10
    for i, (series_name, pts) in enumerate(series_data.items()):
        if not any(y is not None and y > 0 for _, y in pts):
            continue
        color = COLORS.get(series_name, "#999")
        label = LINE_LABELS.get(series_name, series_name)
        y_off = legend_y + i * 20
        lines.append(
            f'<line x1="{legend_x}" y1="{y_off}" x2="{legend_x + 20}" y2="{y_off}" '
            f'stroke="{color}" stroke-width="2.5"/>'
        )
        lines.append(
            f'<circle cx="{legend_x + 10}" cy="{y_off}" r="3" '
            f'fill="{color}" stroke="white" stroke-width="1"/>'
        )
        lines.append(
            f'<text x="{legend_x + 26}" y="{y_off + 4}" fill="#333" font-size="11">{label}</text>'
        )

    # Title
    lines.append(
        f'<text x="{CHART_W / 2}" y="{MARGIN["top"] - 12}" '
        f'text-anchor="middle" font-size="14" font-weight="600" fill="#222">{title}</text>'
    )

    # X axis label
    lines.append(
        f'<text x="{MARGIN["left"] + PLOT_W / 2}" y="{CHART_H - 8}" '
        f'text-anchor="middle" fill="#666" font-size="12">{x_label}</text>'
    )

    # Y axis label
    y_label_x = 14
    y_label_y = MARGIN["top"] + PLOT_H / 2
    lines.append(
        f'<text x="{y_label_x}" y="{y_label_y}" '
        f'text-anchor="middle" fill="#666" font-size="12" '
        f'transform="rotate(-90, {y_label_x}, {y_label_y})">Time (ms)</text>'
    )

    lines.append("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------


def extract_series(results, category, op, scaling, x_key):
    """Extract time series for a specific (category, op, scaling) combo."""
    filtered = [
        r
        for r in results
        if r["category"] == category and r["op"] == op and r["scaling"] == scaling
    ]
    series = {"pandas": [], "jaxframe": []}

    has_jit = any(r.get("jit_ms") is not None for r in filtered)
    if has_jit:
        series["jit"] = []

    for r in sorted(filtered, key=lambda r: r[x_key]):
        x = r[x_key]
        series["pandas"].append((x, r["pandas_ms"]))
        series["jaxframe"].append((x, r["jaxframe_ms"]))
        if has_jit:
            series["jit"].append((x, r.get("jit_ms")))

    return series


def extract_overhead(results):
    """Extract overhead comparison data."""
    filtered = [r for r in results if r["category"] == "overhead"]
    series = {"raw_jnp": [], "jit": []}
    for r in sorted(filtered, key=lambda r: r["n_rows"]):
        series["raw_jnp"].append((r["n_rows"], r["raw_jnp_ms"]))
        series["jit"].append((r["n_rows"], r["jaxframe_jit_ms"]))
    return series


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

CATEGORY_DISPLAY = {
    "scalar_reductions": "Scalar Reductions",
    "arithmetic_chains": "Arithmetic Chains",
    "column_reductions": "Per-Column Reductions",
    "cumulative": "Cumulative Operations",
    "shift_diff": "Shift & Diff",
    "rolling": "Rolling Windows",
    "expanding": "Expanding Windows",
    "ewm": "Exponentially Weighted",
    "data_cleaning": "Data Cleaning",
    "sorting": "Sorting",
    "groupby": "GroupBy Aggregations",
    "overhead": "Abstraction Overhead",
}


def generate_report(data, output_path):
    """Generate HTML report from benchmark data."""
    metadata = data["metadata"]
    results = data["results"]

    # Discover categories and ops
    cat_ops = defaultdict(set)
    for r in results:
        if r["category"] != "overhead":
            cat_ops[r["category"]].add(r["op"])

    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html lang='en'>")
    html.append("<head>")
    html.append("<meta charset='utf-8'>")
    html.append("<title>JaxFrame Benchmark Report</title>")
    html.append("<style>")
    html.append(CSS)
    html.append("</style>")
    html.append("</head>")
    html.append("<body>")

    # Header
    html.append("<h1>JaxFrame Benchmark Report</h1>")
    html.append("<div class='meta'>")
    html.append(f"<span>Device: <b>{metadata['device']}</b></span>")
    html.append(f"<span>JAX: {metadata['jax_version']}</span>")
    html.append(f"<span>NumPy: {metadata['numpy_version']}</span>")
    html.append(f"<span>pandas: {metadata['pandas_version']}</span>")
    html.append(f"<span>Python: {metadata['python_version']}</span>")
    html.append(f"<span>{metadata['timestamp'][:19]}</span>")
    html.append("</div>")

    # Executive summary table
    html.append("<h2>Executive Summary</h2>")
    html.append(_summary_table(results))

    # Crossover analysis
    html.append("<h2>Crossover Analysis</h2>")
    html.append("<p>Data size at which jaxframe JIT first beats pandas:</p>")
    html.append(_crossover_table(results))

    # Charts by category
    for cat_key in CATEGORY_DISPLAY:
        if cat_key not in cat_ops and cat_key != "overhead":
            continue

        cat_display = CATEGORY_DISPLAY.get(cat_key, cat_key)
        html.append(f"<h2>{cat_display}</h2>")

        if cat_key == "overhead":
            # Special handling for overhead
            series = extract_overhead(results)
            if series["raw_jnp"]:
                svg = make_svg_chart(
                    "Abstraction Overhead: raw JAX vs jaxframe JIT",
                    "Rows",
                    series,
                )
                html.append(f"<div class='chart'>{svg}</div>")
            continue

        ops = sorted(cat_ops[cat_key])

        for op_name in ops:
            # Row scaling chart
            row_series = extract_series(results, cat_key, op_name, "rows", "n_rows")
            if any(pts for pts in row_series.values()):
                title = f"{op_name} — Row Scaling (10 cols)"
                svg = make_svg_chart(title, "Rows", row_series)
                html.append(f"<div class='chart'>{svg}</div>")

            # Column scaling chart
            col_series = extract_series(results, cat_key, op_name, "cols", "n_cols")
            if any(pts for pts in col_series.values()):
                title = f"{op_name} — Column Scaling (1K rows)"
                svg = make_svg_chart(title, "Columns", col_series)
                html.append(f"<div class='chart'>{svg}</div>")

    html.append("</body></html>")

    Path(output_path).write_text("\n".join(html))
    return output_path


def _summary_table(results):
    """Build HTML summary table: best speedup per category at max rows."""
    rows_data = [r for r in results if r["scaling"] == "rows" and r["category"] != "overhead"]

    # Find max n_rows per category
    cat_max_rows = {}
    for r in rows_data:
        key = r["category"]
        cat_max_rows[key] = max(cat_max_rows.get(key, 0), r["n_rows"])

    # Get results at max rows
    summary = []
    for r in rows_data:
        if r["n_rows"] == cat_max_rows.get(r["category"]):
            jit_speedup = (
                r["pandas_ms"] / r["jit_ms"] if r.get("jit_ms") and r["jit_ms"] > 0 else None
            )
            eager_speedup = r["pandas_ms"] / r["jaxframe_ms"] if r["jaxframe_ms"] > 0 else None
            summary.append(
                {
                    "category": CATEGORY_DISPLAY.get(r["category"], r["category"]),
                    "op": r["op"],
                    "n_rows": r["n_rows"],
                    "pandas_ms": r["pandas_ms"],
                    "jaxframe_ms": r["jaxframe_ms"],
                    "jit_ms": r.get("jit_ms"),
                    "eager_speedup": eager_speedup,
                    "jit_speedup": jit_speedup,
                }
            )

    # Sort by JIT speedup descending
    summary.sort(key=lambda s: s.get("jit_speedup") or s.get("eager_speedup") or 0, reverse=True)

    lines = []
    lines.append("<table>")
    lines.append("<thead><tr>")
    lines.append("<th>Category</th><th>Op</th><th>Rows</th>")
    lines.append("<th>pandas (ms)</th><th>jaxframe (ms)</th><th>JIT (ms)</th>")
    lines.append("<th>Eager Speedup</th><th>JIT Speedup</th>")
    lines.append("</tr></thead><tbody>")

    for s in summary:
        jit_str = f"{s['jit_ms']:.2f}" if s["jit_ms"] is not None else "N/A"
        eager_sp = f"{s['eager_speedup']:.1f}x" if s["eager_speedup"] else "N/A"
        jit_sp = f"{s['jit_speedup']:.1f}x" if s["jit_speedup"] else "N/A"

        # Color the speedup cell
        jit_class = ""
        if s.get("jit_speedup"):
            if s["jit_speedup"] >= 5:
                jit_class = " class='fast'"
            elif s["jit_speedup"] >= 1:
                jit_class = " class='ok'"
            else:
                jit_class = " class='slow'"

        eager_class = ""
        if s.get("eager_speedup"):
            if s["eager_speedup"] >= 5:
                eager_class = " class='fast'"
            elif s["eager_speedup"] >= 1:
                eager_class = " class='ok'"
            else:
                eager_class = " class='slow'"

        lines.append(
            f"<tr><td>{s['category']}</td><td><code>{s['op']}</code></td>"
            f"<td>{_format_number(s['n_rows'])}</td>"
            f"<td>{s['pandas_ms']:.2f}</td><td>{s['jaxframe_ms']:.2f}</td>"
            f"<td>{jit_str}</td>"
            f"<td{eager_class}>{eager_sp}</td><td{jit_class}>{jit_sp}</td></tr>"
        )

    lines.append("</tbody></table>")
    return "\n".join(lines)


def _crossover_table(results):
    """Find crossover points where JIT beats pandas."""
    # Group by (category, op)
    grouped = defaultdict(list)
    for r in results:
        if r["scaling"] == "rows" and r.get("jit_ms") is not None:
            grouped[(r["category"], r["op"])].append(r)

    lines = []
    lines.append("<table>")
    lines.append("<thead><tr><th>Category</th><th>Op</th><th>Crossover (rows)</th></tr></thead>")
    lines.append("<tbody>")

    for (cat, op), entries in sorted(grouped.items()):
        entries.sort(key=lambda r: r["n_rows"])
        crossover = None
        for r in entries:
            if r["jit_ms"] is not None and r["jit_ms"] < r["pandas_ms"]:
                crossover = r["n_rows"]
                break

        cat_display = CATEGORY_DISPLAY.get(cat, cat)
        cross_str = _format_number(crossover) if crossover else "> max tested"
        lines.append(
            f"<tr><td>{cat_display}</td><td><code>{op}</code></td><td>{cross_str}</td></tr>"
        )

    lines.append("</tbody></table>")
    return "\n".join(lines)


CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 24px;
    background: #fafafa;
    color: #222;
    line-height: 1.5;
}
h1 {
    font-size: 28px;
    margin-bottom: 8px;
    color: #111;
}
h2 {
    font-size: 20px;
    margin: 32px 0 12px;
    padding-bottom: 6px;
    border-bottom: 2px solid #e0e0e0;
    color: #333;
}
.meta {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    font-size: 13px;
    color: #666;
    margin-bottom: 24px;
    padding: 8px 12px;
    background: #f0f0f0;
    border-radius: 6px;
}
.meta b { color: #333; }
.chart {
    display: inline-block;
    margin: 8px 4px;
    background: white;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 8px 0;
    font-size: 13px;
    background: white;
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
th, td {
    padding: 8px 12px;
    text-align: left;
    border-bottom: 1px solid #eee;
}
th {
    background: #f5f5f5;
    font-weight: 600;
    color: #555;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
td code {
    background: #f0f0f0;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 12px;
}
tr:hover { background: #f9f9f9; }
.fast { color: #2e7d32; font-weight: 600; }
.ok { color: #f57f17; font-weight: 600; }
.slow { color: #c62828; font-weight: 600; }
p { margin: 8px 0; color: #555; }
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # Find input file
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        input_path = Path(__file__).parent / "results" / "latest.json"

    if not input_path.exists():
        print(f"Error: {input_path} not found")
        print("Run 'uv run python benchmarks/run.py' first to generate data.")
        sys.exit(1)

    data = json.loads(input_path.read_text())

    # Write to report.html
    results_dir = Path(__file__).parent / "results"
    report_path = results_dir / "report.html"

    generate_report(data, report_path)
    print(f"Report written to: {report_path}")
    print(f"Open in browser: file://{report_path.resolve()}")


if __name__ == "__main__":
    main()

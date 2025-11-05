#!/usr/bin/env python3
"""
Visualize GryFlux graph timeline events as a packet-oriented Gantt chart.

Usage:
    python3 scripts/plot_graph_timeline.py \
        --input graph_timeline.json --output timeline.html \
        [--time-unit {us,ms}] [--limit-packets N] [--packets-per-page M]
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go
from plotly.express.colors import qualitative


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize GryFlux graph timeline events")
    parser.add_argument("--input", "-i", type=Path, default=Path("graph_timeline.json"), help="事件 JSON 文件路径")
    parser.add_argument("--output", "-o", type=Path, default=Path("timeline.html"), help="输出 HTML 文件路径")
    parser.add_argument("--min-duration-ns", type=int, default=0, help="过滤掉耗时更短的节点")
    parser.add_argument("--limit-packets", type=int, default=0, help="仅展示最早的前 N 个数据包")
    parser.add_argument("--packets-per-page", type=int, default=15, help="分页展示时每页的数据包数量")
    parser.add_argument(
        "--time-unit",
        choices=["us", "ms"],
        default="us",
        help="横轴单位，us=微秒，ms=毫秒",
    )
    return parser.parse_args()


def load_events(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "events" in data:
        return data["events"]
    if isinstance(data, list):
        return data
    raise ValueError("输入 JSON 格式不符合预期，需包含 events 数组。")


def build_intervals(events: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    started: Dict[Tuple[str, str], Dict] = {}
    rows: List[Dict] = []
    first_ts: Dict[str, int] = {}

    for evt in events:
        packet_raw = evt.get("packet_id", evt.get("packet"))
        node = evt.get("node")
        etype = evt.get("type")
        timestamp = evt.get("timestamp_ns", 0)

        if packet_raw is None or node is None or etype is None:
            continue

        packet_key = str(packet_raw)
        first_ts.setdefault(packet_key, timestamp)

        key = (packet_key, node)
        if etype == "started":
            started[key] = evt
        elif etype in {"finished", "failed"}:
            start_evt = started.pop(key, None)
            if not start_evt:
                continue
            rows.append(
                {
                    "packet": packet_key,
                    "node": node,
                    "start_ns": start_evt["timestamp_ns"],
                    "end_ns": evt["timestamp_ns"],
                    "duration_ns": max(evt.get("duration_ns", 0), 0),
                    "thread": evt.get("thread"),
                    "status": etype,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("没有找到匹配的 started/finished 事件。")

    df["packet_numeric"] = pd.to_numeric(df["packet"], errors="coerce")
    packet_order = (
        df.groupby("packet")
        .agg(first_ts=("start_ns", "min"), packet_numeric=("packet_numeric", "first"))
        .reset_index()
    )

    packet_order = packet_order.sort_values(
        ["packet_numeric", "first_ts"],
        na_position="last",
    ).reset_index(drop=True)

    def make_label(row: pd.Series) -> str:
        num = row["packet_numeric"]
        if pd.notna(num):
            try:
                return f"Packet #{int(num):03d}"
            except (ValueError, OverflowError):
                pass
        return f"Packet {row['packet']}"

    packet_order["packet_label"] = packet_order.apply(make_label, axis=1)
    label_mapping = dict(zip(packet_order["packet"], packet_order["packet_label"]))
    df["packet_label"] = df["packet"].map(label_mapping)

    return df.reset_index(drop=True), packet_order


def convert_time_columns(df: pd.DataFrame, unit: str) -> pd.DataFrame:
    ref = df["start_ns"].min()
    scale = 1_000.0 if unit == "us" else 1_000_000.0
    df = df.copy()
    df["start_offset"] = (df["start_ns"] - ref) / scale
    df["duration"] = df["duration_ns"] / scale
    df["end_offset"] = df["start_offset"] + df["duration"]
    df["start_abs"] = pd.to_datetime(df["start_ns"], unit="ns")
    df["duration_display_us"] = df["duration_ns"] / 1_000.0
    df["duration_display_ms"] = df["duration_ns"] / 1_000_000.0
    return df


def build_groups(packet_order: pd.DataFrame, limit_packets: int, per_page: int) -> Tuple[List[List[str]], List[str]]:
    if limit_packets > 0:
        ordered = packet_order.head(limit_packets)
    else:
        ordered = packet_order

    labels = ordered["packet_label"].tolist()
    if not labels:
        raise ValueError("没有可用的数据包用于可视化。")

    per_page = max(1, per_page)
    groups = [labels[i : i + per_page] for i in range(0, len(labels), per_page)]
    return groups, labels


def plot(df: pd.DataFrame, packet_order: pd.DataFrame, output: Path, min_duration_ns: int, limit_packets: int, unit: str, per_page: int) -> None:
    if min_duration_ns > 0:
        df = df[df["duration_ns"] >= min_duration_ns]
    if df.empty:
        raise ValueError("过滤后没有可视化的数据。")

    df = convert_time_columns(df, unit)
    groups, ordered_labels = build_groups(packet_order, limit_packets, per_page)
    df = df[df["packet_label"].isin(ordered_labels)]

    color_seq = qualitative.Plotly
    nodes = sorted(df["node"].unique())
    color_map = {node: color_seq[i % len(color_seq)] for i, node in enumerate(nodes)}
    packet_index = {label: idx for idx, label in enumerate(packet_order["packet_label"])}

    row_height = 2.4
    lane_spacing = row_height / max(len(nodes), 1)
    packet_positions = {label: packet_index[label] * row_height for label in ordered_labels}
    node_offsets = {
        node: (i - (len(nodes) - 1) / 2.0) * lane_spacing * 0.8
        for i, node in enumerate(nodes)
    }

    fig = go.Figure()
    traces_per_group: List[List[int]] = [[] for _ in groups]
    legend_stub_indices: List[int] = []

    for group_idx, labels in enumerate(groups):
        subset = df[df["packet_label"].isin(labels)]
        for node in nodes:
            node_df = subset[subset["node"] == node]
            if node_df.empty:
                continue

            x_vals: List[float] = []
            y_vals: List[float] = []
            texts: List[str] = []

            for row in node_df.itertuples():
                base = packet_positions[row.packet_label] + node_offsets[node]
                x_vals.extend([row.start_offset, row.end_offset, None])
                y_vals.extend([base, base, None])
                texts.extend([
                    (f"Packet: {row.packet_label}<br>Node: {node}<br>"
                     f"Start offset ({unit}): {row.start_offset:.3f}<br>"
                     f"End offset ({unit}): {row.end_offset:.3f}<br>"
                     f"Thread: {row.thread or 'N/A'}<br>"
                     f"Status: {row.status}<br>"
                     f"Start time: {row.start_abs}"),
                    (f"Packet: {row.packet_label}<br>Node: {node}<br>"
                     f"Start offset ({unit}): {row.start_offset:.3f}<br>"
                     f"End offset ({unit}): {row.end_offset:.3f}<br>"
                     f"Thread: {row.thread or 'N/A'}<br>"
                     f"Status: {row.status}<br>"
                     f"Start time: {row.start_abs}"),
                    "",
                ])

            trace = go.Scatter(
                name=node,
                x=x_vals,
                y=y_vals,
                mode="lines",
                line=dict(color=color_map[node], width=9),
                hoverinfo="text",
                text=texts,
                visible=(group_idx == 0),
                showlegend=False,
                legendgroup=node,
            )
            fig.add_trace(trace)
            traces_per_group[group_idx].append(len(fig.data) - 1)

    if not fig.data:
        raise ValueError("没有可绘制的节点数据。")

    for node in nodes:
        stub = go.Scatter(
            name=node,
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=color_map[node], width=9),
            hoverinfo="skip",
            showlegend=True,
            legendgroup=node,
        )
        fig.add_trace(stub)
        legend_stub_indices.append(len(fig.data) - 1)

    steps = []
    for group_idx, labels in enumerate(groups):
        visibility = [False] * len(fig.data)
        for idx in traces_per_group[group_idx]:
            visibility[idx] = True
        min_pos = min(packet_positions[label] for label in labels)
        max_pos = max(packet_positions[label] for label in labels)
        for stub_idx in legend_stub_indices:
            visibility[stub_idx] = True
        steps.append(
            {
                "args": [
                    {"visible": visibility},
                    {
                        "title": f"GryFlux Graph Timeline (Packets {labels[0]} - {labels[-1]})",
                        "yaxis": {
                            "categoryorder": "array",
                            "categoryarray": labels,
                            "tickvals": [packet_positions[label] for label in labels],
                            "ticktext": labels,
                            "range": [min_pos - row_height * 0.6, max_pos + row_height * 0.6],
                        },
                    },
                ],
                "label": f"{labels[0]}-{labels[-1]}",
                "method": "update",
            }
        )

    initial_labels = groups[0]
    min_initial = min(packet_positions[label] for label in initial_labels)
    max_initial = max(packet_positions[label] for label in initial_labels)
    fig.update_layout(
        barmode="overlay",
        title=f"GryFlux Graph Timeline (Packets {initial_labels[0]} - {initial_labels[-1]})",
        xaxis_title=f"Offset ({unit})",
        yaxis_title="Packet",
        bargap=0.08,
    height=min(900, max(600, len(initial_labels) * 55 + 220)),
        yaxis=dict(
            automargin=True,
            categoryorder="array",
            categoryarray=initial_labels,
            tickvals=[packet_positions[label] for label in initial_labels],
            ticktext=initial_labels,
            range=[min_initial - row_height * 0.6, max_initial + row_height * 0.6],
        ),
        sliders=[
            {
                "pad": {"b": 20, "t": 20},
                "currentvalue": {"visible": False},
                "len": 0.9,
                "steps": steps,
            }
        ],
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=12),
        ),
        margin=dict(l=70, r=30, t=70, b=50),
        font=dict(size=12),
        autosize=True,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(type="linear")
    fig.write_html(
        str(output),
        include_plotlyjs="cdn",
        full_html=True,
        default_width="100%",
        config={"responsive": True},
    )


def main() -> None:
    args = parse_args()
    events = load_events(args.input)
    intervals, order = build_intervals(events)
    plot(
        intervals,
        order,
        args.output,
        args.min_duration_ns,
        args.limit_packets,
        args.time_unit,
        args.packets_per_page,
    )
    print(f"Timeline saved to {args.output}")


if __name__ == "__main__":
    main()

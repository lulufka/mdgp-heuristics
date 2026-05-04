from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path

import networkx as nx

from mdgp.core.evaluation import cluster_density, partition_density, validate_partition
from mdgp.core.types import Partition


@dataclass(frozen=True)
class SvgStyle:
    width: int = 1400
    min_height: int = 700
    margin: int = 28
    title_height: int = 86
    cluster_cell_width: int = 240
    cluster_cell_height: int = 180
    node_radius: float = 9.0
    edge_width: float = 1.8
    font_size: int = 11


@dataclass(frozen=True)
class PartitionPanel:
    title: str
    G: nx.Graph
    partition: Partition
    layout_seed: int = 42
    highlight: str | None = None


def write_partition_svg(
    G: nx.Graph,
    partition: Partition,
    output_path: str | Path,
    *,
    title: str,
    seed: int = 42,
    show_labels: bool = True,
    style: SvgStyle | None = None,
) -> None:
    validate_partition(G, partition)
    style = style or SvgStyle()
    panel = PartitionPanel(title=title, G=G, partition=partition, layout_seed=seed)
    svg = render_svg_page([panel], show_labels=show_labels, style=style)
    write_svg(output_path, svg)


def write_partition_comparison_svg(
    G: nx.Graph,
    left_partition: Partition,
    right_partition: Partition,
    output_path: str | Path,
    *,
    title: str,
    left_title: str,
    right_title: str,
    seed: int = 42,
    show_labels: bool = True,
    style: SvgStyle | None = None,
) -> None:
    validate_partition(G, left_partition)
    validate_partition(G, right_partition)
    style = style or SvgStyle()
    left_density = partition_density(G, left_partition)
    right_density = partition_density(G, right_partition)
    left_highlight, right_highlight = comparison_highlights(left_density, right_density)
    panels = [
        PartitionPanel(
            title=left_title,
            G=G,
            partition=left_partition,
            layout_seed=seed,
            highlight=left_highlight,
        ),
        PartitionPanel(
            title=right_title,
            G=G,
            partition=right_partition,
            layout_seed=seed,
            highlight=right_highlight,
        ),
    ]
    svg = render_svg_page(panels, title=title, show_labels=show_labels, style=style)
    write_svg(output_path, svg)


def comparison_highlights(left_density: float, right_density: float) -> tuple[str, str]:
    tolerance = 1e-9
    if abs(left_density - right_density) <= tolerance:
        return "tie", "tie"
    if left_density > right_density:
        return "winner", "loser"
    return "loser", "winner"


def write_svg(output_path: str | Path, svg: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg, encoding="utf-8")


def render_svg_page(
    panels: list[PartitionPanel],
    *,
    show_labels: bool,
    style: SvgStyle,
    title: str | None = None,
) -> str:
    height = page_height(panels, style)
    page_title = title or panels[0].title

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{style.width}" height="{height}" viewBox="0 0 {style.width} {height}">',
        "<style>",
        ".title { font: 700 20px Arial, sans-serif; fill: #1f2933; }",
        ".panel-title { font: 700 15px Arial, sans-serif; fill: #1f2933; }",
        ".winner-title { fill: #1f7a4d; }",
        ".loser-title { fill: #52606d; }",
        ".tie-title { fill: #7c3aed; }",
        ".subtitle { font: 13px Arial, sans-serif; fill: #52606d; }",
        ".label { font: 12px Arial, sans-serif; fill: #323f4b; }",
        ".node-label { font: 11px Arial, sans-serif; fill: #ffffff; text-anchor: middle; dominant-baseline: central; pointer-events: none; }",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff" />',
        f'<text class="title" x="{style.margin}" y="34">{escape(page_title)}</text>',
    ]

    for panel_index, panel in enumerate(panels):
        bounds = panel_bounds(panel_index, len(panels), style)
        lines.extend(
            render_panel(
                panel,
                bounds,
                show_labels,
                show_panel_title=len(panels) > 1,
                style=style,
            )
        )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def page_height(panels: list[PartitionPanel], style: SvgStyle) -> int:
    max_rows = max(
        (cluster_rows(panel.G, panel.partition, style, len(panels)) for panel in panels),
        default=1,
    )
    required_height = style.title_height + max_rows * style.cluster_cell_height + style.margin
    return max(style.min_height, required_height)


def cluster_rows(G: nx.Graph, partition: Partition, style: SvgStyle, panel_count: int) -> int:
    panel_width = available_panel_width(style, panel_count)
    columns = panel_columns(panel_width, style)
    cluster_count = len(visible_clusters(G, partition, style))
    return max(1, (cluster_count + columns - 1) // columns)


def available_panel_width(style: SvgStyle, panel_count: int) -> float:
    gap = style.margin
    return (style.width - 2 * style.margin - gap * (panel_count - 1)) / panel_count


def panel_columns(panel_width: float, style: SvgStyle) -> int:
    cell_gap = 18
    return max(1, int((panel_width + cell_gap) // (style.cluster_cell_width + cell_gap)))


def panel_bounds(
    panel_index: int,
    panel_count: int,
    style: SvgStyle,
) -> tuple[float, float, float, float]:
    gap = style.margin
    total_width = style.width - 2 * style.margin - gap * (panel_count - 1)
    panel_width = total_width / panel_count
    x = style.margin + panel_index * (panel_width + gap)
    y = style.title_height
    return x, y, panel_width, 0.0


def render_panel(
    panel: PartitionPanel,
    bounds: tuple[float, float, float, float],
    show_labels: bool,
    show_panel_title: bool,
    style: SvgStyle,
) -> list[str]:
    x, y, width, _ = bounds
    clusters = visible_clusters(panel.G, panel.partition, style)
    columns = panel_columns(width, style)
    cell_gap = 18
    cell_width = style.cluster_cell_width

    lines: list[str] = []
    if show_panel_title:
        title_class = panel_title_class(panel)
        title_suffix = panel_title_suffix(panel)
        lines.append(
            f'<text class="{title_class}" x="{x:.2f}" y="{y - 36:.2f}">'
            f'{escape(panel.title + title_suffix)}</text>'
        )

    lines.append(
        f'<text class="subtitle" x="{x:.2f}" y="{y - 16:.2f}">{summary_text(panel.G, panel.partition, style)}</text>'
    )

    if not clusters:
        lines.append(
        f'<text class="label" x="{x:.2f}" y="{y + 24:.2f}">'
            "No clusters to display</text>"
        )
        return lines

    for cluster_number, cluster in enumerate(clusters):
        col = cluster_number % columns
        row = cluster_number // columns
        cell_x = x + col * (cell_width + cell_gap)
        cell_y = y + row * style.cluster_cell_height
        lines.extend(
            render_cluster_cell(
                panel.G,
                cluster,
                cluster_number + 1,
                cell_x,
                cell_y,
                cell_width,
                style.cluster_cell_height,
                show_labels,
                panel.layout_seed,
                style,
            )
        )

    return lines


def panel_title_class(panel: PartitionPanel) -> str:
    if panel.highlight == "winner":
        return "panel-title winner-title"
    if panel.highlight == "loser":
        return "panel-title loser-title"
    if panel.highlight == "tie":
        return "panel-title tie-title"
    return "panel-title"


def panel_title_suffix(panel: PartitionPanel) -> str:
    if panel.highlight == "winner":
        return " (higher density)"
    if panel.highlight == "tie":
        return " (same density)"
    return ""


def visible_clusters(G: nx.Graph, partition: Partition, style: SvgStyle) -> list[set[int]]:
    clusters = [set(cluster) for cluster in partition]
    return sorted(
        clusters,
        key=lambda cluster: (-len(cluster), -G.subgraph(cluster).number_of_edges(), sorted(cluster)),
    )


def summary_text(G: nx.Graph, partition: Partition, style: SvgStyle) -> str:
    counts = cluster_size_counts(partition)
    return escape(
        f"n={G.number_of_nodes()} | m={G.number_of_edges()} | "
        f"clusters={len(partition)} | density={partition_density(G, partition):.3f} | "
        f"Clusters of size 1={counts.get(1, 0)} | Clusters of size 2={counts.get(2, 0)}"
    )


def cluster_size_counts(partition: Partition) -> dict[int, int]:
    counts: dict[int, int] = {}
    for cluster in partition:
        size = len(cluster)
        counts[size] = counts.get(size, 0) + 1
    return counts


def render_cluster_cell(
    G: nx.Graph,
    cluster: set[int],
    display_index: int,
    x: float,
    y: float,
    width: float,
    height: float,
    show_labels: bool,
    layout_seed: int,
    style: SvgStyle,
) -> list[str]:
    subgraph = induced_cluster_graph(G, cluster)
    title = (
        f"#{display_index}: |V|={len(cluster)}, "
        f"|E|={subgraph.number_of_edges()}, density={cluster_density(G, cluster):.2f}"
    )
    graph_x = x + 18
    graph_y = y + 34
    graph_width = width - 36
    graph_height = height - 52
    positions = cluster_layout(
        subgraph,
        graph_x,
        graph_y,
        graph_width,
        graph_height,
        seed=cluster_layout_seed(cluster, layout_seed),
    )

    lines = [
        f'<text class="label" x="{x:.2f}" y="{y + 14:.2f}">{escape(title)}</text>'
    ]
    lines.extend(render_cluster_edges(subgraph, positions, style))
    lines.extend(render_cluster_nodes(subgraph, positions, show_labels, style))
    return lines


def induced_cluster_graph(G: nx.Graph, cluster: set[int]) -> nx.Graph:
    return G.subgraph(cluster).copy()


def cluster_layout(
    G: nx.Graph,
    x: float,
    y: float,
    width: float,
    height: float,
    seed: int,
) -> dict[int, tuple[float, float]]:
    if G.number_of_nodes() == 0:
        return {}

    raw_positions = nx.spring_layout(G, seed=seed)
    return scale_positions(raw_positions, x, y, width, height)


def cluster_layout_seed(cluster: set[int], layout_seed: int) -> int:
    return layout_seed + sum((index + 1) * node for index, node in enumerate(sorted(cluster)))


def scale_positions(
    raw_positions: dict[int, tuple[float, float]],
    x: float,
    y: float,
    width: float,
    height: float,
) -> dict[int, tuple[float, float]]:
    if len(raw_positions) == 1:
        node = next(iter(raw_positions))
        return {node: (x + width / 2, y + height / 2)}

    xs = [pos[0] for pos in raw_positions.values()]
    ys = [pos[1] for pos in raw_positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    padding = max(style_padding(width, height), 0.0)
    drawable_width = max(width - 2 * padding, 1.0)
    drawable_height = max(height - 2 * padding, 1.0)

    return {
        node: (
            x + padding + ((pos[0] - min_x) / span_x) * drawable_width,
            y + padding + ((pos[1] - min_y) / span_y) * drawable_height,
        )
        for node, pos in raw_positions.items()
    }


def style_padding(width: float, height: float) -> float:
    return min(width, height) * 0.08


def render_cluster_edges(
    G: nx.Graph,
    positions: dict[int, tuple[float, float]],
    style: SvgStyle,
) -> list[str]:
    lines: list[str] = []
    for u, v in G.edges():
        x1, y1 = positions[int(u)]
        x2, y2 = positions[int(v)]
        lines.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="#627d98" stroke-opacity="0.72" stroke-width="{style.edge_width}" />'
        )
    return lines


def render_cluster_nodes(
    G: nx.Graph,
    positions: dict[int, tuple[float, float]],
    show_labels: bool,
    style: SvgStyle,
) -> list[str]:
    lines: list[str] = []
    for node in sorted(G.nodes()):
        x, y = positions[int(node)]
        lines.append(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{style.node_radius}" '
            f'fill="#334e68" stroke="#ffffff" stroke-width="1.6">'
            f"<title>node {escape(str(node))}</title>"
            "</circle>"
        )
        if show_labels:
            lines.append(
                f'<text class="node-label" x="{x:.2f}" y="{y:.2f}" '
                f'font-size="{style.font_size}">{escape(str(node))}</text>'
            )
    return lines

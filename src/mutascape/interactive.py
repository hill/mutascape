"""Interactive visualisations for mutascape fitness landscapes.

Uses Plotly for 3D surfaces, animations, and hover interactivity.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray
from scipy.interpolate import griddata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from mutascape.epistasis import AMINO_ACIDS
from mutascape.landscape import Landscape


def _encode_variants_onehot(
    variants: list[dict[int, str]],
    positions: list[int],
    amino_acids: tuple[str, ...],
) -> NDArray[np.float64]:
    """One-hot encode mutation dicts into a feature matrix.

    Each variant becomes a vector of length ``n_positions * n_amino_acids``
    where each block of 20 encodes which amino acid occupies that position.
    """
    n_aa = len(amino_acids)
    aa_idx = {aa: i for i, aa in enumerate(amino_acids)}
    matrix = np.zeros((len(variants), len(positions) * n_aa), dtype=np.float64)

    for row, variant in enumerate(variants):
        for col, pos in enumerate(positions):
            aa = variant.get(pos)
            if aa is not None and aa in aa_idx:
                matrix[row, col * n_aa + aa_idx[aa]] = 1.0

    return matrix


def _embed_2d(
    features: NDArray[np.float64],
    method: str = "pca",
    random_state: int = 42,
) -> tuple[NDArray[np.float64], str]:
    """Reduce a feature matrix to 2D coordinates.

    Returns ``(coords, method_label)``.
    """
    method = method.lower()

    if method == "pca":
        reducer = PCA(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(features)
        return coords, "PCA"

    if method == "tsne":
        perplexity = min(30, features.shape[0] - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        coords = reducer.fit_transform(features)
        return coords, "t-SNE"

    raise ValueError(f"Unknown embedding method: '{method}'. Choose 'pca' or 'tsne'.")


def _build_full_genotypes(
    variants: list[dict[int, str]],
    wild_type: dict[int, str],
) -> list[dict[int, str]]:
    """Expand sparse mutation dicts to full genotypes (all positions filled)."""
    full = []
    for v in variants:
        genotype = dict(wild_type)
        genotype.update(v)
        full.append(genotype)
    return full


def _variant_labels(
    variants: list[dict[int, str]],
    wild_type: dict[int, str],
) -> list[str]:
    """Build human-readable labels like 'A10G+L20I' for hover text."""
    labels = []
    for v in variants:
        if not v:
            labels.append("WT")
            continue
        parts = []
        for pos in sorted(v):
            wt_aa = wild_type.get(pos, "?")
            parts.append(f"{wt_aa}{pos}{v[pos]}")
        labels.append("+".join(parts))
    return labels


def _evaluate_at_alpha(
    landscape: Landscape,
    variants: list[dict[int, str]],
    alpha: float,
) -> NDArray[np.float64]:
    """Evaluate variants at a given alpha without mutating the original landscape."""
    ls = Landscape(
        wild_type=landscape.wild_type,
        single_site_effects=landscape.single_site_effects,
        contact_map=landscape.contact_map,
        couplings=landscape.couplings,
        alpha=alpha,
        noise_sigma=0.0,
        seed=landscape.seed,
        wt_sequence=landscape.wt_sequence,
    )
    return ls.evaluate_batch(variants)


def _interpolate_surface(
    x: NDArray,
    y: NDArray,
    fitness: NDArray,
    grid_resolution: int,
) -> tuple[NDArray, NDArray, NDArray]:
    """Interpolate scattered (x, y, fitness) onto a regular grid.

    Returns ``(xi, yi, zi_grid)`` where ``zi_grid`` has shape
    ``(grid_resolution, grid_resolution)``.
    """
    x_margin = (x.max() - x.min()) * 0.05
    y_margin = (y.max() - y.min()) * 0.05
    xi = np.linspace(x.min() - x_margin, x.max() + x_margin, grid_resolution)
    yi = np.linspace(y.min() - y_margin, y.max() + y_margin, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    zi_grid = griddata((x, y), fitness, (xi_grid, yi_grid), method="cubic", fill_value=np.nan)

    return xi, yi, zi_grid


def _make_surface(
    xi: NDArray,
    yi: NDArray,
    zi_grid: NDArray,
    *,
    colorscale: str,
    cmin: float,
    cmax: float,
    opacity: float,
) -> go.Surface:
    return go.Surface(
        x=xi,
        y=yi,
        z=zi_grid,
        colorscale=colorscale,
        opacity=opacity,
        cmin=cmin,
        cmax=cmax,
        showscale=True,
        colorbar=dict(title="Fitness", len=0.6),
        hoverinfo="skip",
    )


def _make_scatter(
    x: NDArray,
    y: NDArray,
    fitness: NDArray,
    labels: list[str],
    *,
    colorscale: str,
    cmin: float,
    cmax: float,
) -> go.Scatter3d:
    return go.Scatter3d(
        x=x,
        y=y,
        z=fitness,
        mode="markers",
        marker=dict(
            size=2.5,
            color=fitness,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            showscale=False,
            line=dict(width=0),
        ),
        text=[f"{lab}<br>fitness: {f:.3f}" for lab, f in zip(labels, fitness)],
        hoverinfo="text",
        name="variants",
    )


def morphing_landscape(
    landscape: Landscape,
    n_samples: int = 2000,
    n_mutations: int | tuple[int, int] = (1, 4),
    alphas: list[float] | None = None,
    embed_method: str = "pca",
    seed: int = 42,
    width: int = 900,
    height: int = 700,
    colorscale: str = "Viridis",
    surface_opacity: float = 0.85,
    grid_resolution: int = 60,
    title: str | None = None,
) -> go.Figure:
    """Animated 3D fitness landscape that morphs as epistatic strength changes.

    Samples a fixed set of combinatorial variants, embeds their mutation
    profiles into 2D, then sweeps alpha to show how the fitness surface
    transitions from smooth (additive) to rugged (epistatic).

    The surface is interpolated from scattered variant fitness values onto
    a regular grid, giving a continuous landscape rather than a point cloud.

    Args:
        landscape: A mutascape Landscape instance.
        n_samples: Number of random variants to sample.
        n_mutations: Fixed mutation count or (min, max) range.
        alphas: Alpha values to sweep. Defaults to [0, 0.25, 0.5, 1, 2, 3, 5].
        embed_method: Dimensionality reduction for 2D layout ('pca' or 'tsne').
        seed: Random seed for sampling and embedding.
        width: Figure width in pixels.
        height: Figure height in pixels.
        colorscale: Plotly colourscale name.
        surface_opacity: Opacity of the interpolated surface (0 to 1).
        grid_resolution: Number of grid points per axis for surface interpolation.
        title: Custom plot title.

    Returns:
        plotly Figure with animation slider and play/pause controls.
    """
    if alphas is None:
        alphas = [0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0]

    # Sample variants from a seeded copy so we don't mutate the caller's RNG.
    # _sample() takes a fixed int for n_mutations, so when a range is given
    # we split samples evenly across depths.
    sampler_ls = Landscape(
        wild_type=landscape.wild_type,
        single_site_effects=landscape.single_site_effects,
        contact_map=landscape.contact_map,
        couplings=landscape.couplings,
        alpha=landscape.alpha,
        noise_sigma=0.0,
        seed=seed,
        wt_sequence=landscape.wt_sequence,
    )

    if isinstance(n_mutations, tuple):
        min_k, max_k = n_mutations
        depths = list(range(min_k, max_k + 1))
        per_depth = max(1, n_samples // len(depths))
        variants: list[dict[int, str]] = []
        for k in depths:
            batch, _ = sampler_ls._sample(n=per_depth, n_mutations=k)
            variants.extend(batch)
    else:
        variants, _ = sampler_ls._sample(n=n_samples, n_mutations=n_mutations)

    # Encode mutations and embed into 2D
    full_genotypes = _build_full_genotypes(variants, landscape.wild_type)
    features = _encode_variants_onehot(full_genotypes, landscape.positions, AMINO_ACIDS)
    coords, method_label = _embed_2d(features, method=embed_method, random_state=seed)

    x, y = coords[:, 0], coords[:, 1]
    labels = _variant_labels(variants, landscape.wild_type)

    # Evaluate fitness at each alpha
    all_fitness = {alpha: _evaluate_at_alpha(landscape, variants, alpha) for alpha in alphas}

    # Consistent z-axis range across all frames
    global_zmin = min(f.min() for f in all_fitness.values())
    global_zmax = max(f.max() for f in all_fitness.values())
    z_pad = (global_zmax - global_zmin) * 0.05
    zmin = global_zmin - z_pad
    zmax = global_zmax + z_pad

    surface_kw: dict[str, Any] = dict(colorscale=colorscale, cmin=global_zmin, cmax=global_zmax)
    scatter_kw: dict[str, Any] = dict(colorscale=colorscale, cmin=global_zmin, cmax=global_zmax)

    # Build animation frames
    frames = []
    for alpha in alphas:
        fitness = all_fitness[alpha]
        xi, yi, zi_grid = _interpolate_surface(x, y, fitness, grid_resolution)
        frames.append(
            go.Frame(
                data=[
                    _make_surface(xi, yi, zi_grid, opacity=surface_opacity, **surface_kw),
                    _make_scatter(x, y, fitness, labels, **scatter_kw),
                ],
                name=f"{alpha:.2f}",
                layout=go.Layout(
                    title=dict(text=title or f"Fitness Landscape (\u03b1 = {alpha:.2f})")
                ),
            )
        )

    # Initial frame
    init_fitness = all_fitness[alphas[0]]
    xi, yi, zi_grid = _interpolate_surface(x, y, init_fitness, grid_resolution)

    fig = go.Figure(
        data=[
            _make_surface(xi, yi, zi_grid, opacity=surface_opacity, **surface_kw),
            _make_scatter(x, y, init_fitness, labels, **scatter_kw),
        ],
        frames=frames,
    )

    # Slider steps
    slider_steps = [
        dict(
            args=[[f"{a:.2f}"], dict(frame=dict(duration=600, redraw=True), mode="immediate")],
            label=f"{a:.2f}",
            method="animate",
        )
        for a in alphas
    ]

    fig.update_layout(
        title=dict(text=title or f"Fitness Landscape (\u03b1 = {alphas[0]:.2f})"),
        scene=dict(
            xaxis=dict(title=f"{method_label} 1", showgrid=False, zeroline=False),
            yaxis=dict(title=f"{method_label} 2", showgrid=False, zeroline=False),
            zaxis=dict(title="Fitness", range=[zmin, zmax]),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        ),
        width=width,
        height=height,
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                x=0.05,
                y=0.0,
                xanchor="left",
                yanchor="bottom",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=800, redraw=True),
                                fromcurrent=True,
                                transition=dict(duration=400, easing="cubic-in-out"),
                            ),
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            dict(frame=dict(duration=0, redraw=False), mode="immediate"),
                        ],
                    ),
                ],
            ),
        ],
        sliders=[
            dict(
                active=0,
                pad=dict(t=40),
                steps=slider_steps,
                currentvalue=dict(prefix="\u03b1 = ", visible=True, xanchor="center"),
            ),
        ],
    )

    return fig

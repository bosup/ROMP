import os
#import pandas as pd
from dataclasses import asdict, dataclass

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from momp.graphics.func_map import spatial_metrics_map
from momp.lib.control import make_case, ref_cfg_layout
from momp.lib.convention import Case
from momp.lib.loader import get_cfg, get_setting
from momp.utils.printing import tuple_to_str


@dataclass
class PanelMapPlotConfig:
    max_cols: int = 4
    figsize_col: float = 3.0
    figsize_row: float = 3.4
    left: float = 0.04
    right: float = 0.98
    top: float = 0.90
    bottom: float = 0.10
    wspace: float = 0.02
    hspace: float = 0.30
    cbar_hpad: float = 0.22
    cbar_vshift: float = 0.012
    cbar_frac_ref: float = 0.70
    cbar_frac_mod: float = 0.70
    n_ticks: int = 7


@dataclass
class PanelMapLayout:
    fig: plt.Figure
    map_axes: list
    cax_ref: plt.Axes
    cax_mod: plt.Axes | None
    text_scale: float
    nrows: int
    ncols: int


def build_panel_layout(n_panels: int, plot_cfg: PanelMapPlotConfig) -> PanelMapLayout:
    """Create a Cartopy-safe layout with a dedicated top colorbar row."""
    ncols = max(1, min(plot_cfg.max_cols, n_panels))
    nrows = int(np.ceil(n_panels / ncols))

    fig_w = max(7.5, plot_cfg.figsize_col * ncols)
    fig_h = max(4.0, plot_cfg.figsize_row * nrows + 0.7)
    fig = plt.figure(figsize=(fig_w, fig_h))


    # 
    gspec = GridSpec(
        nrows + 1,
        ncols,
        figure=fig,
        height_ratios=[0.06] + [1.0] * nrows,
        left=plot_cfg.left,
        right=plot_cfg.right,
        top=plot_cfg.top,
        bottom=plot_cfg.bottom,
        wspace=plot_cfg.wspace,
        hspace=plot_cfg.hspace,
    )

    map_axes = []
    for idx in range(n_panels):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gspec[row + 1, col], projection=ccrs.PlateCarree())
        ax.set_anchor("N")
        map_axes.append(ax)

    cax_ref = fig.add_subplot(gspec[0, 0])
    cax_ref.set_position(_shrink_horizontally(cax_ref.get_position(), plot_cfg.cbar_frac_ref, plot_cfg.cbar_vshift))

    cax_mod = None
    if n_panels > 1:
        mod_cell = gspec[0, 1:] if ncols > 1 else gspec[0, :]
        cax_mod = fig.add_subplot(mod_cell)
        cax_mod.set_position(_shrink_horizontally(cax_mod.get_position(), plot_cfg.cbar_frac_mod, plot_cfg.cbar_vshift))

    text_scale = min(fig_w / ncols, fig_h / max(1, nrows)) / 5.0 * 1.2

    return PanelMapLayout(
        fig=fig,
        map_axes=map_axes,
        cax_ref=cax_ref,
        cax_mod=cax_mod,
        text_scale=text_scale,
        nrows=nrows,
        ncols=ncols,
    )


def _shrink_horizontally(pos, frac: float, vshift: float):
    new_w = pos.width * frac
    new_x = pos.x0 + (pos.width - new_w) / 2.0
    return [new_x, pos.y0 + vshift, new_w, pos.height]


def panel_map_mae_far_mr(
    model_list,
    verification_window,
    var_name,
    cfg,
    setting,
    plot_cfg: PanelMapPlotConfig | None = None,
    **kwargs,
):

    window_str = tuple_to_str(verification_window)
    plot_cfg = plot_cfg or PanelMapPlotConfig()
    n = len(model_list)
    layout = build_panel_layout(n, plot_cfg)
    fig = layout.fig
    axes = []
    ims = []

    da_all = []
    unit = "days"

    for i, model in enumerate(model_list):
        fi = os.path.join(cfg.dir_out,"spatial_metrics_{}_{}.nc")
        fi = fi.format(model, window_str)
        ds = xr.open_dataset(fi)
        da = ds[var_name]
        unit = "days"
        if var_name in ["false_alarm_rate", "miss_rate"]:
            da = da*100
            unit = r"$(\%)$"
        ds.close()

        if i == 0:
            da_all.append(da)
        elif i > 0:
            da = da - da_all[0] 
            da_all.append(da)

    da_combined = xr.concat(da_all[1:], dim='model')
    v_low = da_combined.quantile(0.05).item()
    v_high = da_combined.quantile(0.95).item()
    limit = max(abs(v_low), abs(v_high))
    vmin = -limit
    vmax = limit

    for i, model in enumerate(model_list):
        da = da_all[i]

        combi = (model, verification_window)
        if model == cfg.ref_model:
            cfg_ref, _ = ref_cfg_layout(cfg, ref_model=model, verification_window=verification_window)
            case = make_case(Case, combi, vars(cfg_ref))
        else:
            case = make_case(Case, combi, vars(cfg))

        case_cfg = {**asdict(case), **asdict(setting)}

        if i > 0:
            #show_ylabel, title, cmap = False, None, 'RdBu_r'
            show_ylabel, title, cmap = False, r"$\Delta$" + unit, "RdBu_r"
        else:
            #show_ylabel, title, cmap = True, f"{var_name} {window_str} day", 'YlOrRd'
            #show_ylabel, title, cmap = True, None, 'YlOrRd'
            show_ylabel, title, cmap = True, unit, "YlOrRd"

        print(i, show_ylabel, title, cmap)

        if i > 0:
            vmin, vmax = -limit, limit
        else:
            vmin, vmax = None, None

        ax = layout.map_axes[i]
        fig, ax, im, _ = spatial_metrics_map(da, model, fig=fig, ax=ax, domain_mask=True, n_colors=0,
                                         show_ylabel=show_ylabel, cmap=cmap, title=title, panel=True, 
                                         text_scale=layout.text_scale, vmin=vmin, vmax=vmax, **case_cfg)

        axes.append(ax)
        ims.append(im)

    cbar_ref = fig.colorbar(ims[0], cax=layout.cax_ref, orientation="horizontal")
    cbar_ref.ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ref_vmin = da_all[0].quantile(0.1).item()
    ref_vmax = da_all[0].quantile(0.9).item()
    ref_ticks = np.linspace(ref_vmin, ref_vmax, plot_cfg.n_ticks).astype(int)
    cbar_ref.set_ticks(ref_ticks)
    cbar_ref.set_ticklabels([str(v) for v in ref_ticks])
    cbar_ref.ax.tick_params(labelsize=7, direction="in", length=1.5)

    if n > 1 and layout.cax_mod is not None:
        cbar_mod = fig.colorbar(ims[1], cax=layout.cax_mod, orientation="horizontal", extend="both")
        cbar_mod.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        mod_ticks = np.linspace(-limit, limit, plot_cfg.n_ticks).astype(int)
        cbar_mod.set_ticks(mod_ticks)
        cbar_mod.set_ticklabels([str(v) for v in mod_ticks])
        cbar_mod.ax.tick_params(labelsize=7, direction="in", length=1.5)

    fig.suptitle(f"{var_name} {window_str} day forecast", fontsize=12, y=0.98)
    #plt.subplots_adjust(top=0.88, hspace=0.1) # this will change colorbar size and position!!!
#    plt.tight_layout(rect=[0, 0, 0.99, 1]) # doesn't work for Cartopy axes

    plot_filename = f"map_{var_name}_{window_str}.png"
    plot_path = os.path.join(cfg.dir_fig, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {plot_path}")

    plt.show()

    return fig, axes, ims
    

if __name__ == "__main__":

    from itertools import product
    from dataclasses import asdict
    import xarray as xr
    #from momp.stats.benchmark import compute_metrics_multiple_years
    from momp.lib.control import iter_list, make_case
    from momp.lib.control import ref_cfg_layout, ref_model_case
    from momp.lib.convention import Case
    from momp.lib.loader import get_cfg, get_setting
    #from momp.metrics.error import create_spatial_far_mr_mae
    from momp.graphics.func_map import spatial_metrics_map
    from momp.utils.printing import tuple_to_str
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    from matplotlib.ticker import MaxNLocator, FixedLocator
    from matplotlib.gridspec import GridSpec
    from matplotlib import colors as mcolors


    cfg, setting = get_cfg(), get_setting()

    model_list = cfg.model_list
    model_list = (cfg.ref_model,) + model_list
    verification_window = cfg.verification_window_list[0]
    print(verification_window)
    window_str = tuple_to_str(verification_window)
    
    var_name = "mean_mae"

    n = len(model_list)
    #fig = plt.figure(figsize=(8, 5/4*n), constrained_layout=True)
    fig = plt.figure(figsize=(8, 5/4*n))
    axes = []
    ims = []
    fig_width, fig_height = fig.get_size_inches()
    text_scale =  min(fig_width/n, fig_height/1) / 5 * 1.2
    #print(txt_scale)
    #import sys
    #sys.exit()

    gs = GridSpec(
        2, n,
        height_ratios=[1, 0.043],   # plots, then colorbars
        hspace=0.02,
        wspace=0.07
    )

    for i, model in enumerate(model_list):
        fi = os.path.join(cfg.dir_out,"spatial_metrics_{}_{}.nc")
        fi = fi.format(model, window_str)
        ds = xr.open_dataset(fi)
        da = ds[var_name]
        ds.close()

        combi = (model, verification_window)
        if model == cfg.ref_model:
            cfg_ref, _ = ref_cfg_layout(cfg, ref_model=model, verification_window=verification_window)
            case = make_case(Case, combi, vars(cfg_ref))
        else:
            case = make_case(Case, combi, vars(cfg))

        case_cfg = {**asdict(case), **asdict(setting)}

        if i > 0:
            show_ylabel, title, cmap = False, None, 'RdBu_r'
        else:
            show_ylabel, title, cmap = True, f"{var_name} {window_str} day", 'YlOrRd'

        print(i, show_ylabel, title, cmap)

        #ax = fig.add_subplot(1, n, i+1, projection=ccrs.PlateCarree())
        ax = fig.add_subplot(gs[0, i], projection=ccrs.PlateCarree())
        #ax = fig.add_subplot(gs[0, i])

        fig, ax, im, _ = spatial_metrics_map(da, model, fig=fig, ax=ax, domain_mask=True, 
                                         show_ylabel=show_ylabel, cmap=cmap, title=title, panel=True, 
                                         text_scale=text_scale, **case_cfg)

        axes.append(ax)
        ims.append(im)
        i += 1


    #cbar_ax_ref = fig.add_axes([0.10, 0.05, 0.30, 0.02])
    #cbar_ax_mod = fig.add_axes([0.40, 0.05, 0.30, 0.02])
    #cb_ref = fig.colorbar(ims[0], cax=cbar_ax_ref, orientation='horizontal')
    #cb_mod = fig.colorbar(ims[1], cax=cbar_ax_mod, orientation='horizontal')

    #cbar_ref = fig.colorbar(ims[0], ax=axes[0], orientation='horizontal', fraction=0.046, pad=0.02)
    #cbar_ref.set_label("Reference units")
    #
    #cbar_mod = fig.colorbar(ims[1], ax=axes[1:], orientation='horizontal', fraction=0.046, pad=0.02)
    #cbar_mod.set_label("Model units")

#    cbar_ref.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#    cbar_mod.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
#
#    ticks = cb_ref.get_ticks()
#    cb_ref.set_ticks(ticks[::2])

    cax_ref = fig.add_subplot(gs[1, 0])
    cbar_ref = fig.colorbar(
        ims[0],
        cax=cax_ref,
        orientation='horizontal'
    )
    
    #cax_mod = fig.add_subplot(gs[1, 1:])

    gs_cell = gs[1, 1:]  # full width across remaining panels
    pos = gs_cell.get_position(fig)  # bounding box of this cell
    # shrink width by 10% on each side
    cax_mod = fig.add_axes([pos.x0 + 0.1, pos.y0, pos.width * 0.6, pos.height])

    cbar_mod = fig.colorbar(
        ims[1],
        cax=cax_mod,
        orientation='horizontal'
    )


    for cb in (cbar_ref, cbar_mod):
        cb.ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        for i, label in enumerate(cb.ax.get_xticklabels()):
            if i % 2 == 1:
                label.set_visible(False)


    for cb in (cbar_ref, cbar_mod):
        # Access the bins from BoundaryNorm
        if isinstance(cb.norm, mcolors.BoundaryNorm):
            boundaries = cb.norm.boundaries  # the levels
            # Compute centers of each bin
            tick_locs = 0.5 * (boundaries[:-1] + boundaries[1:])
            #cb.set_ticks(tick_locs_to_show)
            tick_locs_to_show = tick_locs[::2]
            cb.set_ticks(tick_locs_to_show)

            # Set tick labels as integers
            #cb.set_ticklabels(np.arange(len(tick_locs)))

            cb.ax.xaxis.set_minor_locator(plt.NullLocator())
            # Label every 4th tick
            #tick_labels = [str(i) if (idx % 4 == 0) else ''
            #               for idx, i in enumerate(np.arange(len(tick_locs)))]

            tick_labels = [str(int(i)) for i in np.arange(len(tick_locs_to_show))]

            cb.set_ticklabels(tick_labels)
            cb.ax.tick_params(labelsize=7, direction='in', length=1.5)



    fig.suptitle(f"{var_name} (in days) {window_str} day forecast", fontsize=15)
    plt.tight_layout(rect=[0, 0, 0.99, 1])
    plt.show()

    plot_filename = f"map_{var_name}_{model_name}_{window_str}.png"
    plot_path = os.path.join(cfg.dir_fig, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {plot_path}")





























"""Original twelve-trap Counter figure, using revision predictions and cached basemap."""
from pathlib import Path
from AIedes.utils.counter_experiment import read_json


def make_timeseries(output, destination=None):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import geopandas as gpd
    destination = destination or output / 'figures'
    config = read_json(output / 'experiment.json')['config']
    frames = []
    for dataset, data_key, prediction_file, prediction_col, ids in [
        ('AIMSurv', 'weekly_data', 'evaluation/main_predictions.csv', 'prediction', [363,362,279,448,456,376,221,1244,1260]),
        ('VectorNet', 'external_data', 'reports/vectornet_predictions.csv', 'main', [2002,2036,2054]),
    ]:
        raw = pd.read_pickle(config[data_key]).reset_index(drop=True)
        raw['row_id'] = np.arange(len(raw))
        predictions = pd.read_csv(output / prediction_file)
        joined = predictions.merge(raw[['row_id','id_trap','end_date','weeklyRates','decimalLatitude','decimalLongitude']],
                                   on='row_id', suffixes=('', '_raw'), validate='one_to_one')
        if len(joined) != len(predictions):
            raise ValueError('Raw coordinates and prediction rows disagree')
        # VectorNet identifiers include fractions; CSV parsing can change the last bit.
        np.testing.assert_allclose(joined.id_trap, joined.id_trap_raw, rtol=0, atol=1e-10)
        np.testing.assert_allclose(joined.target, joined.weeklyRates)
        if not (pd.to_datetime(joined.date) == pd.to_datetime(joined.end_date)).all():
            raise ValueError('Prediction and observation dates disagree')
        joined = joined.loc[joined.id_trap.isin(ids)].copy()
        if set(joined.id_trap) != set(ids):
            raise ValueError(f'Missing original {dataset} traps')
        joined['dataset'] = dataset
        if dataset == 'VectorNet':
            joined['country'] = 'Italy'
        joined['split'] = 'test'
        joined = joined.rename(columns={'decimalLatitude':'lat','decimalLongitude':'lon', 'target':'target_data', prediction_col:'predictions'})
        joined['end_date'] = pd.to_datetime(joined.end_date)
        frames.append(joined)
    data_egg_most_populatet = pd.concat(frames, ignore_index=True)
    data_egg_most_populatet[['dataset','id_trap','country','end_date','target_data','predictions','lat','lon']].to_csv(destination / 'timeseries_predictions.csv', index=False)
    locations = data_egg_most_populatet.groupby(['dataset','id_trap'], sort=True).first().reset_index()
    locations[['dataset','id_trap','country','lat','lon']].to_csv(destination / 'timeseries_traps.csv', index=False)
    gdf_most_populatet_grouped_mercator = gpd.GeoDataFrame(locations, geometry=gpd.points_from_xy(locations.lon, locations.lat), crs=4326).to_crs(3857)
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    from matplotlib.transforms import blended_transform_factory as blend
    import matplotlib.patheffects as pe
    import pycountry

    # --- CONFIG --------------------------------------------------------------------
    START, END = pd.Timestamp("2020-06-01"), pd.Timestamp("2020-12-31")
    month_ticks  = pd.date_range("2020-06-01", "2020-12-01", freq="MS")
    month_labels = [d.strftime("%b") for d in month_ticks]

    # --- HELPERS -------------------------------------------------------------------

    curve_styles = {
        "actual": {
            "color": "#1f77b4",   # matplotlib blue
            "linestyle": "-",
            "linewidth": 1.8,
            "fill": False,         # fill below the line
            "fill_alpha": 0.2
        },
        "predicted": {
            "color": "#ff7f0e",   # matplotlib orange
            "linestyle": "--",
            "linewidth": 2.2,
            "fill": False
        }
    }

    # Marker styles for train/test data - removed differentiation
    marker_styles = {
        "default": {"marker": "o", "markersize": 4}
    }

    def get_country_abbreviation(country_name):
        try:
            # Get the official alpha_2 country code
            return country_name
            #return pycountry.countries.lookup(country_name).alpha_2
        except LookupError:
            # Fallback to the first two letters if the country is not found
            return country_name[:2]

    def series_to_fake_2020(df, col, window=3):
        wk = (df["end_date"].dt.isocalendar().week.astype(int)
              .rename("week").to_frame().join(df[[col, "split"]])
              .groupby("week").agg({col: 'mean', 'split': lambda x: list(x)}).sort_index())
        idx = pd.to_datetime([pd.Timestamp.fromisocalendar(2020, int(w), 1) for w in wk.index])
        wk.index = idx
        wk = wk[(wk.index >= START) & (wk.index <= END)]

        # Extract the series and split info
        wk_series = wk[col]
        wk_split = wk['split']

        return (wk_series.rolling(window=window, center=True, min_periods=1).mean(),
                wk_series,
                wk_split)

    def mini_y_axis(ax):
        ymin, ymax = ax.get_ylim()
        ymax2 = int(ymax / 2)
        ax.yaxis.set_major_locator(mticker.FixedLocator([0, ymax2]))
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
        ax.tick_params(axis='y', length=0, pad=2)
        ax.spines['left'].set_visible(False)
        # short vertical line at x=0 (axes coords), from 0 to ymax2 (data coords)
        ax.plot([0.0, 0.0], [0, ymax2], transform=blend(ax.transAxes, ax.transData),
                color='black', lw=0.8, clip_on=False)

    def plot_points_by_split(ax, index, values, split_info, color, alpha=0.6, zorder=4):
        """Plot points with the same marker regardless of split"""
        for i, (date, value, splits) in enumerate(zip(index, values, split_info)):
            # Use default marker style for all points
            marker_style = marker_styles["default"]

            ax.plot(date, value,
                    marker=marker_style["marker"],
                    linestyle="",
                    color=color,
                    markersize=marker_style["markersize"],
                    alpha=alpha,
                    zorder=zorder)

    def plot_trap(ax, df, trap_id, y_max):
        targ, all_targ, split_targ = series_to_fake_2020(df, "target_data", window=3)
        pred, all_pred, split_pred = series_to_fake_2020(df, "predictions", window=3)

        # --- Actual curve ---
        s = curve_styles["actual"]
        if s["fill"]:
            ax.fill_between(targ.index, targ.values, 0,
                            color=s["color"], alpha=s["fill_alpha"], zorder=1)
        line1, = ax.plot(targ.index, targ.values,
                         color=s["color"], linestyle=s["linestyle"],
                         lw=s["linewidth"], label="Actual (moving avg)", zorder=2)

        # Plot actual points with default markers
        plot_points_by_split(ax, all_targ.index, all_targ.values, split_targ.values,
                            s["color"], alpha=0.6, zorder=4)

        # Add white contour
        line1.set_path_effects([
            pe.Stroke(linewidth=s["linewidth"]+2, foreground='white'),
            pe.Normal()
        ])

        # --- Predicted curve ---
        s = curve_styles["predicted"]
        if s["fill"]:
            ax.fill_between(pred.index, pred.values, 0,
                            color=s["color"], alpha=s["fill_alpha"], zorder=1)
        line2, = ax.plot(pred.index, pred.values,
                         color=s["color"], linestyle=s["linestyle"],
                         lw=s["linewidth"], label="Predicted (moving avg)", zorder=3)

        # Plot predicted points with default markers
        plot_points_by_split(ax, all_pred.index, all_pred.values, split_pred.values,
                            s["color"], alpha=0.6, zorder=5)

        # Add white contour
        line2.set_path_effects([
            pe.Stroke(linewidth=s["linewidth"]+2, foreground='white'),
            pe.Normal()
        ])
        # cosmetics
        ax.set_xlim(START, END)
        # Preserve original scale unless revised weekly points would be clipped.
        maximum = max(float(all_targ.max()), float(all_pred.max()))
        if maximum > y_max:
            y_max = np.ceil(maximum * 1.05 / 20) * 20
        ax.set_ylim(-0.1*y_max, y_max)
        ax.grid(False)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.patch.set_alpha(0.0)  # transparent subplot background

        # inline label (right-center) - use standard black color
        country = (df["country"].iloc[0].capitalize() if "country" in df.columns and len(df) else "Unk")
        dataset = (df["dataset"].iloc[0] if "dataset" in df.columns and len(df) else "?")

        ax.text(0.97, 0.5, f"{get_country_abbreviation(country)} [{dataset[:3]}]",
                transform=ax.transAxes,
                ha='right', va='center',
                fontsize=9, fontweight='bold',
                color='black')  # Use standard black color

        # minimal y axis
        mini_y_axis(ax)

    # --- DATA SELECTION ------------------------------------------------------------
    aim_ids = (data_egg_most_populatet
               .loc[data_egg_most_populatet["dataset"]=="AIMSurv", ["id_trap","country"]]
               .drop_duplicates().sort_values("country")["id_trap"].values[:9])
    vec_ids = (data_egg_most_populatet
               .loc[data_egg_most_populatet["dataset"]=="VectorNet", ["id_trap","country"]]
               .drop_duplicates().sort_values("country")["id_trap"].values[:3])

    # --- FIGURE --------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 7))
    # 1 row, 2 cols
    outer = fig.add_gridspec(1, 2, width_ratios=[4, 1.3], wspace=0.03)

    # Left: time series grid
    left = outer[0].subgridspec(6, 2, hspace=0.0, wspace=0.05)
    axes = np.array([[fig.add_subplot(left[r, c]) for c in range(2)] for r in range(6)])

    # Right: map, bottom-aligned and with fixed aspect
    ax_map = fig.add_subplot(outer[1])
    ax_map.set_aspect('auto')      # remove fixed aspect constraint
    ax_map.set_box_aspect(None)    # (undo any previous set_box_aspect)
    ax_map.set_anchor('S')         # still bottom-align

    # PLOT TIMESERIES ON LEFT PANEL
    # draw
    mult = -0.10
    for ax, tid in zip(axes[0:3, 0], aim_ids[:3]):
        df = data_egg_most_populatet.loc[data_egg_most_populatet["id_trap"]==tid]
        plot_trap(ax, df, tid, y_max=500)
    for ax, tid in zip(axes[3:-1, 0], vec_ids):
        df = data_egg_most_populatet.loc[data_egg_most_populatet["id_trap"]==tid]
        plot_trap(ax, df, tid, y_max=400)
    ax = axes[-1, 0]
    df = data_egg_most_populatet.loc[data_egg_most_populatet["id_trap"]==vec_ids[-1]]
    plot_trap(ax, df, tid, y_max=120)

    for ax, tid in zip(axes[:, 1], aim_ids[3:]):
        df = data_egg_most_populatet.loc[data_egg_most_populatet["id_trap"]==tid]
        plot_trap(ax, df, tid, y_max=120)

    # global x ticks only on last row
    for ax in axes.ravel():
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
    for ax in axes[-1, :]:
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels)
        ax.tick_params(axis='x', labelbottom=True, labelsize=9)

    ## PLOT MAP ON RIGHT PANEL

    # Clean axes
    ax_map.set_xticks([]); ax_map.set_yticks([])
    ax_map.set_xlabel(''); ax_map.set_ylabel('')

    # Europe bounds → Web Mercator
    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    min_x, min_y = transformer.transform(-5, 30)   # SW
    max_x, max_y = transformer.transform(40, 60)   # NE
    ax_map.set_xlim(min_x, max_x); ax_map.set_ylim(min_y, max_y)

    gdf = gdf_most_populatet_grouped_mercator.copy()

    # Get bounds of your points in Web Mercator
    xmin, ymin, xmax, ymax = gdf.total_bounds

    # Add padding (in meters; tweak factor=0.1 → 10% margin)
    pad_x = (xmax - xmin) * 0.1
    pad_y = (ymax - ymin) * 0.01

    ax_map.set_xlim(xmin - pad_x, xmax + pad_x)
    ax_map.set_ylim(ymin - pad_y, ymax + pad_y)

    # ensure Web Mercator for points + basemap
    if gdf.crs is None or gdf.crs.to_epsg() != 3857:
        gdf = gdf.to_crs(3857)

    # basemap first (fixed zoom; keep your limits)
    assets = Path(__file__).parent / 'assets'
    basemap = read_json(assets / 'legacy_trap_basemap.json')
    ax_map.imshow(plt.imread(assets / 'legacy_trap_basemap-000.png'), extent=basemap['extent'], interpolation='bilinear')
    ax_map.set_xlim(basemap['extent'][:2]); ax_map.set_ylim(basemap['extent'][2:])
    ax_map.text(.005,.005,basemap['attribution'],transform=ax_map.transAxes,fontsize=8,va='bottom',zorder=10)

    # PLOT POINTS using standard blue color
    gdf.plot(ax=ax_map,
             color='#238b45',              # Standard matplotlib blue
             markersize=50, alpha=0.9,
             edgecolor="black", linewidth=0.5,
             legend=False)

    ax_map.set_title('Trap locations', fontsize=12, fontweight='bold')

    # bold axis labels (apply only to the left panel)
    fig.supxlabel("Month", fontsize=12, fontweight="bold", x=0.4, y=0.05)  # shift left
    fig.supylabel("Weekly Egg-laying Rate", fontsize=12, fontweight="bold", x=0.08)

    # Create comprehensive legend including marker points
    from matplotlib.lines import Line2D

    # Get original legend handles and labels (for line types)
    handles, labels = axes[0,0].get_legend_handles_labels()

    # Add marker legend entries for actual and predicted points
    actual_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
                          markersize=4, alpha=0.6, linestyle='', label='Actual (weekly)')
    predicted_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e',
                             markersize=4, alpha=0.6, linestyle='', label='Predicted (weekly)')

    # Combine all handles and labels
    all_handles = handles + [actual_marker, predicted_marker]
    all_labels = labels + ['Actual data points', 'Predicted data points']

    # shared legend (place between left plots and right map)
    fig.legend(all_handles, all_labels,
               loc="center left",
               bbox_to_anchor=(0.7, 0.7),   # adjust anchor closer to map
               frameon=False,
               #fontsize="Large",
               #title="Series Types\n[the y-axis is not shared]",
               # the title is centered over two lines
               #fontsize="small"
               )

    #plt.subplots_adjust(wspace=0.05, hspace=0.0)
    plt.savefig(destination / "most_populated_traps_time_series.pdf", bbox_inches="tight")
    plt.savefig(destination / "most_populated_traps_time_series.png", bbox_inches="tight", dpi=180)
    plt.close(fig)

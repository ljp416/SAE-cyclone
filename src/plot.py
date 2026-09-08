"""One feature map: pressure anomaly, sparse activation and cyclone target."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter
from matplotlib.tri import Triangulation


def plot_feature(activation, physical, coastlines, *, title, lon_range=None, lat_range=None):
    """Display real mesh values; changing the map crop does not change F1 ranking."""
    lon = (physical['longitude'] + 180) % 360 - 180
    lat = physical['latitude']
    target = physical['target'].astype(bool)
    anomaly = physical['mslp_anomaly']
    west, east, south, north = physical['map_bounds']
    west, east = lon_range if lon_range is not None else (west, east)
    south, north = lat_range if lat_range is not None else (south, north)
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        raise ValueError('Use increasing longitude and latitude ranges without crossing the dateline.')
    frame = (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
    if frame.sum() < 3:
        raise ValueError('The map range contains too few mesh nodes.')
    tri = Triangulation(lon[frame], lat[frame])
    tx, ty = tri.x[tri.triangles], tri.y[tri.triangles]
    tri.set_mask(np.max(np.hypot(tx - np.roll(tx, 1, axis=1), ty - np.roll(ty, 1, axis=1)), axis=1) > 4)
    limit = max(float(np.quantile(np.abs(anomaly[frame]), 0.99)), 1e-6)
    cmap = LinearSegmentedColormap.from_list('pressure', ['#2b5f8e', '#f4f6f3', '#b03d36'])
    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    pressure = ax.tripcolor(tri, anomaly[frame], shading='flat', cmap=cmap,
                            norm=TwoSlopeNorm(0, -limit, limit), rasterized=True)
    ax.tricontour(tri, anomaly[frame], levels=np.linspace(-limit, limit, 9),
                  colors='#33424b', linewidths=0.4, alpha=0.5)
    offsets, vertices = coastlines['offsets'], coastlines['vertices']
    segments = [vertices[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    ax.add_collection(LineCollection(segments, colors='#1c2730', linewidths=0.6))
    active = frame & (activation > 0)
    points = ax.scatter(lon[active], lat[active], c=activation[active], s=23,
                        cmap='viridis', vmin=0, vmax=0.5, zorder=4)
    visible_target = target & frame
    for color, width, label in [('white', 2.5, None), ('black', 0.9, 'Cyclone target')]:
        ax.scatter(lon[visible_target], lat[visible_target], s=60, facecolors='none',
                   edgecolors=color, linewidths=width, label=label, zorder=5)
    ax.set(xlim=(west, east), ylim=(south, north), title=title, xlabel='Longitude', ylabel='Latitude')
    ax.set_aspect(1 / np.cos(np.deg2rad((south + north) / 2)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{abs(x):g}°' + ('W' if x < 0 else 'E')))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{abs(y):g}°' + ('S' if y < 0 else 'N')))
    ax.legend(loc='lower right', fontsize=9)
    fig.colorbar(pressure, ax=ax, shrink=0.8, pad=0.02, label='MSLP anomaly (hPa)', extend='both')
    fig.colorbar(points, ax=ax, shrink=0.8, pad=0.02, label='SAE activation',
                 extend='max' if np.any(activation[active] > 0.5) else 'neither')
    return fig

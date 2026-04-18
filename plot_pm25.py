"""
PM2.5 Spatial Distribution Heatmap for India (February–March 2023)
===================================================================
Reads two monthly NetCDF files, computes the Feb–Mar average, applies
linear and nearest-neighbor spatial interpolation to fill minor gaps,
and produces a publication-quality PNG with India district boundaries
overlaid.

Output: pm25_india_feb_mar_2023.png
"""

import os
import warnings
import urllib.request

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
import matplotlib.font_manager as fm
import xarray as xr
import geopandas as gpd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from shapely import contains_xy

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Chinese font setup – use Noto Sans CJK if available, else fall back to
# the default sans-serif (Chinese characters may render as boxes in that case)
# ---------------------------------------------------------------------------
_cjk_candidates = ["Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Serif CJK SC",
                    "Noto Serif CJK JP", "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei"]
_available = {f.name for f in fm.fontManager.ttflist}
_cjk_font = next((f for f in _cjk_candidates if f in _available), None)
if _cjk_font:
    plt.rcParams["font.family"] = _cjk_font
    print(f"Using CJK font: {_cjk_font}")
else:
    print("No CJK font found – Chinese characters may not render correctly.")
plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

NC_FEB = os.path.join(SCRIPT_DIR, "India_202302-202302.nc")
NC_MAR = os.path.join(SCRIPT_DIR, "India_202303-202303.nc")
GEOJSON = os.path.join(SCRIPT_DIR, "INDIA_INDIA_DISTRICTS.geojson")
OUTPUT  = os.path.join(SCRIPT_DIR, "pm25_india_feb_mar_2023.png")

# ---------------------------------------------------------------------------
# 1. Read & average NetCDF data
# ---------------------------------------------------------------------------
print("Reading NetCDF files …")
ds_feb = xr.open_dataset(NC_FEB)
ds_mar = xr.open_dataset(NC_MAR)

pm25_feb = ds_feb["PM25"].values   # shape (lat, lon)
pm25_mar = ds_mar["PM25"].values

lat = ds_feb["lat"].values          # 1-D, ascending (6.8 … 35.4)
lon = ds_feb["lon"].values          # 1-D, ascending (68.2 … 97.2)

ds_feb.close()
ds_mar.close()

# Replace NaN / masked fill values
pm25_feb = np.where(np.isfinite(pm25_feb) & (pm25_feb > 0), pm25_feb, np.nan)
pm25_mar = np.where(np.isfinite(pm25_mar) & (pm25_mar > 0), pm25_mar, np.nan)

# Average over the two months (NaN-safe)
pm25_avg = np.nanmean(np.stack([pm25_feb, pm25_mar], axis=0), axis=0)

# Build a land mask: True where at least one month has valid data
land_mask = np.isfinite(pm25_avg)

print(f"  Grid size : {lat.size} × {lon.size}")
print(f"  Valid cells: {land_mask.sum():,}")
print(f"  PM2.5 range: {np.nanmin(pm25_avg):.1f} – {np.nanmax(pm25_avg):.1f} μg/m³")
print(f"  PM2.5 mean : {np.nanmean(pm25_avg):.1f} μg/m³")

# ---------------------------------------------------------------------------
# 2. Spatial interpolation (linear + nearest-neighbor) to fill isolated NaN gaps on land
# ---------------------------------------------------------------------------
print("Applying spatial interpolation …")

LON2, LAT2 = np.meshgrid(lon, lat)

# Collect valid (source) points
src_mask = land_mask & np.isfinite(pm25_avg)
src_lon  = LON2[src_mask]
src_lat  = LAT2[src_mask]
src_val  = pm25_avg[src_mask]

# Interpolate onto the full grid using linear then nearest fallback
pm25_interp = griddata(
    (src_lon, src_lat), src_val,
    (LON2, LAT2), method="linear"
)
# Fill any remaining NaN inside land mask with nearest-neighbor
pm25_nn = griddata(
    (src_lon, src_lat), src_val,
    (LON2, LAT2), method="nearest"
)
pm25_interp = np.where(np.isfinite(pm25_interp), pm25_interp, pm25_nn)

# Apply light Gaussian smoothing for visual appeal
pm25_smooth = gaussian_filter(pm25_interp, sigma=1.5)

# Keep smoothed field; final masking will be applied after boundary GeoJSON is loaded
pm25_plot = pm25_smooth.copy()

# ---------------------------------------------------------------------------
# 3. Color map – China-style rainbow color bar
# ---------------------------------------------------------------------------
VMIN, VMAX = 0, 80   # μg/m³ – same axis range as the reference China map

# Use classic China-map rainbow (blue→cyan→green→yellow→orange→red)
cmap = plt.get_cmap("jet")
norm = mcolors.Normalize(vmin=VMIN, vmax=VMAX)

# ---------------------------------------------------------------------------
# 4. Load district boundaries (GeoJSON) – download from public source if local
#    file is a placeholder (< 10 kB); cache to /tmp to avoid repeated downloads
# ---------------------------------------------------------------------------
# Public source: GADM-based India district polygons (594 districts, WGS-84)
_DISTRICT_URL   = ("https://raw.githubusercontent.com/geohacker/india/"
                   "master/district/india_district.geojson")
_DISTRICT_CACHE = "/tmp/india_districts_cache.geojson"

india_gdf   = None
use_geojson = False

def _is_real_geojson(path):
    """Return True only if the file looks like a real GeoJSON (≥ 10 kB, starts with '{')."""
    if not os.path.isfile(path):
        return False
    if os.path.getsize(path) < 10_000:
        return False
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read(20).strip().startswith("{")

# Determine which path to use
_geojson_path = None
if _is_real_geojson(GEOJSON):
    _geojson_path = GEOJSON
elif _is_real_geojson(_DISTRICT_CACHE):
    print("Using cached district GeoJSON.")
    _geojson_path = _DISTRICT_CACHE
else:
    print("Downloading India district boundaries …")
    try:
        urllib.request.urlretrieve(_DISTRICT_URL, _DISTRICT_CACHE)
        print(f"  Saved to {_DISTRICT_CACHE}")
        _geojson_path = _DISTRICT_CACHE
    except Exception as exc:
        print(f"  Download failed ({exc}) – will use data-derived boundary.")

if _geojson_path:
    try:
        india_gdf = gpd.read_file(_geojson_path)
        if india_gdf.crs and india_gdf.crs.to_epsg() != 4326:
            india_gdf = india_gdf.to_crs("EPSG:4326")
        use_geojson = True
        print(f"Loaded district GeoJSON: {len(india_gdf)} districts, CRS={india_gdf.crs}")
    except Exception as exc:
        print(f"Could not read district GeoJSON ({exc}) – using data-derived boundary.")

# Build final plot mask from country geometry when available to avoid
# white gaps between raster color and boundary line.
if use_geojson and india_gdf is not None:
    try:
        country_geom = india_gdf.union_all()
    except AttributeError:
        country_geom = india_gdf.geometry.unary_union

    # Expand by ~half grid cell so colored raster reaches the plotted boundary.
    _dlat = float(np.median(np.diff(lat)))
    _dlon = float(np.median(np.diff(lon)))
    _buffer = 0.5 * np.hypot(_dlat, _dlon)
    country_geom = country_geom.buffer(_buffer)

    country_mask = contains_xy(country_geom, LON2, LAT2)
else:
    country_mask = land_mask

pm25_plot = np.where(country_mask, pm25_plot, np.nan)

# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------
print("Rendering map …")

fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
fig.patch.set_facecolor("white")
ax.set_facecolor("#ddeeff")          # light ocean colour

# --- PM2.5 heatmap ---
mesh = ax.pcolormesh(
    LON2, LAT2, pm25_plot,
    cmap=cmap, norm=norm,
    shading="auto", zorder=2
)

# --- Boundaries ---
if use_geojson and india_gdf is not None:
    # Draw only admin-2 (city/county-level) boundaries as a single merged layer
    # to avoid mixed thick/thin appearance caused by overlapping segments.
    # Simplify linework slightly to reduce visual clutter while preserving
    # city-level boundaries.
    try:
        linework = india_gdf.boundary.union_all()
        linework = linework.simplify(0.02, preserve_topology=True)
        gpd.GeoSeries([linework], crs="EPSG:4326").plot(
            ax=ax, linewidth=0.10, color="#222222", alpha=0.70, zorder=4
        )
    except AttributeError:
        # older geopandas compatibility
        linework = india_gdf.boundary.unary_union
        linework = linework.simplify(0.02, preserve_topology=True)
        gpd.GeoSeries([linework], crs="EPSG:4326").plot(
            ax=ax, linewidth=0.10, color="#222222", alpha=0.70, zorder=4
        )
else:
    # Draw boundary derived from the land mask using contour at 0.5
    # (binary mask: 1 = land, 0 = ocean)
    _mask_float = land_mask.astype(float)
    _mask_smooth = gaussian_filter(_mask_float, sigma=1.0)
    ax.contour(
        LON2, LAT2, _mask_smooth,
        levels=[0.5],
        colors=["#111111"], linewidths=[1.0], zorder=5
    )

# --- Map extent & ticks ---
ax.set_xlim(lon.min() - 0.5, lon.max() + 0.5)
ax.set_ylim(lat.min() - 0.5, lat.max() + 0.5)

ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(1))

ax.set_xlabel("Longitude (°E)", fontsize=11, labelpad=6)
ax.set_ylabel("Latitude (°N)",  fontsize=11, labelpad=6)

ax.tick_params(axis="both", which="major", labelsize=9,
               direction="in", length=5, width=0.8)
ax.tick_params(axis="both", which="minor",
               direction="in", length=3, width=0.5)

# Format tick labels with °E / °N
def fmt_lon(x, _):
    return f"{x:.0f}°E"

def fmt_lat(x, _):
    return f"{x:.0f}°N"

ax.xaxis.set_major_formatter(ticker.FuncFormatter(fmt_lon))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt_lat))

for spine in ax.spines.values():
    spine.set_linewidth(0.8)

# --- Color bar (horizontal, below map – matching reference style) ---
cbar = fig.colorbar(
    mesh, ax=ax,
    orientation="horizontal",
    fraction=0.04, pad=0.08,
    extend="max",
    aspect=40,
)
cbar.set_label("2–3月平均 PM2.5 (μg/m³)", fontsize=11, labelpad=6)
cbar.ax.tick_params(labelsize=9)
# Ticks at 0, 10, 20, … 80 – exactly as in the reference China map
cbar_ticks = list(range(0, VMAX + 1, 10))
cbar.set_ticks(cbar_ticks)
cbar.ax.set_xticklabels([str(v) for v in cbar_ticks])

# --- Title ---
ax.set_title(
    "2023年2–3月印度 PM2.5 空间分布图\n"
    "(February – March 2023 Average)",
    fontsize=14, fontweight="bold", pad=14
)

# --- Annotation ---
ax.text(
    0.99, 0.01,
    "Data: CHAP-V2  |  Resolution: 0.1°",
    transform=ax.transAxes,
    fontsize=6.5, color="#555555",
    ha="right", va="bottom",
)

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
plt.tight_layout()
fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved → {OUTPUT}")

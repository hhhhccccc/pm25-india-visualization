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

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator
import xarray as xr
import geopandas as gpd
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

warnings.filterwarnings("ignore")

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
# Fill any remaining NaN inside land mask with nearest-neighbour
pm25_nn = griddata(
    (src_lon, src_lat), src_val,
    (LON2, LAT2), method="nearest"
)
pm25_interp = np.where(np.isfinite(pm25_interp), pm25_interp, pm25_nn)

# Apply light Gaussian smoothing for visual appeal
pm25_smooth = gaussian_filter(pm25_interp, sigma=1.5)

# Restore NaN outside the land mask
pm25_plot = np.where(land_mask, pm25_smooth, np.nan)

# ---------------------------------------------------------------------------
# 3. Custom colour map  blue → green → yellow → red
# ---------------------------------------------------------------------------
LEVELS   = [0, 10, 30, 60, 100, 150]
COLOURS  = ["#1a80ff", "#00cc44", "#ffdd00", "#ff4400", "#990000"]

cmap = mcolors.LinearSegmentedColormap.from_list(
    "pm25",
    list(zip(
        np.linspace(0, 1, len(COLOURS)),
        COLOURS
    ))
)
norm = mcolors.BoundaryNorm(LEVELS, cmap.N)

# ---------------------------------------------------------------------------
# 4. Load district boundaries (GeoJSON) – graceful fallback
# ---------------------------------------------------------------------------
india_gdf   = None
use_geojson = False

if os.path.isfile(GEOJSON):
    try:
        # Check if it is a real GeoJSON (not a placeholder)
        with open(GEOJSON, "r", encoding="utf-8") as fh:
            first_chars = fh.read(20).strip()
        if first_chars.startswith("{") or first_chars.startswith("["):
            india_gdf = gpd.read_file(GEOJSON)
            use_geojson = True
            print(f"Loaded GeoJSON: {len(india_gdf)} districts, CRS={india_gdf.crs}")
        else:
            print("GeoJSON file appears to be a placeholder – using data-derived boundary.")
    except Exception as exc:
        print(f"Could not read GeoJSON ({exc}) – using data-derived boundary.")
else:
    print("GeoJSON not found – using data-derived boundary.")

# ---------------------------------------------------------------------------
# 5. Plot
# ---------------------------------------------------------------------------
print("Rendering map …")

fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
fig.patch.set_facecolor("#f8f8f8")
ax.set_facecolor("#b0d4f0")          # ocean colour

# --- PM2.5 heatmap ---
mesh = ax.pcolormesh(
    LON2, LAT2, pm25_plot,
    cmap=cmap, norm=norm,
    shading="auto", zorder=2
)

# --- Contour lines for extra visual structure ---
contour_levels = [10, 30, 60, 100]
cs = ax.contour(
    LON2, LAT2, pm25_plot,
    levels=contour_levels,
    colors="white", linewidths=0.4, alpha=0.5, zorder=3
)

# --- Boundaries ---
if use_geojson and india_gdf is not None:
    # Reproject to WGS-84 if needed
    if india_gdf.crs and india_gdf.crs.to_epsg() != 4326:
        india_gdf = india_gdf.to_crs("EPSG:4326")
    india_gdf.boundary.plot(
        ax=ax, linewidth=0.4, edgecolor="#333333", zorder=4
    )
    # Outer country border (union of all districts)
    try:
        outer = india_gdf.union_all()
        gpd.GeoSeries([outer]).boundary.plot(
            ax=ax, linewidth=1.2, edgecolor="#111111", zorder=5
        )
    except AttributeError:
        # older geopandas
        outer = india_gdf.geometry.unary_union
        gpd.GeoSeries([outer]).boundary.plot(
            ax=ax, linewidth=1.2, edgecolor="#111111", zorder=5
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

# --- Colour bar ---
cbar = fig.colorbar(
    mesh, ax=ax,
    orientation="vertical",
    fraction=0.025, pad=0.02,
    extend="max",
    ticks=LEVELS,
)
cbar.set_label("PM₂.₅ Concentration (μg/m³)", fontsize=10, labelpad=8)
cbar.ax.tick_params(labelsize=9)
cbar.ax.set_yticklabels([str(v) for v in LEVELS])

# Custom legend patches
legend_items = [
    mpatches.Patch(color=COLOURS[0], label="0–10 μg/m³  (Low)"),
    mpatches.Patch(color=COLOURS[1], label="10–30 μg/m³ (Moderate-Low)"),
    mpatches.Patch(color=COLOURS[2], label="30–60 μg/m³ (Moderate-High)"),
    mpatches.Patch(color=COLOURS[3], label="60–100 μg/m³ (High)"),
    mpatches.Patch(color=COLOURS[4], label=">100 μg/m³  (Very High)"),
]
ax.legend(
    handles=legend_items,
    loc="lower left",
    fontsize=7.5,
    framealpha=0.85,
    edgecolor="gray",
    title="PM₂.₅ Level",
    title_fontsize=8,
)

# --- Title ---
ax.set_title(
    "India PM₂.₅ Concentration Spatial Distribution\n"
    "February – March 2023 (Monthly Average)",
    fontsize=13, fontweight="bold", pad=12
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

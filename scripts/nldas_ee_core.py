#Open one daily NLDAS NetCDF file.
#Find the four NLDAS variables DroughtWatch needs.
#Prepare each raster so exactextract can read it correctly.
#Calculate coverage-weighted county statistics.

import numpy as np
import xarray as xr
import rioxarray 
from exactextract import exact_extract



#NLDAS grid information

NLDAS_RESOLUTION = 0.125
NLDAS_WEST_CENTER = -124.9375
NLDAS_SOUTH_CENTER = 25.0625
NLDAS_CRS = "EPSG:4326"

ROOT_SURFACE_RATIO_MIN = 5.0



#variable names
#different NLDAS files can use slightly different capitalization
VARIABLE_CANDIDATES = {
    "soilm_surface": [
        "SOILM_0_10CM",
        "SOILM_0_10cm",
        "SoilM_0_10cm",
    ],
    "soilm_root_0_100": [
        "SOILM_0_100CM",
        "SOILM_0_100cm",
        "SoilM_0_100cm",
    ],
    "et": [
        "EVAP",
        "Evap",
        "et",
        "ET",
    ],
}


def find_variables(dataset):
    #match logical names to the variable names in one NetCDF file
    found = {}

    for logical_name, candidates in VARIABLE_CANDIDATES.items():
        actual_name = next(
            (name for name in candidates if name in dataset.data_vars),
            None,
        )

        if actual_name is None:
            raise ValueError(
                f"Could not find NLDAS variable '{logical_name}'. "
                f"Available variables: {list(dataset.data_vars)}"
            )

        found[logical_name] = actual_name

    return found



#open one daily NLDAS file and identify the variables we need
def open_nldas_day(nc_path):
    dataset = xr.open_dataset(
        nc_path,
        mask_and_scale=True,
    )

    variable_map = find_variables(dataset)

    return dataset, variable_map



#prepare a raster for exactextract
def _spatial_dimensions(data_array):
    #find the longitude/x and latitude/y dimension names
    x_name = next(
        (name for name in ["x", "lon", "longitude"] if name in data_array.dims),
        None,
    )

    y_name = next(
        (name for name in ["y", "lat", "latitude"] if name in data_array.dims),
        None,
    )

    if x_name is None or y_name is None:
        raise ValueError(
            f"Could not find spatial dimensions in {data_array.dims}"
        )

    return x_name, y_name


def sanitize_da(dataset, variable_name):
    #return one NLDAS variable as a geospatial DataArray
    #removes a time dimension if present
    #renames longitude/latitude to x/y
    #replaces extreme fill values with NaN
    #assigns EPSG:4326
    
    data = dataset[variable_name]
    x_name, y_name = _spatial_dimensions(data)

    #daily files sometimes still have a one-element time dimension
    for dimension in list(data.dims):
        if (
            dimension not in [x_name, y_name]
            and data.sizes[dimension] == 1
        ):
            data = data.squeeze(dimension, drop=True)

    extra_dimensions = [
        dimension
        for dimension in data.dims
        if dimension not in [x_name, y_name]
    ]

    if extra_dimensions:
        raise ValueError(
            f"{variable_name} still has non-spatial dimensions: "
            f"{extra_dimensions}"
        )

    rename = {}

    if x_name != "x":
        rename[x_name] = "x"

    if y_name != "y":
        rename[y_name] = "y"

    if rename:
        data = data.rename(rename)

    #prevents very large sentinel values from being treated as real measurements
    data = data.where(np.abs(data) < 1e19)

    data = data.rio.set_spatial_dims(
        x_dim="x",
        y_dim="y",
        inplace=False,
    )

    if data.rio.crs is None:
        data = data.rio.write_crs(NLDAS_CRS)

    return data.rename(variable_name)



#check that the raster is on the NLDAS grid
def check_grid(data_array):
    
    #check NLDAS grid spacing and center registration
    x = np.asarray(data_array["x"].values, dtype=float)
    y = np.asarray(data_array["y"].values, dtype=float)

    x_resolution = float(np.abs(np.diff(x)).mean())
    y_resolution = float(np.abs(np.diff(y)).mean())

    problems = []

    if abs(x_resolution - NLDAS_RESOLUTION) > 1e-4:
        problems.append(
            f"x resolution is {x_resolution}, expected {NLDAS_RESOLUTION}"
        )

    if abs(y_resolution - NLDAS_RESOLUTION) > 1e-4:
        problems.append(
            f"y resolution is {y_resolution}, expected {NLDAS_RESOLUTION}"
        )

    x_position = (
        (float(x.min()) - NLDAS_WEST_CENTER)
        / NLDAS_RESOLUTION
    )

    y_position = (
        (float(y.min()) - NLDAS_SOUTH_CENTER)
        / NLDAS_RESOLUTION
    )

    x_offset = abs(x_position - round(x_position))
    y_offset = abs(y_position - round(y_position))

    if x_offset > 1e-3:
        problems.append(
            f"x coordinates are off the NLDAS center grid "
            f"by {x_offset:.3f} cells"
        )

    if y_offset > 1e-3:
        problems.append(
            f"y coordinates are off the NLDAS center grid "
            f"by {y_offset:.3f} cells"
        )

    return {
        "x_resolution": x_resolution,
        "y_resolution": y_resolution,
        "x_min": float(x.min()),
        "x_max": float(x.max()),
        "y_min": float(y.min()),
        "y_max": float(y.max()),
        "problems": problems,
    }



# exactextract helper
def _county_stats(raster, counties, operations, rename_columns):
   #run exactextract once and give the output column names
    result = exact_extract(
        raster,
        counties,
        operations,
        include_cols=["county_fips"],
        output="pandas",
    )

    return result.rename(columns=rename_columns)



#county extraction

#calculate county-level NLDAS statistics for one day
def extract_one_day(
    dataset,
    counties_gdf,
    var_map,
    date_str,
    check_ratio=False,
):
    
    

    #surface soil moisture
    surface = sanitize_da(
        dataset,
        var_map["soilm_surface"],
    )

    surface_stats = _county_stats(
        surface,
        counties_gdf,
        ["mean", "stdev", "count"],
        {
            "mean": "soilm_surface_mean",
            "stdev": "soilm_surface_std",
            "count": "nldas_n_valid_pixels",
        },
    )

   #root-zone moisture
    root = sanitize_da(
        dataset,
        var_map["soilm_root_0_100"],
    )

    root_stats = _county_stats(
        root,
        counties_gdf,
        ["mean", "stdev"],
        {
            "mean": "soilm_root_0_100_mean",
            "stdev": "soilm_root_0_100_std",
        },
    )

    #actual evapotranspiration
    et = sanitize_da(
        dataset,
        var_map["et"],
    )

    et_stats = _county_stats(
        et,
        counties_gdf,
        ["mean"],
        {
            "mean": "et_mean",
        },
    )

    #total raster coverage, including cells whose value is NaN
    #surface raster supplies the grid geometry
    ones = xr.ones_like(surface).rename("ones")
    ones = ones.rio.set_spatial_dims(
        x_dim="x",
        y_dim="y",
        inplace=False,
    )

    if ones.rio.crs is None:
        ones = ones.rio.write_crs(NLDAS_CRS)

    total_stats = _county_stats(
        ones,
        counties_gdf,
        ["count"],
        {
            "count": "nldas_n_total_pixels",
        },
    )

    #merge extraction results into county table
    output = (
        surface_stats
        .merge(root_stats, on="county_fips", validate="one_to_one")
        .merge(et_stats, on="county_fips", validate="one_to_one")
        .merge(total_stats, on="county_fips", validate="one_to_one")
    )

    output.insert(1, "date", date_str)

    return output[
        [
            "county_fips",
            "date",
            "soilm_surface_mean",
            "soilm_surface_std",
            "soilm_root_0_100_mean",
            "soilm_root_0_100_std",
            "et_mean",
            "nldas_n_valid_pixels",
            "nldas_n_total_pixels",
        ]
    ]

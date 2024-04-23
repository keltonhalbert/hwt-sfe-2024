import os
import glob
import logging
import numpy as np
import xarray as xr 
import pandas as pd
import datetime as dt

import nwsspc.sharp.calc.constants as constants
import nwsspc.sharp.calc.interp as interp
import nwsspc.sharp.calc.thermo as thermo
import nwsspc.sharp.calc.parcel as parcel
import nwsspc.sharp.calc.params as params
import nwsspc.sharp.calc.winds as winds
import nwsspc.sharp.calc.layer as layer

## set  up logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)
LOG = logging.getLogger(__name__)

BASE_OUTPUT_DIR = "/raid/efp/se2024/ftp/spc/kthalbert/gfs_soundings/"

def read_grib2(file_name):
    ## We can't open a monolithic grib2 file with xarray 
    ## due to the multi-coordinate nature of grib2 messages. 
    ## We will open a dataset for each coordinate group: pressure data, 
    ## surface data, 2 meter data, and 10 meter data. We'll do some metadata 
    ## manipulation along the way to make things play nicely, and then 
    ## merge everything into a single dataset we can work with for computations.
    ds_pres = xr.open_dataset(
        file_name,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "isobaricInhPa"},
            "indexpath": "",
        },
    )

    ds_sfc = xr.open_dataset(
        file_name,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "surface", "stepType": "instant"},
            "indexpath": "",
        }
    ) 

    ds_2m = xr.open_dataset(
        file_name,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 2},
            "indexpath": "",
        }
    ) 

    ds_10m = xr.open_dataset(
        file_name,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 10},
            "indexpath": "",
        }
    ) 

    ## drop some unecessary/unused metadata
    ds_2m = ds_2m.reset_coords(['heightAboveGround', 'time', 'step'], drop=True).squeeze()
    ds_10m = ds_10m.reset_coords(['heightAboveGround', 'time', 'step'], drop=True).squeeze()
    ds_sfc = ds_sfc.reset_coords(['surface', 'time', 'step'], drop=True).squeeze()
    #ds_pres = ds_pres.reset_coords(['time', 'step'], drop=True).squeeze()

    ## copy over the fields we want from the 2D data
    ds_pres["sp"] = ds_sfc["sp"]
    ds_pres["t2m"] = ds_2m["t2m"]
    ds_pres["d2m"] = ds_2m["d2m"]
    ds_pres["u10"] = ds_10m["u10"]
    ds_pres["v10"] = ds_10m["v10"]

    return ds_pres

def preprocess_ds(ds):
    """
    Preprocess the xarray dataset to get 
    dewpoint temperature, 3D pressure, and
    other necessary variables. 
    """
    ## We convert the whole dataset to float32 representation 
    ## for usage with the sounding library. 
    ds = ds.astype('float32')

    nz, ny, nx = ds["t"].shape 

    ## Tile the pressure levels into a 3D grid
    pres = np.tile(
        ds["isobaricInhPa"].values.flatten()[:, np.newaxis, np.newaxis], 
        (1, ny, nx)
    )

    pres = pres.astype('float32') * constants.HPA_TO_PA
  
    sat_mixr = thermo.mixratio(pres.flatten(), ds["t"].values.flatten())
    sat_mixr = sat_mixr.reshape(nz, ny, nx) 

    mixr = thermo.mixratio(ds["q"].values.flatten()).reshape(nz, ny, nx)
    mixr[mixr < constants.TOL] = constants.TOL

    mixr_2m = thermo.mixratio(
        ds["sp"].values.flatten(), 
        ds["d2m"].values.flatten()
    ).reshape(ds.t2m.shape)

    dwpk = thermo.temperature_at_mixratio(mixr.flatten(), pres.flatten())
    dwpk = dwpk.reshape(nz, ny, nx) 
    ds["pres"] = xr.DataArray(pres, dims=["isobaricInhPa", "latitude", "longitude"])
    ds["dwpk"] = xr.DataArray(dwpk, dims=["isobaricInhPa", "latitude", "longitude"])
    ds["mixr"] = xr.DataArray(mixr, dims=["isobaricInhPa", "latitude", "longitude"])
    ds["mixr_2m"] = xr.DataArray(mixr_2m, dims=["latitude", "longitude"])

    ## Set some metadata 
    ds["pres"].attrs = {
        "units": "Pa",
        "long_name": "Atmospheric Pressure",
    }

    ds["dwpk"].attrs = {
        "units": "degK",
        "long_name": "Dew Point Temperature",
    }

    ds["mixr"].attrs = {
        "units": "none",
        "long_name": "Water Vapor Mixing Ratio",
    }

    ds["mixr_2m"].attrs = {
        "units": "none",
        "long_name": "2 Meter Water Vapor Mixing Ratio",
    }
    return ds

def construct_profile(
        ds_longitude, ds_latitude, init_time, valid_time, 
        ds_psfc,          ds_tsfc, ds_dsfc, ds_msfc, ds_usfc, ds_vsfc, 
        ds_pres, ds_hght, ds_tmpk, ds_dwpk, ds_mixr, ds_uwin, ds_vwin
    ):

    """
    Construct vertical profiles of merged surface and pressure level
    data. Profiles are constructed by discarding any pressure level 
    data that are higher than the surface pressure, and heights are 
    recomputed using the moist hypsometric equation with the virtual 
    temperature correction. Because there is no surface geopoential height 
    data within the grib2 fields, heights are computed as AGL instead of MSL.
    """
    prof = {}
    prof["lat"] = ds_latitude
    prof["lon"] = ds_longitude


    valid_time = dt.datetime.utcfromtimestamp(int(valid_time)/1e9)
    init_time = dt.datetime.utcfromtimestamp(int(init_time)/1e9)
    forecast_hour = (valid_time - init_time).total_seconds()//3600.0
    prof["init"] = init_time
    prof["valid"] = valid_time
    prof["fh"] = forecast_hour
    ## get the indices of below-ground pressure surfaces
    where_below_ground = np.where(ds_pres > ds_psfc)[0]
    if len(where_below_ground) > 0:
        start_idx = where_below_ground[-1]+1
    else:
        start_idx = 0

    if np.abs(ds_pres[start_idx] - ds_psfc) < 0.5:
        start_idx += 1
    ## 1 for surface + the length of the pressure array - the number 
    ## of levels below ground
    new_len = 1 + ds_pres.shape[0] - start_idx 

    ## create some empty arrays to store our merged
    ## profile data on a per-2D gridpoint basis
    prof["pres"] = np.zeros((new_len), dtype="float32")
    prof["hght"] = np.zeros((new_len), dtype="float32")
    prof["tmpk"] = np.zeros((new_len), dtype="float32")
    prof["vtmp"] = np.zeros((new_len), dtype="float32")
    prof["dwpk"] = np.zeros((new_len), dtype="float32")
    prof["mixr"] = np.zeros((new_len), dtype="float32")
    prof["uwin"] = np.zeros((new_len), dtype="float32")
    prof["vwin"] = np.zeros((new_len), dtype="float32")

    prof["pres"][0] = ds_psfc
    prof["tmpk"][0] = ds_tsfc
    prof["dwpk"][0] = ds_dsfc
    prof["mixr"][0] = ds_msfc
    prof["uwin"][0] = ds_usfc
    prof["vwin"][0] = ds_vsfc

    prof["pres"][1:] = ds_pres[start_idx:]
    prof["hght"][1:] = ds_hght[start_idx:]
    prof["tmpk"][1:] = ds_tmpk[start_idx:]
    prof["dwpk"][1:] = ds_dwpk[start_idx:]
    prof["mixr"][1:] = ds_mixr[start_idx:]
    prof["uwin"][1:] = ds_uwin[start_idx:]
    prof["vwin"][1:] = ds_vwin[start_idx:]


    prof["uwin"][:] = prof["uwin"][:]*1.944
    prof["vwin"][:] = prof["vwin"][:]*1.944

    wspd = winds.vector_magnitude(prof["uwin"], prof["vwin"])
    wdir = winds.vector_angle(prof["uwin"], prof["vwin"])

    prof["wspd"] = wspd
    prof["wdir"] = wdir

    prof["vtmp"] = thermo.virtual_temperature(prof["tmpk"], prof["mixr"])
    prof["spfh"] = thermo.specific_humidity(prof["mixr"])
    prof["theta"] = thermo.theta(prof["pres"], prof["tmpk"])

    ## re-compute the heights using the moist hypsometric 
    ## equation since surface geopotential heights are not 
    ## part of the grib2 file. That means heights
    ## are in units of meters AGL and not MSL. 
    tv_bar = (prof["vtmp"][:-1] + prof["vtmp"][1:]) / 2.0
    log_dp = np.log(prof["pres"][:-1] / prof["pres"][1:])
    dz = ((constants.RDGAS * tv_bar)/constants.GRAVITY) * log_dp
    prof["hght"][1:] = np.cumsum(dz)
    prof["hght"] += 0

    ## MSE is used for things like ECAPE
    prof["moist_static_energy"] = \
        thermo.moist_static_energy(
                prof["hght"] - prof["hght"][0], 
                prof["tmpk"], 
                prof["spfh"]
            )

    return prof


def save_txt(prof, output_date, model=None):

    stid = prof["stid"]
    wmo = "{:<06}".format(prof["wmo"])
    valid_time = prof["valid"]
    init_time = prof["init"]
    YYMMDD = valid_time.strftime("%y%m%d")
    HH = valid_time.strftime("%H")
    lat = prof["lat"]
    lon = prof["lon"]
    elev = prof["elev"]
    fh = prof["fh"]

    output_dir = f"{output_date}/"
    output_name = f"gfs_{str(wmo).zfill(6)}.{init_time.strftime('%Y%m%d%H')}.f{str(int(fh)).zfill(3)}.txt"

    total_path = os.path.join(BASE_OUTPUT_DIR, output_dir)
    if not os.path.exists(total_path):
        os.makedirs(total_path)

    header_str = \
    f"""
     SNPARM = PRES;HGHT;TMPC;DWPC;DRCT;SKNT
     STNPRM = SHOW

     STID = {stid}          STNM =   {wmo}   TIME = {YYMMDD}/{HH}00         
     SLAT = {lat}       SLON = {lon}    SELV = {elev}
     STIM = {str(int(fh)).zfill(2)}

     SHOW = -9999

     PRES \t HGHT \t TMPC \t DWPC \t DRCT \t SKNT
    """

    file_path = os.path.join(BASE_OUTPUT_DIR, output_dir, output_name)
    snd_file = open(file_path, "w")
    snd_file.write(header_str)

    for idx in np.arange(prof["pres"].shape[0]):
        data_str = ""
        for col in ['pres', 'hght', 'tmpc', 'dwpc', 'wdir', 'wspd']:
            data_str += "\t%8.2f " % prof[col][idx]
        snd_file.write(data_str + "\n")

    snd_file.write("\n")

    snd_file.close()



def gen_sounding_txt(profiles, sites_df, output_date):
    lats = sites_df["lat"].to_numpy()
    ## we want positive values for longitude 
    lons = sites_df["lon"].to_numpy() + 360.0
    names = sites_df["icao"].to_numpy()
    wmo = sites_df["synop"].to_numpy()
    elev = sites_df["elev"].to_numpy()
    for stn_idx in np.arange(profiles.shape[0]):
        prof = profiles[stn_idx, stn_idx]
        prof["stid"] = names[stn_idx]
        prof["wmo"] = wmo[stn_idx]
        prof["elev"] = elev[stn_idx]

        prof["pres"] = prof["pres"] / 100.0
        prof["tmpc"] = prof["tmpk"] - 273.15
        prof["dwpc"] = prof["dwpk"] - 273.15
        prof["wspd"] = prof["wspd"]
        prof["wdir"] = prof["wdir"]

        save_txt(prof, output_date)
        print(f"Wrote: GFS, {prof['init']}, {prof['valid']}, {prof['fh']}, {wmo[stn_idx]}, {names[stn_idx]}")



def main():
    today = dt.datetime.utcnow()
    seven_days_ago = today - dt.timedelta(days=7)
    YYYYMMDD = seven_days_ago.strftime("%Y%m%d")

    gfs_files = sorted(glob.glob(f"/raid/spc/efp/gfs/{YYYYMMDD}/gfs.t00z.pgrb2.0p25.f*"))
    sites_df = pd.read_csv("/home/kelton.halbert/CODEBASE/aifs-snds/sites.csv")

    for fname in gfs_files:

        ds = read_grib2(fname)
        lats = sites_df["lat"].to_numpy()
        ## we want positive values for longitude 
        lons = sites_df["lon"].to_numpy() + 360.0

        ds = ds.sel(latitude=lats, longitude=lons, method="nearest").load()

        LOG.info(f"Preprocessing dataset for: {fname}")
        ds = preprocess_ds(ds)
        LOG.info(f"Finished preprocessing for: {fname}")

        ## load all data into memory as float32
        ## for usage with sounding library
        ds = ds.load().astype('float32')
            
        LOG.info(f"Constructing profiles for: {fname}")
        input_core_dims = [
            [], [], [], [],
            [], [], [], 
            [], [], [],
            ['isobaricInhPa',], ['isobaricInhPa',],
            ['isobaricInhPa',], ['isobaricInhPa',],
            ['isobaricInhPa',], ['isobaricInhPa',],
            ['isobaricInhPa',],
        ]
        output_core_dims = [()]

        ## We're going to loop over every grid point in parallel,
        ## merging the surface data with the pressure level data 
        ## and masking out below-surface levels. This will return 
        ## a 2D array (ny, nx) of dictionaries, in which the 
        ## dictionaries contain our grouped profile data.
        ds["profiles"] = xr.apply_ufunc(
            construct_profile,
            ds["longitude"], ds["latitude"], ds["time"].values, ds["valid_time"].values,
            ds["sp"],            ds["t2m"], ds["d2m"],  ds["mixr_2m"], ds["u10"], ds["v10"], 
            ds["pres"], ds["gh"], ds["t"],   ds["dwpk"], ds["mixr"],    ds["u"],   ds["v"], 
            input_core_dims=input_core_dims,
            output_core_dims=[[]],
            dask="parallelized",
            vectorize=True
        )

        gen_sounding_txt(ds["profiles"].values, sites_df, today.strftime("%Y%m%d")) 

        ds.close()
        ds = None



if __name__ == "__main__":
    main()
    

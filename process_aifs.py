import os
import glob
import logging
import numpy as np
import pandas as pd
import xarray as xr 
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

BASE_OUTPUT_DIR = "/raid/efp/se2024/ftp/spc/kthalbert/ai_soundings/"

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

    if model == "graphcast":
        output_dir = f"graphcast/{output_date}/"
        output_name = f"graphcast_{str(wmo).zfill(6)}.{init_time.strftime('%Y%m%d%H')}.f{str(int(fh)).zfill(3)}.txt"
    elif model == "pangu":
        output_dir = f"pangu/{output_date}/"
        output_name = f"pangu_{wmo}.{init_time.strftime('%Y%m%d%H')}.f{str(int(fh)).zfill(3)}.txt"
    elif model == "fcst_v2":
        output_dir = f"fcst_v2/{output_date}/"
        output_name = f"fcst_v2_{str(wmo).zfill(6)}.{init_time.strftime('%Y%m%d%H')}.f{str(int(fh)).zfill(3)}.txt"

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


def gen_sounding_txt(nc_file, sites_df, output_date, model=None):
    ds = xr.open_dataset(nc_file, engine="netcdf4")
    ds = ds.drop_vars(["u10", "v10", "t2", "msl", "w", "apcp"], errors="ignore")
    ds = ds.load()

    lats = sites_df["lat"].to_numpy()
    ## we want positive values for longitude 
    lons = sites_df["lon"].to_numpy() + 360.0
    names = sites_df["icao"].to_numpy()
    wmo = sites_df["synop"].to_numpy()
    elev = sites_df["elev"].to_numpy()

    profiles = ds.sel(latitude=lats, longitude=lons, method="nearest")

    ds.close()
    ds = None

    for time_idx in np.arange(profiles.time.shape[0]):
        for stn_idx in np.arange(profiles.longitude.shape[0]):
            init_time = dt.datetime.utcfromtimestamp(int(profiles["time"][0].values)/1e9)
            valid_time = dt.datetime.utcfromtimestamp(int(profiles["time"][time_idx].values)/1e9)
            forecast_hour = (valid_time - init_time).total_seconds()//3600.0

            ## convert geopotential to geopotential height
            hght = profiles["z"][time_idx, :, stn_idx, stn_idx].astype('float32') / 9.81 
            pres = hght.level.values.astype('float32') * constants.HPA_TO_PA
            tmpk = profiles["t"][time_idx, :, stn_idx, stn_idx].astype('float32')

            if model == "fcst_v2":
                relh = profiles["r"][time_idx, :, stn_idx, stn_idx].astype('float32') / 100.0
                rsat = thermo.mixratio(pres, tmpk)
                mixr = relh * rsat 
            else: 
                spfh = profiles["q"][time_idx, :, stn_idx, stn_idx].astype('float32')
                mixr = thermo.mixratio(spfh)
            dwpk = thermo.temperature_at_mixratio(mixr, pres)

            uwin = profiles["u"][time_idx, :, stn_idx, stn_idx].astype('float32') * 1.944
            vwin = profiles["v"][time_idx, :, stn_idx, stn_idx].astype('float32') * 1.944

            wspd = winds.vector_magnitude(uwin, vwin)
            wdir = winds.vector_angle(uwin, vwin)


            prof = {}
            prof["init"] = init_time
            prof["valid"] = valid_time
            prof["fh"] = forecast_hour
            prof["stid"] = names[stn_idx]
            prof["lat"] = lats[stn_idx]
            prof["lon"] = lons[stn_idx]
            prof["wmo"] = wmo[stn_idx]
            prof["elev"] = elev[stn_idx]

            prof["pres"] = pres / 100.0
            prof["hght"] = hght
            prof["tmpc"] = tmpk - 273.15
            prof["dwpc"] = dwpk - 273.15
            prof["wspd"] = wspd
            prof["wdir"] = wdir

            save_txt(prof, output_date, model=model)
            print(f"Wrote: {model}, {init_time}, {valid_time}, {forecast_hour}, {wmo[stn_idx]}, {names[stn_idx]}")

    profiles.close()
    profiles = None


def main():

    sites_df = pd.read_csv("/home/kelton.halbert/CODEBASE/aifs-snds/sites.csv")

    today = dt.datetime.utcnow() 

    seven_days_ago = today - dt.timedelta(days=7)
    YYYYMMDD = seven_days_ago.strftime("%Y%m%d")

    graphcast_files = sorted(glob.glob(f"/raid/efp/se2024/ftp/ai/gsl/graphcast/{YYYYMMDD}/00/GRAP_v100_GFS_*.nc"))
    pangu_files = sorted(glob.glob(f"/raid/efp/se2024/ftp/ai/gsl/pangu/{YYYYMMDD}/00/PANG_v100_GFS_*.nc"))
    fourcast_files = sorted(glob.glob(f"/raid/efp/se2024/ftp/ai/gsl/fcst_v2/{YYYYMMDD}/00/FOUR_v200_GFS_*.nc"))

    for fname in graphcast_files:
        gen_sounding_txt(fname, sites_df, today.strftime("%Y%m%d"), model="graphcast")

    for fname in pangu_files:
        gen_sounding_txt(fname, sites_df, today.strftime("%Y%m%d"), model="pangu")

    for fname in fourcast_files:
        gen_sounding_txt(fname, sites_df, today.strftime("%Y%m%d"), model="fcst_v2")


if __name__ == "__main__":
    main()



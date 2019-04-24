# coding: utf-8

from pathlib import Path
from tqdm import tqdm
import zipfile
import urllib.request
import datetime
import pandas as pd
import numpy as np

urls = {
    "2000": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/223",
    "2001": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/224",
    "2002": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/225",
    "2003": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/226",
    "2004": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/202",
    "2005": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/203",
    "2006": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/227",
    "2007": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/228",
    "2008": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/229",
    "2009": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/230",
    "2010": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/231",
    "2011": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/232",
    "2012": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/233",
    "2013": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/234",
    "2014": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/235",
    "2015": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/236",
    "2016": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/242",
    "2017": "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/262",
}

metadata_url = "http://powietrze.gios.gov.pl/pjp/archives/downloadFile/265"

data_dir = Path("./data")
zip_dir = data_dir / "zip"
extracted_dir = data_dir / "raw"
hdf_file = data_dir / "data.h5"


def download_raw_data(
    urls=urls,
    zip_dir=zip_dir,
    extracted_dir=extracted_dir,
    cleanup=True,
    download=False,
):
    "Downloads the zipped Excel files and unzips them into `extracted_dir`"
    for url in tqdm(urls.values(), "Downloading and unzipping"):
        name = zip_dir / url.split("/")[-1]
        if not name.exists() or download:
            urllib.request.urlretrieve(url, name)
        zipfile.ZipFile(name, "r").extractall(extracted_dir)
        if cleanup:
            name.unlink()


def merc_from_arrays(phi, lam):
    "Converts latitude and longitude to a mercator projection"
    phi = phi / (180 / np.pi)
    lam = lam / (180 / np.pi)
    r_major = 6378137.000
    x = r_major * lam
    y = r_major * np.log(np.tan(np.pi / 4 + phi / 2))
    return (x, y)


def add_web_mercator(metadata):
    "Adds two columns with web mercator coordinates to the metadata"
    E, N = merc_from_arrays(metadata["WGS84 φ N"], metadata["WGS84 λ E"])
    metadata["web_mercator_E"] = E
    metadata["web_mercator_N"] = N


def get_metadata(
    metadata_url=metadata_url,
    cleanup=True,
    download=False,
    path=extracted_dir / "metadata.xlsl",
):
    """Downloads the metadata file as xlsx and """

    if not path.exists() or download:
        print(f"Downloading the metadata.xml from {metadata_url}")
        urllib.request.urlretrieve(metadata_url, path)

    metadata = pd.read_excel(path).drop(["Nr"], axis=1)

    if cleanup:
        metadata_file.unlink()

    # clean up the -999.0 values for latitude and longitude
    for col in ["WGS84 φ N", "WGS84 λ E"]:
        metadata.loc[metadata[col] < -180, col] = np.nan

    add_web_mercator(metadata)

    return metadata


def detect_kod_stacji(df):
    """Finds the row with the station code among the top 10 rows"""
    for idx, row in df.reset_index(drop=True).head(10).iterrows():
        if row.str.contains("^[A-Z][a-z]").mean() > 0.9:
            return idx
    raise Exception(f"Could not find Kod stacji, {df.head(10)}")


def detect_values(df):
    """Returns the indices of the rows indexed by date.  These rows contain
the values from the sensors."""
    return df.index.map(lambda x: isinstance(x, datetime.datetime))


def format_table(df):
    """Cuts out the pieces of the data frame corresponding to the station
names and sensor readings.  This is necessary because Excel files from
different years have slightly different formats (more/fewer columns/rows).

    """
    kod_stacji_idx = detect_kod_stacji(df)
    values_idx = detect_values(df)
    df_new = df[values_idx].copy()
    df_new.index = pd.to_datetime(df_new.index)
    df_new.index.names = ["Date"]
    df_new.columns = list(df.iloc[kod_stacji_idx])
    return df_new


def convert_to_kod_stacji(df, metadata):
    """Determines if the data frame `df` uses old or new station names and
converts the names to the new ones."""
    if not set(df.columns).difference(metadata["Kod stacji"]):
        return

    df.columns = list(
        pd.DataFrame(index=df.columns)
        .merge(
            metadata[["Kod stacji", "Stary Kod stacji"]],
            how="left",
            left_index=True,
            right_on="Stary Kod stacji",
        )
        .loc[:, "Kod stacji"]
    )


def get_file_name(path, year, measure, interval=24):
    """Generates the Excel file name based on the year and measured value
(e.g. SO2 or PM2.5).  Moreover, some files have PM2.5 in their names
and some have PM25, so we this function also checks for the
alternative names in this special case.

    """
    xls_file = Path(path) / f"{year}_{measure}_{interval}g.xlsx"
    if not xls_file.exists():
        xls_file = Path(str(xls_file).replace("PM2.5", "PM25"))

    if not xls_file.exists():
        print(f"Warning: file {xls_file} not found.")
        return None

    return xls_file


def convert_to_float(df):
    """Cleans up the values by converting the commas to dots and casting the
results to the float32 type"""
    return df.apply(lambda x: x.str.replace(",", ".").astype("float32"))


def to_dataframe(xls_file, metadata):

    # short circuit to an empty data frame in case the file does not
    # exist
    if not xls_file:
        return pd.DataFrame()

    df = pd.read_excel(xls_file, header=None, index_col=0)
    df = format_table(df)
    convert_to_kod_stacji(df, metadata)
    df = df.loc[:, ~df.columns.duplicated()]
    df = convert_to_float(df)
    df.drop(df.columns[df.columns.isnull()], axis=1, inplace=True)
    return df


def get_indicators(extracted_dir=extracted_dir):
    return set(
        path.name.split("_")[1].replace("PM25", "PM2.5")
        for path in extracted_dir.glob("*.xlsx")
        if "depozycja" not in path.name.lower() and "jony" not in path.name.lower()
    )


def xlsx_to_hdf(
    metadata,
    extracted_dir=extracted_dir,
    hdf_file=hdf_file,
    indicators=["PM2.5", "PM10", "SO2", "NO2"],
    years=urls.keys(),
    interval=24,
):

    for indicator in tqdm(indicators, "Saving to a hdf5 format"):
        lst = [
            to_dataframe(
                get_file_name(extracted_dir, year, indicator, interval=interval),
                metadata,
            )
            for year in years
        ]
        all_data = pd.concat(lst, axis=0, sort=False)
        all_data.to_hdf(hdf_file, f"{interval}g/{indicator}", complevel=5)
    metadata.to_hdf(hdf_file, "metadata", complevel=5)


# download_raw_data(cleanup=False)

metadata = get_metadata(cleanup=False)
if hdf_file.exists():
    hdf_file.unlink()
xlsx_to_hdf(metadata)
# xlsx_to_hdf(metadata, interval=1)
# pd.read_hdf(hdf_file, "24g/PM2.5")

"""
2022-11-02 edits by thomas mrked as #t_edit_11_2022
    fixed an issue where the EIA923 started using 'respondent frequency' instead of 'reporting frequency'
"""




import os
from os.path import join, abspath, dirname, split
import pandas as pd
# from util.utils import getParentDir
import sys
from src.params import DATA_PATHS
from src.util import download_unzip
PY3 = sys.version_info.major == 3


def get_annual_plants(year,
                      website='https://www.eia.gov/electricity/data/eia923/',
                      plant_frame_kws={'header': 4}):
    """
    Download the EIA-923 file for a given year if it doesn't already exist,
    and extract a list of plant ids for facilities that report annually. This
    function assumes that EIA is continuing to use the same structure for
    links.

    inputs:
        year: (int) year of data to download
        website: (str) the website for EIA-923 data, which is used when
            downloading zip files

    ouputs:
        annual_id: a Pandas Series with plant ids
    """

    import zipfile
    if PY3:
        from urllib.request import urlretrieve
    else:
        from urllib import urlretrieve

    # unzip_path = None

    # Get the project top-level path
    # ap = abspath(__file__)
    # top_path = getParentDir(dirname(ap), level=2)

    # Make folder if it doesn't exist
    # path = join(top_path, 'Data storage', 'EIA downloads')
    # path = DATA_PATHS['eia923']
    # if not path.exists():
    #     os.mkdir(path)

    # Download the 923 zip file if it doesn't exist
    folder = f'f923_{year}'
    fname = folder + '.zip'
    # existing_zips = [split(x)[-1] for x in DATA_PATHS['eia923'].glob('*.zip')]
    unzip_path = DATA_PATHS['eia923'] / folder
    if not unzip_path.exists():
        try:
            url = website + f'xls/{fname}'
            download_unzip(url, unzip_path)
        except ValueError:
            url = website + f'archive/xls/{fname}'
            download_unzip(url, unzip_path)
    # if fname not in existing_zips:
    #     url = website + f'xls/{fname}'
    #     urlretrieve(url, filename=save_path)

    # See if the unzipped folder exists. If not, unzip the file
    # unzip_path = join(path, fname.split('.')[0])
    # unzip_path = DATA_PATHS['eia923'] / fname.split('.')[0]
    # print(save_path)
    # z_file = zipfile.ZipFile(save_path)
    # z_names = z_file.namelist()
    # if not unzip_path.exists():
    #     # Extract the zipfile into new folder
    #     z_file.extractall(unzip_path)

    # Read the sheet "Page 6 Plant Frame" from the appropriate file.
    # year_fn = glob.glob(join(unzip_path, '*Schedules_2*'))[0]
    # Use the z_names variable, which is a list of file names in the zipfile
    files = unzip_path.glob('*.xlsx')
    read_path = [x for x in files if '_Schedules_2' in x.name][0]
    # read_path = join(unzip_path, year_fn)
    # read_path = unzip_path / year_fn
    df = pd.read_excel(read_path, sheet_name='Page 6 Plant Frame',
                       **plant_frame_kws)

    # Column names in EIA documents can have line breaks and extra spaces
    df.columns = [col.lower().replace('\n', ' ').strip() for col in df.columns]

    # Get the plant ids for just plants that report annually
    #t_edit_11_2022 updated 'reporting frequency' to 'respondent frequency'
    annual_plants = df.loc[df['respondent frequency'] == 'A', 'plant id']
    annual_plants = (annual_plants.drop_duplicates()
                                  .reset_index(drop=True))

    return annual_plants


def states_in_nerc():
    """
    Function to create a file that will list all of the states in each of the
    NERC regions.

    output:
        JSON file that lists all states in each NERC region
    """
    import json
    import geopandas as gpd
    # from shapely.geometry import Point
    from geopandas import GeoDataFrame

    # Get the project top-level path
    ap = abspath(__file__)
    top_path = getParentDir(dirname(ap), level=2)

    # Read NERC shapefile
    nerc_path = join(top_path, 'Data storage', 'NERC_Regions_EIA',
                'NercRegions_201610.shp')
    nerc = gpd.read_file(nerc_path)

    state_path = join(top_path, 'Data storage', 'cb_2016_us_state_500k',
                    'cb_2016_us_state_500k.shp')
    states = gpd.read_file(state_path)
    state_cols = ['STUSPS', 'geometry']
    nerc_cols = ['NERC', 'geometry']

    df = gpd.sjoin(states[state_cols], nerc.loc[nerc['NERC'] != '-', nerc_cols])

    s = pd.DataFrame(columns=['NERC', 'state'])
    s['NERC'] = df.loc[:, 'NERC'].values
    s['state'] = df.loc[:, 'STUSPS'].values

    # Still need to output s to a file.
    state_dict = {}

    for NERC in s['NERC'].unique():
        state_dict[NERC] = s.loc[s['NERC'] == NERC, 'state'].tolist()

    json_path = join(top_path, 'Data storage', 'Derived data',
                    'NERC_states.json')
    with open(json_path, 'w') as f:
        json.dump(state_dict, f, indent=4)


def facility_location_data(eia_facility):
    """
    Output a csv with the state and lat/lon for each facility in the eia data
    """
    # Get the project top-level path
    ap = abspath(__file__)
    top_path = getParentDir(dirname(ap), level=2)
    out_path = join(top_path, 'Data storage', 'Facility labels')

    facility_location = pd.DataFrame(columns=['plant id',
                                              'state',
                                              'lat',
                                              'lon'])

    eia_facility['state'] = eia_facility.geography.str[-2:]
    cols = ['plant id', 'state', 'lat', 'lon']
    unique_df = eia_facility.loc[:, cols].drop_duplicates()

    assert len(unique_df) == len(eia_facility['plant id'].unique())

    unique_df.to_csv(join(out_path, 'Facility locations.csv'), index=False)

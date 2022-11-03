# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 12:30:00 2022

@author: tdeet
"""

import pandas
import zipfile
import requests
import io
from lxml import etree
import warnings


DATA_DATE = '2022-11-01' #hard copied from params.py



zipurl = 'https://gaftp.epa.gov/DMDnload/emissions/daily/quarterly/2022/'

def getZipLinks(url):
    print("Getting links from: " + url)
    page = requests.get(url, verify=False)
    html = page.content.decode("utf-8")
    tree = etree.parse(io.StringIO(html), parser=etree.HTMLParser())
    refs = tree.xpath("//a")    
    refs_list = list(set([link.get('href', '') for link in refs]))
    zips_list = [z for z in refs_list if '.zip' in z]
    return zips_list


zlink_list = getZipLinks(zipurl)

convert_dict = {'kg' : 1., 'tons' : 907.1847, 'lbs' : 0.453592}

def agg_df_temp(df_temp_input):
    df_temp_input['month'] = df_temp_input['OP_DATE'].str[0:2].astype(int)
    df_temp_input['year'] = df_temp_input['OP_DATE'].str[-4:].astype(int)
    df_temp_input['co2mass_kg'] = df_temp_input['CO2_MASS (tons)'] * convert_dict['tons']
    df_temp_input['noxmass_kg'] = df_temp_input['NOX_MASS (tons)'] * convert_dict['tons']
    df_temp_input['so2mass_kg'] = df_temp_input['SO2_MASS (tons)'] * convert_dict['tons']
    df_temp_input = df_temp_input[['year', 'month', 'ORISPL_CODE', 'GLOAD (MWh)', 'HEAT_INPUT (mmBtu)', 'co2mass_kg', 'noxmass_kg', 'so2mass_kg']]
    df_temp_input.columns = ['year', 'month', 'plant id', 'gload_mwh', 'heatinput_mmbtu', 'co2mass_kg', 'noxmass_kg', 'so2mass_kg']
    return df_temp_input.groupby(['year', 'month', 'plant id'], as_index=False).agg('sum')


warnings.filterwarnings('ignore')
for zlink in zlink_list:
    zipfileurl = zipurl + zlink
    r = requests.get(zipfileurl, verify=False)
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open(zlink.replace('.zip', '.csv')) as f:
            df_temp = pandas.read_csv(f)
            df_temp_agg = agg_df_temp(df_temp)
    if zlink == zlink_list[0]:
        df_cems = df_temp_agg.copy()
    else:
        df_cems = pandas.concat([df_cems, df_temp_agg])
warnings.resetwarnings()


#%% we need to incorporate the above code at some point into the cems.py file, but for now, let's just update the most recent parquet file manually here

cems_parquet_dir = 'C:/Users/tdeet/Documents/analysis/carbon_index_project/carbon-index/data/transformed_data/epa_emissions/'
last_cems_parquet = cems_parquet_dir + 'epa_emissions_2022-07-15.parquet'
last_cems_df = pandas.read_parquet(last_cems_parquet)

#remove the previous file's 2022 data in case it's been recently updated in CEMS
last_cems_df = last_cems_df[last_cems_df['year']<2022]

#concat in the newly downloaded and aggregated data
new_cems_df = pandas.concat([last_cems_df, df_cems])
new_cems_df.reset_index(drop=True, inplace=True)

#save as a new parquet file using the params data_date string
path = cems_parquet_dir + 'epa_emissions_%s.parquet'%DATA_DATE
new_cems_df.to_parquet(path, index=False)
 


# This .py file generates the deliverables that answer questions 1 through 3 of 
# Voltus' Programming Assignment: Load Analysis

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
import Voltus_Project_Nathan_Functions_Q3 as functions_q3
import Voltus_Project_Nathan_Functions_Q1_2_Viz as functions_q12v

# connecting to the server
conn = psycopg2.connect(
    database="interval_load_data",
    user='postgres',
    password='Onei9yepahShac0renga',
    host="test-interval-load-data.cwr8xr5dhgm1.us-west-2.rds.amazonaws.com",
    port="5432"
)

# querying the sites and their time period into a pandas DataFrame
site_df = pd.read_sql_query("SELECT site_id, \
                             MIN(interval_end) as time_start, \
                             MAX(interval_end) as time_end, \
                             (MAX(interval_end) - MIN(interval_end)) as series_length, \
                             COUNT(*) as interval_counts, \
                             AVG(interval_width) as mean_interval_width \
                             FROM intervals \
                             GROUP BY site_id;", conn)

# this site had only a single interval of data and will be excluded from further analysis
insufficient_data_sites = pd.read_sql_query("SELECT site_id, COUNT(*) FROM intervals \
                                             GROUP BY site_id \
                                             HAVING COUNT(*) < 24;", conn)
# These sites had either all 0's
# or if some non-0 values existed then they were very small
# and only had a few unique values that didn't look like a load profile.
# These sites will be excluded from further analysis as they appear to have corrupted/unusable data
corrupt_sites = pd.read_sql_query("""SELECT site_id, PERCENTILE_CONT(0.5) WITHIN GROUP(ORDER BY "kWh") \
                                     FROM intervals \
                                     GROUP BY site_id \
                                     HAVING PERCENTILE_CONT(0.5) WITHIN GROUP(ORDER BY "kWh") < 0.05;""", conn)
# excluding the sites mentioned above
excluded_sites = pd.concat([insufficient_data_sites['site_id'], corrupt_sites['site_id']])
excluded_sites = list(excluded_sites)
site_df = site_df.loc[site_df['site_id'].isin(list(excluded_sites)) == False]
site_df.reset_index(drop=True, inplace=True)
site_list = list(site_df['site_id'])

# querying all the load data into a pandas DataFrame
all_data_df = pd.read_sql_query("""SELECT "kWh"*(3600./interval_width) as "avg_kW",
                                           interval_end, interval_width, site_id \
                                    FROM intervals \
                                    ORDER BY site_id, interval_end asc;"""
                                , conn)

# answering questions 1 and 2
start_stop_data, dr_event_data = functions_q12v.site_operating_and_dr_status(all_data_df, site_list)
# saving the results as .csv
# ensure deliverables folder exists
start_stop_data.to_csv('deliverables\continuous_flag_and_start_stop_times.csv', index=True)
dr_event_data.to_csv('deliverables\potential_dr_event_days.csv', index=True)

# answering question 3
all_gap_dates = functions_q3.multi_site_gap_date_df(site_list)
# saving the results as .csv 
# ensure deliverables folder exists
all_gap_dates.to_csv('deliverables\gap_dates.csv', index=False)


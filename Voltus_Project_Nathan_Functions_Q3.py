import psycopg2
import pandas as pd

# FUNCTIONS FOR QUESTION 3 - Identifying Data Gaps
def interval_width_transitions(interval_df):
    """
    Given a DataFrame of interval load data, return a zip object
    identifying when there is a transition from one interval_width to another with tuples
    containing the timestamps of the transition start (i.e. last value with the previous width)
    and transition end (i.e. first value with the next width)
    """
    filler_start = interval_df['interval_width'].iloc[-1]
    filler_end = interval_df['interval_width'].iloc[0]
    # compares shifted interval_widths to identify a transition between widths
    transitions_start = (interval_df['interval_width'] 
                         != interval_df['interval_width'].shift(-1,
                                                                axis=0,
                                                                fill_value=filler_start))
    transitions_end = (interval_df['interval_width'] 
                       != interval_df['interval_width'].shift(1,
                                                              axis=0,
                                                              fill_value=filler_end))
    transitions = zip(interval_df.loc[transitions_start, 'interval_end'],
                      interval_df.loc[transitions_end, 'interval_end'])
    return list(transitions)

def resample_intervals(interval_df):
    """
    Given a DataFrame of interval load data, return a DataFrame with 
    resampled data at its provided frequency such that gaps get filled with nulls.
    """ 
    time_series = interval_df.set_index('interval_end', drop=False)
    time_series.sort_index(inplace=True)
    interval_widths = time_series['interval_width'].unique()
    resampled_series = pd.DataFrame()
    # separately resampling segments with different interval_widths
    for interval in interval_widths:
        interval_time_series = time_series.loc[time_series['interval_width'] == interval].asfreq(f'{interval}S')
        resampled_series = pd.concat([resampled_series, 
                                        interval_time_series])
    # resampling the time between a transition from different interval_widths
    if len(interval_widths) > 1:
      transitions = interval_width_transitions(interval_df)
      for transition in transitions:
          min_interval = min(time_series.loc[transition[0], 'interval_width'],
                             time_series.loc[transition[1], 'interval_width'])
          interval_time_series = time_series.loc[((time_series['interval_end'] == transition[0])
                                                  | (time_series['interval_end'] == transition[1]))
                                                 ].asfreq(f'{min_interval}S')
          # the transition start/stop values are already in the data,
          # so they're dropped to prevent duplicates
          interval_time_series.drop([transition[0], transition[1]], inplace=True)
          resampled_series = pd.concat([resampled_series, 
                                        interval_time_series])
    return resampled_series.sort_index()

def data_gap_start_stop(interval_df):
    """
    Given a DataFrame of interval load data, 
    returns the timestamps at the beginning and end of data gaps.
    """
    null_ixs = interval_df.loc[interval_df['interval_end'].isnull()].index
    # identifying all possible interval_ends before and after a null
    pre_nulls = interval_df.shift(-1, axis=0, fill_value=0)
    post_nulls = interval_df.shift(1, axis=0, fill_value=0)
    # removing times that are known nulls and therefore not the beginning or
    # end of a gap, but in fact part of the gap
    pre_null_ixs = pre_nulls.loc[(pre_nulls['interval_end'].isnull()
                                  & (pre_nulls.index.isin(null_ixs) == False))]
    post_null_ixs = post_nulls.loc[(post_nulls['interval_end'].isnull()
                                    & (post_nulls.index.isin(null_ixs) == False))]
    # had to concatenate to ensure times at the beginning and end of 2
    # different gaps (i.e. gap then 1-interval then another gap)
    # were kept twice to create the two gap ranges and sorted to ensure order
    gap_ixs = pd.concat([pre_null_ixs, post_null_ixs]).sort_index()
    return gap_ixs.index

def gap_date_df(interval_df):
    """
    Given a DataFrame of interval load data, 
    returns a DataFrame with all days that include a data gap greater than 1 hour.
    If the gap is greater than 1 hour, but the amount of that gap occurring on a given date
    is less than 1 hour, then that date still is included because of the overall gap length.
    """
    working_df = resample_intervals(interval_df)
    gap_starts_stops = data_gap_start_stop(working_df)
    gap_dates = pd.DataFrame()
    if len(gap_starts_stops) == 0:
        pass
    else:
        range_len = len(gap_starts_stops)
        # checking whether the gap is over an hour in length
        # considers the gap between and interval_end and the start of the next interval
        # not just the gap between interval_ends, which start an interval width before
        # their interval_end time
        for n in range(0, range_len, 2):
            gap_length = gap_starts_stops[n+1] - gap_starts_stops[n]
            gap_end_interval = working_df.loc[gap_starts_stops[n+1],
                                              'interval_width']
            gap_length -= pd.Timedelta(gap_end_interval, unit='s')
            if gap_length > pd.Timedelta(3600, unit='s'):
                # identifying all dates in the gap
                temp_gap_dates = pd.date_range(gap_starts_stops[n],
                                               gap_starts_stops[n+1],
                                               normalize=True)
                temp_gap_dates = pd.DataFrame(temp_gap_dates,
                                              columns=['gap_dates'])
                gap_dates = pd.concat([gap_dates,
                                       temp_gap_dates])
            gap_dates['site_id'] = working_df['site_id'].iloc[0]
    return gap_dates

def multi_site_gap_date_df(list_of_sites):
    """
    Given a list of site_ids, connect to the provided database and loop through 
    querying of each site's data, creating and then returning a DataFrame
    with all days that include a data gap greater than 1 hour.
    """
    connection = psycopg2.connect(database="interval_load_data",
                                  user='postgres',
                                  password='Onei9yepahShac0renga',
                                  host="test-interval-load-data.cwr8xr5dhgm1.us-west-2.rds.amazonaws.com",
                                  port="5432")    
    all_interval_df = pd.read_sql_query("""SELECT * \
                                           FROM intervals \
                                           WHERE "kWh" IS NOT NULL \
                                           ORDER BY site_id, interval_end asc;""",
                                        connection)
    gap_dates_df = pd.DataFrame()
    for site in list_of_sites:
        site_interval_df = all_interval_df.loc[all_interval_df['site_id'] == site]
        temp_gap_dates_df = gap_date_df(site_interval_df)
        gap_dates_df = pd.concat([gap_dates_df,
                                  temp_gap_dates_df])
    return gap_dates_df.reset_index(drop=True)
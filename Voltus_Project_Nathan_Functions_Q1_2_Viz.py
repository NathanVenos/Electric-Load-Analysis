import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

# FUNCTIONS FOR QUESTION 1 & 2 - Classifying Continuously Operating Sites,
# Daily Operating Start and Stop times if not Conitunous,
# And flagging dates that potentially had DR events.
def aggregate_week_ts(working_ts, ops_params, dr=False):
    """
    Given a time series DataFrame, return an aggregate
    time series spanning one week with the median value
    populated for each interval
    """
    # Shifting all data such that the first interval_end is the 0-hour of its day.
    # This allows grouping by day of week in a way that retains the ability to
    # include the most full days of data in the analysis. Sites with only a few
    # days of data whether continuous or not were seen to generally have the same
    # start time for each segment of data, so this method is helpful for preventing,
    # or at least limiting cases where an aggregate day has less than 24-hours of data.
    width = ops_params['interval_width']
    if dr == False:
        init_time = working_ts['interval_end'][0]       
        shifted_time = init_time.replace(hour=0, minute=0, second=0)
        time_shift = shifted_time - init_time
        working_ts['shifted_interval'] = working_ts['interval_end'] + time_shift
        # getting datetime parameters of interest for grouping
        working_ts['shifted_date'] = working_ts['shifted_interval'].dt.date
        working_ts['shifted_day_of_week'] = working_ts['shifted_interval'].dt.dayofweek
        working_ts['shifted_time_of_day'] = working_ts['shifted_interval'].dt.strftime('%H:%M:%S')
        # filtering out days without 24 hours of data out of caution
        daily_intervals = 24 / (width / 3600)
        date_counts = working_ts['shifted_date'].value_counts()
        full_dates = date_counts.loc[date_counts == daily_intervals].index
        working_ts = working_ts.loc[working_ts['shifted_date'].isin(full_dates)].copy()
        ops_params.update({'initial_time_shift': time_shift})
    else:
        # preparing for another time shift that occurs when identifying dr events
        if ops_params['continuous'] == 0:
            time_shift = ops_params['dr_time_shift']
        else:
            time_shift = ops_params['initial_time_shift']
    # performing the day of week and hour of day groupby
    week_df = working_ts[['avg_kW',
                          'shifted_day_of_week',
                          'shifted_time_of_day']].groupby(by=['shifted_day_of_week',
                                                              'shifted_time_of_day']).median()
    # generating metadata and other potentially useful information
    num_daysofweek = len(list(week_df.index.get_level_values(0).unique()))
    ops_params.update({'agg_week_num_days': num_daysofweek})
    week_df['time_shifted'] = time_shift
    week_df['imputed_interval_width'] = width
    return week_df, ops_params

def high_load_start_stop(week_ts):
    """
    Given a DataFrame with known high-load and non-high-load operating times,
    return the dataframe with columns representing any time high-load operation
    starts or stops time for each day.
    """
    # generating shifted time series to identify when the transition between
    # low-load and high-load periods occurs
    working_df = week_ts
    shift_forward_fill = working_df['high_load_period'].iloc[-1]
    shift_backward_fill = working_df['high_load_period'].iloc[0]
    shifted_back_df = working_df['high_load_period'].shift(-1, fill_value=shift_backward_fill)
    shifted_forward_df = working_df['high_load_period'].shift(1, fill_value=shift_forward_fill)
    # comparing the shifted and original time series 
    # to determine transitions as potential start and stop times
    working_df['high_start'] = (working_df['high_load_period'] > shifted_forward_df)
    working_df['high_stop'] = (working_df['high_load_period'] > shifted_back_df)
    return working_df

def week_load_profile_params(week_ts, ops_params, dr=False, base_load_pctl=15, near_base_pctl=5, near_peak_pctl=95):
    """
    Given a time series aggregated by day of week with sub-daily intervals,
    return columns and metdata defining various load profile parameters for each day.
    """
    working_df = week_ts
    working_df['base_load'] = 0
    working_df['near_base_load'] = 0
    working_df['near_peak_load'] = 0
    working_df['near_base_delta'] = 0
    working_df['near_peak_delta'] = 0
    working_df['high_load_period'] = 0
    days = week_ts.index.get_level_values(0).unique()
    for day in days:
        # generating and storing various base, peak and high load parameters
        base_load = np.percentile(working_df.loc[(day, ), 'avg_kW'],
                                  base_load_pctl)
        near_base_load = np.percentile(working_df.loc[(day, ), 'avg_kW'],
                                       near_base_pctl)
        near_peak_load = np.percentile(working_df.loc[(day, ), 'avg_kW'],
                                       near_peak_pctl)
        working_df.loc[(day, ), 'base_load'] = base_load
        working_df.loc[(day, ), 'near_base_load'] = near_base_load
        working_df.loc[(day, ), 'near_peak_load'] = near_peak_load
        # comparing hourly load to determine if a high load hour
        working_df.loc[(day, ), 'near_base_delta'] = (working_df.loc[:, 'avg_kW']
                                                      - working_df.loc[:, 'near_base_load'])
        working_df.loc[(day, ), 'near_peak_delta'] = (working_df.loc[:, 'near_peak_load']
                                                      - working_df.loc[:, 'avg_kW'])
        working_df.loc[(day, ), 'high_load_period'] = (working_df.loc[:, 'near_base_delta']
                                                       > working_df.loc[:, 'near_peak_delta'])
        # generating metadata for use in other functions
        ref_peak = working_df['near_peak_load'].max()
        ref_base = working_df['near_base_load'].min()
        ops_params.update({'base_load_'+str(day): base_load,
                           'near_base_load_'+str(day): near_base_load,
                           'near_peak_load_'+str(day): near_peak_load,
                           'ref_peak': ref_peak,
                           'ref_base': ref_base, })
    if dr == True:
        # calculating additional parameters for the time shifted data being used
        # to identify DR events where the code only focuses on the high load period
        working_df = high_load_start_stop(working_df)
        # taking the earliest potential start time and latest stop time in case the data is noisy
        starts = []
        stops = []
        for day in days:
            day_slice = working_df.loc[(day, ), ['high_start', 'high_stop']]
            # handling exceptions when a load doesn't start i.e. continuous or
            # when start is around midnight, but can vary a little day to day,
            # crossing into a different day and not occuring on the current day
            try:
                start_time = day_slice.loc[day_slice['high_start']].index[0]
            except:
                start_time = '00:00:00'
            start_datetime = datetime.strptime(start_time, '%H:%M:%S')
            start_time = start_datetime.time()
            starts.append(start_time)
            # handling exceptions similar to above
            try:
                stop_time = day_slice.loc[day_slice['high_stop']].index[-1]
            except:
                if ops_params['interval_width'] == 900:
                    stop_time = '23:45:00'
                else:
                    stop_time = '23:00:00'
            stop_datetime = datetime.strptime(stop_time, '%H:%M:%S')
            stop_time = stop_datetime.time()
            stops.append(stop_time)
            high_length = stop_datetime - start_datetime
            high_length_intervals = 1 + (high_length.total_seconds() 
                                         / ops_params['interval_width'])
            # adding metadata to dictionary
            high_start_stop_threshold = (ref_peak + ref_base)/2
            ops_params.update({'high_start_'+str(day): start_time,
                               'high_stop_'+str(day): stop_time,
                               'high_length_intervals_'+str(day): high_length_intervals,
                               'min_high_start': min(starts),
                               'max_high_stop': max(stops),
                               'hi_start_stop_threshold': high_start_stop_threshold})
    return working_df, ops_params

def agg_week_continuous_operation(weekly_df, ops_params, days_iter_index, ref_peak, ref_base, p_threshold):
    """
    Given an aggregated time series with a mult-index where the first index is the day of week 
    and the second is the time of day, this function uses a Welch's t-test to compare 
    the daily base and peak loads to determine whether they exhibit a difference 
    that is statistically significant based on the p_threshold parameter. 
    If no significant difference between the two then the site is considered to be operating continuously.
    """
    # generating the base and peak arrays
    near_peaks = np.array([weekly_df.loc[(day,), 'near_peak_load'][0] for day in days_iter_index])
    base_loads = np.array([weekly_df.loc[(day,), 'base_load'][0] for day in days_iter_index])
    # by default setting sites with one day of data as daily since not enough to compare.
    # Some seemed like they may have been continuous but often also had down peaks.
    # Potentially these were continuous with DR or something, but insufficient
    # data is available to make a definitive assessment
    if len(near_peaks) <= 1:
        ops_params.update({'p_value': np.nan})
        return ops_params
    # handling some edge cases, which may be irrelevant now, but were needed
    # before I removed some sites I deemed to have corrupted data.
    elif ((np.sum(near_peaks) == np.sum(base_loads)) 
          or (list(near_peaks) == list(base_loads))):
        ops_params.update({'continuous': 1, 'p_value': np.nan})
        return ops_params
    # now testing for continuous by performing a 2-sample t-test to
    # determine if the peaks and bases appear to be from the same population.
    # In retrospect, I think a more robust process is needed since some
    # potential continuous sites appeared to have variation over the week,
    # or inconsistent variation within the day that was preventing their detection,
    # or simply the daily variation seems either small enough or variable enough
    # that the classification becomes somewhat ambiguous.
    else:
        near_peak_mean = np.mean(near_peaks)
        near_peak_std = np.std(near_peaks)
        t_stat, p_val = ttest_ind(near_peaks, base_loads, equal_var=False, nan_policy='omit')
        operating_days = []
        if p_val > p_threshold:
            ops_params.update({'continuous': 1})
            # checking for whether each day appears to be an operating day based on
            # whether its daily peak is closer to the reference peak or reference base,
            # which were calculated as the max and min near peak and near base respectively,
            # but could be changed if desired in the future
            for day in days_iter_index:
                daily_peak = weekly_df.loc[(day, ), 'near_peak_load'][0]
                delta_daily_peak_ref_peak = ref_peak - daily_peak
                delta_daily_peak_ref_base = daily_peak - ref_base
                if (((near_peak_mean - near_peak_std) < daily_peak)
                    and (delta_daily_peak_ref_base > delta_daily_peak_ref_peak)):
                    ops_params.update({'dayofweek_'+str(day): 1})
                    operating_days.append(day)
                else:
                    ops_params.update({'dayofweek_'+str(day): 0})
        ops_params.update({'p_value': p_val})
        # repeating the t-test only on operating days because
        # it was observed that sites with base load not far below peak 
        # during operating days and low peak and base during non-operating
        # days we're getting incorrectly flagged as continuous
        near_peaks_op = np.array([weekly_df.loc[(day,), 'near_peak_load'][0] for day in operating_days])
        base_loads_op = np.array([weekly_df.loc[(day,), 'base_load'][0] for day in operating_days])
        if ((len(near_peaks_op) == 1)
            or (list(near_peaks_op) == list(base_loads_op))
            or (np.sum(near_peaks_op) == np.sum(base_loads_op))):
            return ops_params
        else:
            t_stat_op, p_val_op = ttest_ind(near_peaks_op, base_loads_op, equal_var=False, nan_policy='omit')
            if p_val_op > p_threshold:
                ops_params.update({'continuous': 1})
            else:
                ops_params.update({'continuous': 0})
            ops_params.update({'p_value': p_val_op})
        return ops_params

def agg_week_operating_day(weekly_df, days_iter_index, ops_params, p_threshold):
    """
    Given an aggregated time series with a mult-index where the first index is the day of week
    and the second is the time of day, this function compares each day's near_peak_load 
    against a reference near_base_load and near_peak_load to determine operating status 
    based on whether the day's near_peak is closer to a reference near_base (i.e. not operating)
    or is closer to a reference near_peak (i.e. operating).
    It may mis-identify days in which operation is significantly different than usual,
    or if only non-operating dates are present in the time series, or if the time series
    is relatively short but exhibits high variance within that time because 
    these scenarios make it difficult to determine appropriate reference points.
    """
    ops_params.update({'continuous': 0})
    ref_peak = ops_params['ref_peak']
    ref_base = ops_params['ref_base']
    # create a dict of standard size to hold results with nan's as the default to
    # account for time series where less than a week of data is present
    for i in range(7):
        ops_params.update({'dayofweek_'+str(i): np.nan})
    # checking first for continuous processes and their operating days
    # using the same approach for determing operating status, 
    # but only after classifying as continuously operating or not
    ops_params = agg_week_continuous_operation(weekly_df,
                                               ops_params,
                                               days_iter_index,
                                               ref_peak,
                                               ref_base,
                                               p_threshold=p_threshold)
    # checking for daily-varying sites' operating days
    if ops_params['continuous'] == 0:
        for day in days_iter_index:
            daily_peak = weekly_df.loc[(day, ), 'near_peak_load'][0]
            delta_daily_peak_ref_peak = ref_peak - daily_peak
            delta_daily_peak_ref_base = daily_peak - ref_base
            if delta_daily_peak_ref_base > delta_daily_peak_ref_peak:
                ops_params.update({'dayofweek_'+str(day): 1})
            else:
                ops_params.update({'dayofweek_'+str(day): 0})     
    return ops_params

def operating_start_stop(weekly_df, ops_params, low_pct=5, high_pct=95):
    """
    Given an aggregated time series with a mult-index where the first index is the day of week
    and the second is the time of day along with specific load profile parameters,
    return the start and stop time of typical daily operation.
    """
    # aggregating the weekly median data into a daily mean
    keys = ['dayofweek_0', 'dayofweek_1', 'dayofweek_2',
            'dayofweek_3', 'dayofweek_4', 'dayofweek_5', 'dayofweek_6']
    operating_days = [int(key[-1]) for key in keys if (ops_params[key] == 1)]
    operating_week = weekly_df.loc[(operating_days, ), :].reset_index()
    day_df = operating_week[['avg_kW',
                             'shifted_time_of_day']].groupby(by=['shifted_time_of_day']).mean()
    if ops_params['continuous'] == 1:
        ops_params.update({'daily_start': np.nan,
                           'daily_stop': np.nan,
                           'initial_shifted_daily_start': np.nan,
                           'initial_shifted_daily_stop': np.nan})
        return day_df, ops_params
    else:
        # this method for categorizing on-hours based on a threshold
        # was taken from Lawrence Berkeley National Laboratory
        pct_low = np.percentile(day_df['avg_kW'], low_pct)
        pct_high = np.percentile(day_df['avg_kW'], high_pct)
        threshold = pct_low + 0.25 * (pct_high - pct_low)
        ops_params.update({'on_off_threshold': threshold})
        day_df['on_hour'] = (day_df['avg_kW'] > threshold).astype('int')
        # identifying the start and stop times
        shift_forward_fill = day_df['on_hour'].iloc[-1]
        shift_backward_fill = day_df['on_hour'].iloc[0]
        day_df['start'] = day_df['on_hour'].shift(1, fill_value=shift_forward_fill)
        day_df['start'] = day_df['on_hour'] > day_df['start']
        # taking the earliest potential start time in case the data is noisy 
        # and has intermediate troughs (e.g. site with limited data).
        # Since an ending interval, I'm actually using the preceeding interval
        start_time = day_df.loc[day_df['start']].index[0]
        start_index = list(day_df.index).index(start_time)
        start_time = day_df.index[start_index-1]
        start_time = pd.to_datetime(start_time)
        day_df['stop'] = day_df['on_hour'].shift(-1, fill_value=shift_backward_fill)
        day_df['stop'] = day_df['on_hour'] > day_df['stop']
        # taking the latest stop time for the same reason as the earliest start
        stop_time = pd.to_datetime(day_df.loc[day_df['stop']].index[-1])
        # saving the start times in shifted time
        ops_params.update({'initial_shifted_daily_start': start_time.strftime('%H:%M:%S'),
                           'initial_shifted_daily_stop': stop_time.strftime('%H:%M:%S')})
        # transforming the times back to the original timeframe for final reporting
        time_shift = operating_week['time_shifted'][0]
        start_time -= time_shift
        stop_time -= time_shift
        start_time = start_time.strftime('%H:%M:%S')
        stop_time = stop_time.strftime('%H:%M:%S')
        ops_params.update({'daily_start': start_time,
                           'daily_stop': stop_time})
        return day_df, ops_params

def operating_day_identifier(working_ts, ops_params, near_base_pctl=5, near_peak_pctl=95):
    """
    Given a site's Average kW time series and select operating characteristics,
    shift the time series so the 0-hour is the previously determined typical daily
    start time (only if it isn't continuous) 
    then determine which dates the site is operating and only retain those days.
    """
    ref_peak = ops_params['ref_peak']
    ref_base = ops_params['ref_base']
    #extracting just the time from interval
    working_ts['init_interval_end_time'] = working_ts.loc[:, 'interval_end'].dt.time
    # performing the shift on daily data and generating some useful parameters
    if ops_params['continuous'] == 0:
        init_time_of_day = ops_params['initial_shifted_daily_start']
        intial_times = working_ts.loc[working_ts['shifted_time_of_day']==init_time_of_day]
        init_time = intial_times['shifted_interval'][0]
        shifted_time = init_time.replace(hour=0, minute=0, second=0)
        time_shift = shifted_time - init_time
        working_ts['shifted_interval'] = working_ts['shifted_interval'] + time_shift
        working_ts['time_shifted'] = time_shift
        working_ts['shifted_date'] = working_ts['shifted_interval'].dt.date
        working_ts['shifted_day_of_week'] = working_ts['shifted_interval'].dt.dayofweek
        working_ts['shifted_time_of_day'] = working_ts['shifted_interval'].dt.strftime('%H:%M:%S')
        ops_params.update({'dr_time_shift': time_shift})
    else:
        ops_params.update({'dr_time_shift': pd.Timedelta('0 days 00:00:00')})
    # checking whether a given day appears to be an operating day or not
    day_grouped = working_ts.groupby(by=['shifted_date'])
    base_dict = {}
    peak_dict = {}
    ops_dict = {}
    # looping through the groups and generating daily parameters and performing the same
    # method as with the aggregate weekly data to determin operation.
    for name, group in day_grouped:
        day_near_base_load = np.percentile(group['avg_kW'], near_base_pctl)
        day_near_peak_load = np.percentile(group['avg_kW'], near_peak_pctl)
        delta_daily_peak_ref_peak = ref_peak - day_near_peak_load
        delta_daily_peak_ref_base = day_near_peak_load - ref_base
        if delta_daily_peak_ref_base > delta_daily_peak_ref_peak:
            operating = 1
        else:
            operating = 0
        base_dict.update({name: day_near_base_load})
        peak_dict.update({name: day_near_peak_load})
        ops_dict.update({name: operating})
    # mapping the parameters to the dataframe
    working_ts['day_near_base_load'] = working_ts['shifted_date'].map(base_dict)
    working_ts['day_near_peak_load'] = working_ts['shifted_date'].map(peak_dict)
    working_ts['operating'] = working_ts['shifted_date'].map(ops_dict)
    # removing non-operating days
    working_ts = working_ts.loc[working_ts['operating'] == 1].copy()
    return working_ts

def potential_dr_periods(working_ts, ops_params, min_consec_dr_hours=2, min_consec_high_hours=5):
    """
    Given the time series of operating days shifted such that 
    the typical daily start is the 0:00 hour if not continuous
    otherwise the first hour in the time series is set to the 0-hour,
    return a the time series with a DataFrame containing dates of
    potential DR events based on whether the days contiguous high load
    period had consecutive periods of apparently non-operation 
    greater than or equal tomin_consec_dr_hours and consecutive periods
    of on-operation greater than or equal to min_consec_high_hours.
    """
    dayoweek_grouped = working_ts.groupby(by=['shifted_day_of_week'])
    high_datetimes = []
    days_present = []
    # removing non-high load hours by looping through each day of week and retaining the index values
    # for each days high load period. They're checked individualy to allow for intra-week variation
    for name, group in dayoweek_grouped:
        # My current process for identifying DR events relies on the calculated daily start time
        # being before the daily stop time.
        # Not all sites, particularly some continuous ones have had this addressed at this point in time, 
        # so they aren't considered.
        day_high_start = ops_params['high_start_'+str(name)]
        day_high_stop = ops_params['high_stop_'+str(name)]
        days_present.append(name)
        if day_high_start > day_high_stop:
            pass
        else:
            shifted_dt_times_of_day = pd.to_datetime(group['shifted_time_of_day']).dt.time
            high_hours = group.loc[(shifted_dt_times_of_day >= day_high_start)
                                   & (shifted_dt_times_of_day <= day_high_stop)]
            high_datetimes += list(high_hours.index)
    # only retaining high load hours
    working_ts = working_ts.loc[working_ts.index.isin(high_datetimes) == True].copy()
    # adding column with 1 if considered on-operation interval 
    # and 0 if considered non-operation during expected high load times
    if ops_params['continuous'] == 0:
        working_ts['kW_on'] = working_ts['avg_kW'] > ops_params['on_off_threshold']
    else:
        working_ts['kW_on'] = working_ts['avg_kW'] > ops_params['hi_start_stop_threshold']
    working_ts['kW_on'] = working_ts['kW_on'].astype('int')
    
    # now only considering dates with a full compliment of high load hours 
    # and taking rolling sums of whether an hour is on-operation or not
    width = ops_params['interval_width']
    req_hi_ints = int((min_consec_high_hours * 3600) / width)
    req_lo_ints = int((min_consec_dr_hours * 3600) / width)
    day_grouped = working_ts.groupby(by=['shifted_date'])
    potential_dr_times = []
    for name, group in day_grouped:
        dayoweek = group['shifted_day_of_week'][0]
        day_high_length = ops_params['high_length_intervals_'+str(dayoweek)]
        # if a full high load period then check rolling sums
        if group.shape[0] == day_high_length:
            consecutive_highs = group.loc[::-1,'kW_on'].rolling(window=req_hi_ints,
                                                                min_periods=req_hi_ints).apply(lambda x: sum(x))
            consecutive_lows = group.loc[::-1,'kW_on'].rolling(window=req_lo_ints,
                                                               min_periods=req_lo_ints).apply(lambda x: sum(x))
            # if there appears to be normal operation during part of the day 
            # (i.e. required consecutive on-hours is met) and
            # there is an extended (i.e. req_lo_ints) period of off-hours
            # then those hours are marked as potential DR hours.
            if (consecutive_highs.max() == req_hi_ints 
                and consecutive_lows.min() == 0):
                dr_hours = list(consecutive_lows.loc[consecutive_lows==0].index)
                potential_dr_times += dr_hours
    # since only the hours where there is a potential DR event with successive non-operation kWs
    # at least as long as the full req_lo_ints, the last several intervals of the potential DR event
    # aren't captured and must be calculated to ensure that when shifted back to the original time
    # series timeframe, all potential hours are flagged in case a DR even extended over two days
    # due to the timing of their high-load operation in which case both days should be flagged.
    width_mins = width / 60
    final_dr_times_list = []
    for x in range(0, req_lo_ints):
        x_width_mins = str(int(x * width_mins))
        next_int_time = [time + pd.Timedelta(f'0 days 00:{x_width_mins}:00') for time in potential_dr_times]
        final_dr_times_list += next_int_time
    # removing duplicates and sorting
    final_dr_times_list = list(set(final_dr_times_list))
    final_dr_times_list.sort()
    # now generating the dataframe with dates instead of times
    if len(final_dr_times_list) == 0:
        dr_days_df = pd.DataFrame()
    else:
        dr_days_df = pd.DataFrame(final_dr_times_list, columns=['dr_times'])
        dr_days_df['site_id'] = working_ts['site_id'][0]
        dr_days_df['potential_dr_dates'] = dr_days_df['dr_times'].dt.date
        dr_days_df.drop(columns=['dr_times'], inplace=True)
        dr_days_df.drop_duplicates(inplace=True)
    return working_ts, dr_days_df

# I realized both the operating start/stop and DR identification processes used
# a lot of the same data, so I combined them into one function to prevent redundancy 
# and for expediency of my development. Future work could be performed
# to separate them and only perform necessary operations within each.
def site_operating_and_dr_status(site_data_df,
                                 site_id_list,
                                 p_threshold=0.05,
                                 min_consec_dr_hours=2,
                                 min_consec_high_hours=5,
                                 base_load_pctl=15,
                                 near_base_pctl=5,
                                 near_peak_pctl=95,
                                 low_pct=5,
                                 high_pct=95):
    """
    Given site-level Average kW time series',
    categorize the sites as operating continuously or not, 
    determine daily start/stop if not continuously operating,
    identify days when there was potentially a demand response (DR) event,
    and then return this information by site in one dataframe for 
    daily/continuous information and one for DR information.
    """
    ops_data = pd.DataFrame()
    dr_data = pd.DataFrame()
    for site in site_id_list:
        ops_params = {}
        # subsetting to a single site. In retrospect I should have done a groupby to avoid
        # locating each site on each loop
        working_df = site_data_df.loc[site_data_df['site_id'] == site].copy()
        working_df.reset_index(inplace=True, drop=True)
        width = working_df['interval_width'].mean()
        # dropping less frequent interval if not the same for entire time series
        # as I noticed relatively late that the data before and after the interval change
        # didn't appear to line up properly and I didn't have the time to figure it out
        if width.is_integer() == False:
            drop_int = working_df['interval_width'].value_counts().sort_values().index[0]
            working_df = working_df.loc[working_df['interval_width'] != drop_int]
        working_ts = working_df.set_index('interval_end', drop=False)
        working_ts.dropna(inplace=True)
        # at least one example was found where data was provided hourly,
        # but the site had interval widths listed as 15-minute/900-seconds,
        # so interval width is being calculated based on interval timestamps
        width = working_ts['interval_end'].iloc[1] - working_ts['interval_end'].iloc[0]
        width = width.total_seconds()
        working_df.loc[:, 'interval_width'] = width
        ops_params.update({'interval_width': width})
        # creating a weekly time series made up of the median for each interval
        working_week, ops_params = aggregate_week_ts(working_ts, ops_params)
        # calculating near_peak_load and near_base_load parameters
        # as well as reference parameter variables for evaluating operating days
        working_week, ops_params = week_load_profile_params(working_week,
                                                            ops_params,
                                                            base_load_pctl=15,
                                                            near_base_pctl=5,
                                                            near_peak_pctl=95)
        days_index = working_week.index.get_level_values(0).unique()
        # generating select operating parameters for days included in the aggregated week
        ops_params = agg_week_operating_day(working_week,
                                            days_index,
                                            ops_params,
                                            p_threshold=p_threshold)
        # calculating the start/stop times of a typical operating day and concatenating the results
        working_day, ops_params = operating_start_stop(working_week,
                                                       ops_params,
                                                       low_pct=low_pct,
                                                       high_pct=high_pct)
        site_ops_data = pd.DataFrame(ops_params, index=[site])
        ops_data = pd.concat([ops_data, site_ops_data])
        
        # Movning on to Question 2, Demand Response Events

        # determining which dates in the entire sites time series are operating days,
        # and shifting time series daily 0-hour to be daily start time,
        # so entire high load periods should be in a single date to facilitate grouping.
        working_ts_dr = operating_day_identifier(working_ts,
                                                 ops_params,
                                                 near_base_pctl=near_base_pctl,
                                                 near_peak_pctl=near_peak_pctl)
        # creating aggregate weekly parameters for the data in new shifted time
        working_week_dr, ops_params = aggregate_week_ts(working_ts_dr, ops_params, dr=True)
        working_week_dr, ops_params = week_load_profile_params(working_week_dr,
                                                               ops_params,
                                                               dr=True,
                                                               base_load_pctl=15,
                                                               near_base_pctl=5,
                                                               near_peak_pctl=95)
        # identifying the potential DR periods based on consecutive non-operating hours 
        # in expected high load periods that also exhibited consecutive operating hours
        working_ts_dr, site_dr_days = potential_dr_periods(working_ts_dr,
                                                           ops_params,
                                                           min_consec_dr_hours=min_consec_dr_hours,
                                                           min_consec_high_hours=min_consec_high_hours)
        # generating output dataframes
        site_ops_data = pd.DataFrame(ops_params, index=[site])
        dr_data = pd.concat([dr_data, site_dr_days])
    # final preparation of outputs
    ops_data['continuous'] = ops_data['continuous'].astype('bool')
    dr_data.set_index('site_id', inplace=True, drop=True)
    return ops_data[['continuous', 'daily_start', 'daily_stop']], dr_data

def plot_site_operating_status(site_data_df,
                               site_id_list,
                               p_threshold=0.05,
                               base_load_pctl=15,
                               near_base_pctl=5,
                               near_peak_pctl=95,
                               low_pct=5,
                               high_pct=95):
    """
    Given site-level Average kW time series',
    categorize the sites as operating continuously or not, 
    determine daily start/stop if not continuously operating,
    and then plot an aggregated weekly profile with start and
    stop times for each site. This is a modified version of the
    site_operating_and_dr_status function used for plotting
    in a jupyter notebook.
    """
    ops_data = pd.DataFrame()
    for site in site_id_list:
        ops_params = {}
        # subsetting to a single site. In retrospect I should have done a groupby to avoid
        # locating each site on each loop
        working_df = site_data_df.loc[site_data_df['site_id'] == site].copy()
        working_df.reset_index(inplace=True, drop=True)
        width = working_df['interval_width'].mean()
        # dropping less frequent interval if not the same for entire time series
        # as I noticed relatively late that the data before and after the interval change
        # didn't appear to line up properly and I didn't have the time to figure it out
        if width.is_integer() == False:
            drop_int = working_df['interval_width'].value_counts().sort_values().index[0]
            working_df = working_df.loc[working_df['interval_width'] != drop_int]
        working_ts = working_df.set_index('interval_end', drop=False)
        working_ts.dropna(inplace=True)
        # at least one example was found where data was provided hourly,
        # but the site had interval widths listed as 15-minute/900-seconds,
        # so interval width is being calculated based on interval timestamps
        width = working_ts['interval_end'].iloc[1] - working_ts['interval_end'].iloc[0]
        width = width.total_seconds()
        working_df.loc[:, 'interval_width'] = width
        ops_params.update({'interval_width': width})
        # creating a weekly time series made up of the median for each interval
        working_week, ops_params = aggregate_week_ts(working_ts, ops_params)
        # calculating near_peak_load and near_base_load parameters
        # as well as reference parameter variables for evaluating operating days
        working_week, ops_params = week_load_profile_params(working_week,
                                                            ops_params,
                                                            base_load_pctl=15,
                                                            near_base_pctl=5,
                                                            near_peak_pctl=95)
        days_index = working_week.index.get_level_values(0).unique()
        # generating select operating parameters for days included in the aggregated week
        ops_params = agg_week_operating_day(working_week,
                                            days_index,
                                            ops_params,
                                            p_threshold=p_threshold)
        # calculating the start/stop times of a typical operating day and concatenating the results
        working_day, ops_params = operating_start_stop(working_week,
                                                       ops_params,
                                                       low_pct=low_pct,
                                                       high_pct=high_pct)
        site_ops_data = pd.DataFrame(ops_params, index=[site])
        ops_data = pd.concat([ops_data, site_ops_data])
        
        # generating the plots of the aggregated week with lines for start and stop
        # as applicable
        daily_intervals = 24 / (working_week['imputed_interval_width'].iloc[0] / 3600)
        days_in_data = str(int(working_ts.shape[0] / daily_intervals))
        plt.figure(figsize=[15, 4])
        working_week[['avg_kW']].plot()
        try:
            start_time = ops_params['initial_shifted_daily_start']
            stop_time = ops_params['initial_shifted_daily_stop']
            plt_start = list(working_day.index).index(start_time)
            plt_stop = list(working_day.index).index(stop_time)
            for i in range(ops_params['agg_week_num_days']):
                plt.gca().axvline(plt_start+(daily_intervals*i), color='c')
                plt.gca().axvline(plt_stop+(daily_intervals*i), color='m')
        except:
            pass
        plt.gca().set_title('Site ID = '+site
                            +'\n Continuous = '+str(bool(ops_params['continuous']))
                            +'\n Start Time ='+str(start_time)
                            +'\n Stop Time ='+str(stop_time))
        plt.xticks(rotation=45);
    plt.show();
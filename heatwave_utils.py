"""Shared utilities for heatwave detection and climate statistics."""

import pandas as pd


def detect_heatwaves(tmax, abs_threshold=20.0, min_duration=3):
    """
    Detect heatwaves using a scenario-specific 90th-pct JJA threshold plus an
    absolute temperature floor.

    Parameters
    ----------
    tmax          : pd.Series  daily maximum temperature (°C)
    abs_threshold : float      absolute minimum temperature for a heatwave day
    min_duration  : int        minimum consecutive days to qualify as a heatwave

    Returns
    -------
    threshold : float          computed 90th-pct JJA Tmax value
    hw_df     : pd.DataFrame   heatwave events sorted by surplus descending
    """
    threshold = tmax[tmax.index.month.isin([6, 7, 8])].quantile(0.90)
    hot = (tmax > threshold) & (tmax >= abs_threshold)

    events = []
    in_hw, start = False, None
    for i, (date, is_hot) in enumerate(zip(hot.index, hot.values)):
        if is_hot and not in_hw:
            in_hw, start = True, date
        elif not is_hot and in_hw:
            end = hot.index[i - 1]
            spell = tmax[start:end]
            if len(spell) >= min_duration:
                events.append({'start': start, 'end': end,
                               'duration_days': len(spell),
                               'peak_T_C':      round(float(spell.max()), 2),
                               'surplus_Cdays': round(float((spell - threshold).sum()), 2)})
            in_hw = False
    if in_hw:
        spell = tmax[start:]
        if len(spell) >= min_duration:
            events.append({'start': start, 'end': hot.index[-1],
                           'duration_days': len(spell),
                           'peak_T_C':      round(float(spell.max()), 2),
                           'surplus_Cdays': round(float((spell - threshold).sum()), 2)})

    hw_df = (pd.DataFrame(events)
               .sort_values('surplus_Cdays', ascending=False)
               .reset_index(drop=True))
    hw_df.index += 1
    return threshold, hw_df


def climate_stats(df, hw_df, n_years):
    """
    Compute climate statistics for one (location, scenario) combination.

    Parameters
    ----------
    df      : pd.DataFrame  daily data — columns: tmax_C, tmean_C, tmin_C,
                            hurs_mean, sfcWind_mean, clt_mean
    hw_df   : pd.DataFrame  heatwave events table from detect_heatwaves()
    n_years : int           number of years in the record

    Returns
    -------
    dict of abbreviated metric name -> value
    """
    tmax = df['tmax_C']
    tmin = df['tmin_C']

    jja = df[df.index.month.isin([6, 7, 8])]
    djf = df[df.index.month.isin([12, 1, 2])]

    hw_mask = pd.Series(False, index=df.index)
    for _, row in hw_df.iterrows():
        hw_mask[row['start']:row['end']] = True
    hw_days = df[hw_mask]

    peak_tmin = ([tmin[r['start']:r['end']].max() for _, r in hw_df.iterrows()]
                 if len(hw_df) else [])

    nan = float('nan')
    return {
        'Tmax mean (°C)':         round(tmax.mean(), 2),
        'Tmax JJA (°C)':          round(jja['tmax_C'].mean(), 2),
        'Tmax DJF (°C)':          round(djf['tmax_C'].mean(), 2),
        'Tmax record (°C)':       round(tmax.max(), 2),
        'Tmean (°C)':             round(df['tmean_C'].mean(), 2),
        'Tmean JJA (°C)':         round(jja['tmean_C'].mean(), 2),
        'Tmean DJF (°C)':         round(djf['tmean_C'].mean(), 2),
        'Tmin mean (°C)':         round(tmin.mean(), 2),
        'Tmin JJA (°C)':          round(jja['tmin_C'].mean(), 2),
        'Tmin DJF (°C)':          round(djf['tmin_C'].mean(), 2),
        'Tmin record (°C)':       round(tmin.min(), 2),
        'Sum.days ≥25°C /yr':     round((tmax >= 25).sum() / n_years, 1),
        'Hot days ≥30°C /yr':     round((tmax >= 30).sum() / n_years, 1),
        'Trop.nights ≥20°C /yr':  round((tmin >= 20).sum() / n_years, 1),
        'Humidity (%)':           round(df['hurs_mean'].mean(), 1),
        'Wind (m/s)':             round(df['sfcWind_mean'].mean(), 2),
        'Cloud frac. (%)':        round(df['clt_mean'].mean(), 1),
        'HW events':              len(hw_df),
        'HW events /yr':          round(len(hw_df) / n_years, 2),
        'HW days total':          int(hw_mask.sum()),
        'HW duration (d)':        round(hw_df['duration_days'].mean(), 1) if len(hw_df) else 0,
        'HW peak Tmax (°C)':      round(hw_df['peak_T_C'].mean(), 2) if len(hw_df) else nan,
        'HW peak Tmin (°C)':      round(pd.Series(peak_tmin).mean(), 2) if peak_tmin else nan,
        'HW surplus (°C·d)':      round(hw_df['surplus_Cdays'].mean(), 2) if len(hw_df) else 0,
        'Tmax on HW days (°C)':   round(hw_days['tmax_C'].mean(), 2) if len(hw_days) else nan,
        'Tmin on HW days (°C)':   round(hw_days['tmin_C'].mean(), 2) if len(hw_days) else nan,
        'Humid. on HW days (%)':  round(hw_days['hurs_mean'].mean(), 1) if len(hw_days) else nan,
        'Wind on HW days (m/s)':  round(hw_days['sfcWind_mean'].mean(), 2) if len(hw_days) else nan,
    }

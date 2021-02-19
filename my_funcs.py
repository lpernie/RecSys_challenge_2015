import os
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt


def load_file(f_path, to_be_sorted=False, header=None, index_col=None, names=None, dtype=None, limit=None,
              parse_dates=None, date_parser=None, converters=None):
    
    # limit can be used on nrows only if it is sorted.
    nrows = limit if not to_be_sorted else None
    # Read the file
    df = pd.read_csv(f_path, header=header, index_col=index_col,
                     names=names, dtype=dtype, parse_dates=parse_dates,
                     date_parser=date_parser, converters=converters,
                     nrows=nrows
                    )
    
    if to_be_sorted:
        df = df.sort_values(by=["DateTime", "SessionID"])
        if limit:
            df = df.iloc[:limit]      
    return df


def sanity_checks(df, n: int = 5) -> int:
    print(df.head(n=n))
    for col in df.columns:
        print(f'------ {col.upper()} ------')
        print(df[col].describe(datetime_is_numeric=True))
        print('Null are: ',df[col].isnull().sum())
        print('Has this column any duplicates?', df[col].duplicated().any())
    return 0


def create_plots(df, out_folder : str ='./figs/', col_to_exclude : list = [], params=None):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    cols = [x for x in df.columns if not x in col_to_exclude ]
    for c in cols:
        if params.get(c, {}).get('plot', None) == 'date_bar_YM':
            par = params[c]
            df.groupby([df[c].dt.year, df[c].dt.month]).count().plot(kind='bar', y=par.get('y', 1), logy=par.get('logy', False), grid=par.get('grid', True),
                                                                     xlabel=par.get('xlabel', 'Date'), ylabel=par.get('ylabel', 'Count'))
            if par.get('save', True):
                plt.savefig(out_folder + '/' + par.get('name', c + '.png'), bbox_inches = "tight", dpi=300)
            if par.get('show', True):
                plt.show()
            plt.clf()
        if params.get(c, {}).get('plot', None) == 'bar':
            par = params[c]
            df.groupby(df[c]).count().plot(kind='bar', y=par.get('y', 1), logy=par.get('logy', False), grid=par.get('grid', True),
                                           xlabel=par.get('xlabel', 'Category'), ylabel=par.get('ylabel', 'Counts'))
            if par.get('save', True):
                plt.savefig(out_folder + '/' + par.get('name', c + '.png'), bbox_inches = "tight", dpi=300)
            if par.get('show', True):
                plt.show()
            plt.clf()
        if params.get(c, {}).get('plot', None) == 'hist':
            par = params[c]
            _ = plt.hist(df[c], bins=par.get('bins', 'auto'))
            plt.xlabel(par.get('xlabel', 'X'))
            plt.ylabel(par.get('ylabel', 'Count'))
            plt.grid(par.get('grid', True))
            plt.yscale( 'log' if par.get('logy', False) else 'linear' )
            if par.get('save', True):
                plt.savefig(out_folder + '/' + par.get('name', c + '.png'), bbox_inches = "tight", dpi=300)
            if par.get('show', True):
                plt.show()
            plt.clf()
    return None


def process_clicks(clicks):
    print("Processing clicks")
    # Shift each group by periods observations (default is period=1). First click per session has NaN.
    # __DataFrameGroupBy.shift returns a 'DataFrame' where the function shift is applied per group.
    clicks['prev_DateTime'] = clicks.groupby('SessionID')['DateTime'].shift()
    clicks['diff_DateTime'] = clicks["DateTime"] - clicks["prev_DateTime"] # It's in 'dt' format
    # Dwell: DeltaTime of the future click. Last click of the session has NaN. 
    clicks["dwell"] = clicks.groupby('SessionID')['diff_DateTime'].shift(-1).dt.seconds/60  # Type is float (minutes).
    clicks = clicks.sort_values(by=["SessionID", "DateTime"])
    print("Processed clicks shape %s %s" % clicks.shape)
    return clicks


def process_buys(buys):
    # Group into sessions, compute nr of items bought and set label column
    print("Processing buys")
    print("Buys from %s to %s" % (buys.DateTime.min(), buys.DateTime.max()))
    grouped = buys.groupby("SessionID")  # You only group once to save time
    buys_out = pd.DataFrame(index=grouped.groups.keys())
    buys_out["items_bought"] = grouped.ItemID.count() # quantity may be zero which is weird so dont use it
    buys_out["is_buy"] = 1 # for easier merge later on
    buys_out.index.name = "SessionID"
    print("Buys grouped by session %s %s" % buys_out.shape)
    return buys_out


def get_items_cats_percent(clicks, buys_path, limit=None):
    buys = load_file(buys_path, limit=limit, to_be_sorted=False, index_col=0, header=0,
                     dtype={'SessionID': np.int32, 'ItemID': np.int32, 'Price':np.int32, 'Quantity':np.int16},
                     parse_dates=[1]
                     )

    # Percent bought
    item_id_bought_pct = buys.ItemID.value_counts(normalize=True)
    cat_id_viewed_pct = clicks.Category.value_counts(normalize=True)
    item_id_viewed_pct = clicks.ItemID.value_counts(normalize=True)

    return dict(views=dict(item=item_id_viewed_pct, cat=cat_id_viewed_pct), buys=item_id_bought_pct)


def process_sessions(processed_clicks, buys_path, limit=None):
    print("Preprocessing - Grouping clicks into sessions")
    clicks = processed_clicks
    
    # Group clicks by session
    grouped = clicks.groupby("SessionID")
    sessions = pd.DataFrame(index=grouped.groups.keys())
    
    # Number of Items/Categories/Clicks per session
    sessions["total_clicks"] = grouped.ItemID.count()
    sessions["total_items"] = grouped.ItemID.nunique() # N of unique Items
    sessions["total_cats"] = grouped.Category.nunique()
    print("Computed counters")
    
    # Session time-stats
    sessions["max_dwell"] = grouped.dwell.max()
    sessions["mean_dwell"] = grouped.dwell.mean()
    sessions["start_ts"] = grouped.DateTime.min()
    sessions["end_ts"] = grouped.DateTime.max()
    sessions["total_duration"] = (sessions["end_ts"] - sessions["start_ts"]).dt.seconds / 60  # In minutes
    print("Computed dwell and duration")
    
    # Click rate per session
    sessions["total_duration_secs"] = (sessions["end_ts"] - sessions["start_ts"]).dt.seconds
    sessions["click_rate"] = sessions["total_clicks"] / sessions["total_duration_secs"]
    sessions.click_rate = sessions.click_rate.replace(np.inf, np.nan)  # Replace inf with NaN
    sessions.click_rate = sessions.click_rate.fillna(0)  # Replace Nan with 0
    del sessions["total_duration_secs"]
    print("Computed click rate")
    
    # What is the item and the category most viewed in each session? How many times were they viewed?
    # The code does the following:
    # . clicks.groupby('SessionID')['Category'].value_counts() gives: 'Index1=SessionID Index2=Category Count'
    #   . For one SessionID you have all unique Category values that Session has, and on the side you have the count
    # . rename_axis simply renames the index SessionID to cat_most_viewed_n_times, since now it is cat_most_viewed_n_times
    # . reset_index add a default index, transform SessionID and cat_most_viewed_n_times in normal columns, and the count is now called cat_most_viewed
    # . Then you simply drop the duplicates ton only keep the most clicked category and the count
    sessions[['cat_most_viewed_n_times', 'cat_most_viewed']] = clicks.groupby('SessionID')['Category'].value_counts()\
             .rename_axis(['SessionID','cat_most_viewed_n_times'])\
             .reset_index(name='cat_most_viewed')\
             .drop_duplicates('SessionID').set_index('SessionID')[['cat_most_viewed_n_times','cat_most_viewed']]
    
    sessions[['item_most_viewed_n_times', 'item_most_viewed']] = clicks.groupby('SessionID')['ItemID'].value_counts()\
             .rename_axis(['SessionID','item_most_viewed_n_times'])\
             .reset_index(name='item_most_viewed')\
             .drop_duplicates('SessionID').set_index('SessionID')[['item_most_viewed_n_times','item_most_viewed']]
    print("Computed most viewed item/cat per session")
    
    # For the item most viewed in each session, what is its global buy/view frequency?
    freqs = get_items_cats_percent(clicks, buys_path, limit=limit)
    cat_views = pd.DataFrame(freqs["views"]["cat"])
    cat_views.columns = ["cat_views_freqs"]
    sessions = sessions.merge(cat_views, how="left", left_on="cat_most_viewed", right_index=True)
    sessions.cat_views_freqs = sessions.cat_views_freqs.fillna(0)
    item_views = pd.DataFrame(freqs["views"]["item"])
    item_views.columns = ["item_views_freqs"]
    sessions = sessions.merge(item_views, how="left", left_on="item_most_viewed", right_index=True)
    sessions.item_views_freqs = sessions.item_views_freqs.fillna(0)
    item_buys = pd.DataFrame(freqs["buys"])
    item_buys.columns = ["item_buys_freqs"]
    sessions = sessions.merge(item_buys, how="left", left_on="item_most_viewed", right_index=True)
    sessions.item_buys_freqs = sessions.item_buys_freqs.fillna(0)
    print("Computed most viewed/bought freqs")
    
    # Sorting sessions
    sessions = sessions.sort_values(by=["start_ts"])
    sessions.index.name = "SessionID"
    
    print("Sessions shape %s %s" % sessions.shape)
    print("Sessions columns %s " % sessions.columns)
    print("Sessions from %s to %s" % (sessions.start_ts.min(), sessions.start_ts.max()))
    return sessions


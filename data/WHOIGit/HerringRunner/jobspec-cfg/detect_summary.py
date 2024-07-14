#!/usr/bin/env python

#1) list files with glob or text file
#2) count rows in all files
#3) organize files by videoID
#4) output csv of frames w/ detected fishcounts

import os
import sys
import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sn
import datetime as dt


def get_labelfiles_from_txtdir(target):
    if os.path.isfile(target) and target.endswith('.txt'):
        with open(target) as f:
            label_files = [x.strip() for x in f.readlines()]
    elif os.path.isdir(target):
        label_files = [os.path.join(target,x) for x in os.listdir(target) if x.endswith('.txt')]
    else:
        raise FileNotFoundError(target)
    return label_files


def create_df_from_labels(label_files):
    detect_data = []

    for lfile in tqdm(label_files,desc='Processing Detection DF...'):
        frame_id,_ = os.path.splitext(os.path.basename(lfile))
        video_id = frame_id.rsplit('_',1)[0]
        try:
            with open(lfile) as f:
                label_count = len(f.readlines())
        except:
            print('error:',lfile)
            label_count = None
        frame_data = dict(video=video_id, frame=frame_id, count=label_count)
        detect_data.append(frame_data)

    df = pd.DataFrame(detect_data)
    df.set_index('frame', inplace=True)

    return df


cat2count = {1:(0,0,'A'),
             2:(1,4,'B'),
             3:(5,10,'C'),
             4:(11,20,'D'),
             5:(20,40,'E')}
def count2cat(x):
    if x==cat2count[1][1]: return cat2count[1][2]
    elif x<=cat2count[2][1]: return cat2count[2][2]
    elif x<=cat2count[3][1]: return cat2count[3][2]
    elif x<=cat2count[4][1]: return cat2count[4][2]
    elif x>=cat2count[5][0]: return cat2count[5][2]
    else: raise ValueError(x)
        
             
def get_zooniverse_df(csv_positives='test-data/zooniverse_2022-01-28_positives.csv', 
                      csv_negatives='test-data/zooniverse_2022-01-28_negatives.csv'):

    try:
        dfp = pd.read_csv(csv_positives)
        dfn = pd.read_csv(csv_negatives)
    except FileNotFoundError:
        dfp = pd.read_csv('../'+csv_positives)
        dfn = pd.read_csv('../'+csv_negatives)
        
    dfp['frame'] = dfp.frame_name
    dfp['countcat'] = dfp.roundedcount
    dfn['countcat'] = dfn.presence_absence_vol
    dfn.frame = dfn.frame.apply(lambda x:x.rsplit('_',1)[0])
    
    cols = ['video','frame','countcat']
    df = pd.concat([dfp[cols],dfn[cols]])
    df.video = df.video.apply(lambda x: x[:-1])
    df = df.sort_values('frame')
    df.reset_index()
    
    df['countmin'] = df.countcat.apply(lambda x: cat2count[x][0])
    df['countmax'] = df.countcat.apply(lambda x: cat2count[x][1])
    df['countcat'] = df.countcat.apply(lambda x: cat2count[x][2])
    df['countavg'] = df[['countmin','countmax']].mean(axis=1).round().astype(int)

    df.set_index('frame',inplace=True)

    return df
    

def do_zooniverse(df_detect):
    
    df = get_zooniverse_df()
    df['presence'] = df['countcat'].apply(lambda x: 0 if x=='A' else 1)
    df['model_count'] = df_detect['count']
    df['model_count'] = df['model_count'].fillna(0).astype(int)

    df['model_cat'] = df['model_count'].apply(count2cat)
    df['model_presence'] = df['model_cat'].apply(lambda x: 0 if x=='A' else 1)
    
    df_sum = df.groupby('video').sum()
    df_sum['frames_total'] = df.groupby('video').count()['presence']
    df_sum.loc['TOTALS',:]= df.sum(axis=0)

    return df, df_sum


def do_testset(df_detect, testset):
    #1 get list of label files
    if os.path.isdir(testset):
        label_files2 = get_labelfiles_from_txtdir(testset)
    else:
        with open(testset) as f:
            label_files2 = [line.strip() for line in f.readlines()]
    label_files2 = [f.replace('/images/','/labels/').replace('.jpg','.txt') for f in label_files2]
    
    #2 get the counts per frame per video from each datafile (number of rows in each file)
    df = create_df_from_labels(label_files2)
    #COLUMNS: frame, video, count
    
    #3 combine and standardize model results and test results
    df['countcat'] = df['count'].apply(count2cat)
    df['presence'] = df['countcat'].apply(lambda x: 0 if x=='A' else 1)
    
    df['model_count'] = df_detect['count']
    df['model_count'] = df['model_count'].fillna(0).astype(int)
    df['model_cat'] = df['model_count'].apply(count2cat)
    df['model_presence'] = df['model_cat'].apply(lambda x: 0 if x=='A' else 1)
    
    #4 make the summary
    df_sum = df.groupby('video').sum()
    df_sum['frames_total'] = df.groupby('video').count()['presence']
    df_sum.loc['TOTALS',:]= df.sum(axis=0)

    return df, df_sum


def proc_confmats(df, plot_outdir=None, model_name=None):

    videos = sorted(set(df['video']))
    norm_axis = 'index'
    confmats = dict(category_count=dict(),
                    category_norm=dict(),
                    presence_count=dict(),
                    presence_norm=dict())
    
    # ABCDE confusion matrix total
    ABCDE='A B C D E'.split()
    confmats['category_count']['agg'] = pd.crosstab(df['countcat'],df['model_cat'], dropna=False)
    confmats['category_norm']['agg']  = pd.crosstab(df['countcat'],df['model_cat'], dropna=False, normalize=norm_axis)
    confmats['presence_count']['agg'] = pd.crosstab(df['presence'], df['model_presence'])
    confmats['presence_norm']['agg']  = pd.crosstab(df['presence'], df['model_presence'], normalize=norm_axis)
    for video in videos:
        confmats['category_count'][video] = pd.crosstab(df[df['video']==video]['countcat'],
                                                        df[df['video']==video]['model_cat'],
                                                        dropna=False).reindex(index=ABCDE, columns=ABCDE, fill_value=0)
        confmats['category_norm'][video] = pd.crosstab(df[df['video']==video]['countcat'],
                                                       df[df['video']==video]['model_cat'],
                                                       dropna=False, normalize=norm_axis).reindex(index=ABCDE, columns=ABCDE, fill_value=0)
        confmats['presence_count'][video] = pd.crosstab(df[df['video']==video]['presence'],
                                                        df[df['video']==video]['model_presence'])
        confmats['presence_norm'][video] = pd.crosstab(df[df['video']==video]['presence'],
                                                       df[df['video']==video]['model_presence'],
                                                       normalize=norm_axis)
    
    if plot_outdir:
        t1 = (confmats['category_count']['agg'], 'Category frame-counts aggregate')
        t2 = (confmats['presence_count']['agg'], 'Presence/Absence frame-counts aggregate')
        t3 = (confmats['category_norm']['agg'], 'Category frame-counts \naggregate (normalized)')
        t4 = (confmats['presence_norm']['agg'], 'Presence/Absence frame-counts \naggregate (normalized)')
        #plot_confmat(t1, os.path.join(plot_outdir,'confmat_categories.counts.agg.png'))
        #plot_confmat(t2, os.path.join(plot_outdir,'confmat_presence.counts.agg.png'))
        #plot_confmat(t3, os.path.join(plot_outdir,'confmat_categories.norm.agg.png'))
        #plot_confmat(t4, os.path.join(plot_outdir,'confmat_presence.norm.agg.png'))

        col1,col2 = [t1,t3],[t2,t4]
        fig_title = f'Aggregate Stats for model: "{model_name}"' if model_name else None
        plot_confmat([col1,col2], os.path.join(plot_outdir,'confmats_aggregated.png'), fig_title=fig_title)
                               
        # confmat,ax-title tuples
        category_count_list = [(confmats['category_count'][v],f'Category frame-counts \nfor {v}') for v in videos]
        presence_count_list = [(confmats['presence_count'][v],f'Presence/Absence frame-counts \nfor {v}') for v in videos]
        category_norm_list  = [(confmats['category_norm'][v],f'Category frame-counts \n(normalized) for {v}') for v in videos]
        presence_norm_list  = [(confmats['presence_norm'][v],f'Presence/Absence frame-counts \n(normalized) for {v}') for v in videos]
        
        fig_title = f'Aggregate Stats per-video for model: "{model_name}"' if model_name else None
        plot_confmat([category_count_list,category_norm_list,presence_count_list,presence_norm_list],
                     os.path.join(plot_outdir,f'confmats_per-video.png'), fig_title=fig_title)
        
        #plot_confmat(category_count_list, os.path.join(plot_outdir,f'confmat_categories.counts.pervideo.png'))
        #plot_confmat(presence_count_list, os.path.join(plot_outdir,f'confmat_presence.counts.pervideo.png'))
        #plot_confmat(category_norm_list, os.path.join(plot_outdir,f'confmat_categories.norm.pervideo.png'))
        #plot_confmat(presence_norm_list, os.path.join(plot_outdir,f'confmat_presence.norm.pervideo.png'))

    return confmats
    
 
def plot_confmat(confmat_col_rows, outfile, fig_title=None):   
    
    if not isinstance(confmat_col_rows,list):
        confmat,title = confmat_col_rows
        fmt = '.2f' if 'norm' in title else '.0f'
        fig, ax = plt.subplots()
        ax.set_aspect(1)
        ax.set_title(title)
        res = sn.heatmap(confmat.T, annot=True, fmt=fmt, cmap="YlGnBu", cbar=False, ax=ax)
        print('Writing:', outfile)
        fig.savefig(outfile, bbox_inches='tight', dpi=100)
        plt.close(fig)
        return
    elif not isinstance(confmat_col_rows[0],list): # single column
        cols=1
        rows=len(confmat_col_rows)
        confmat_col_rows = [confmat_col_rows]
    else:             
        cols = len(confmat_col_rows)
        rows = len(confmat_col_rows[0])

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols,5*rows), squeeze=False)
    fig.tight_layout(h_pad=6)
    if fig_title:
        fig.suptitle(fig_title)
        plt.subplots_adjust(top= 0.90 if rows==2 else 0.96)

    for confmat_row,ax_row in zip(zip(*confmat_col_rows),axes):
        for (confmat,title),ax in zip(confmat_row,ax_row):
            #print(title.replace('\n',' '))
            ax.set_aspect(1)
            ax.set_title(title)
            fmt = '.2f' if 'norm' in title else '.0f'
            res = sn.heatmap(confmat.T, annot=True, fmt=fmt, cmap="YlGnBu", cbar=False, ax=ax)

    print('Writing:',outfile)
    fig.savefig(outfile, bbox_inches='tight', dpi=100)
    plt.close(fig)
        
    
def quote_video(df):
    # spreadsheet editors like to conflate video id's with long numbers that get converted to sci-notation. this prevents that
    if 'video' not in df.columns.names: 
        df = df.reset_index()
        reset=True
    df['video'] = df['video'].apply(lambda v: f"'{v}")
    if reset: 
        df = df.set_index('video')
    return df


def add_metadata(df):

    def frameid2timestamp(f):
        video_id,ms = f.rsplit('_',1) 
        # example video_id: 20170701145052891 --> 2017-07-01 14:50:52.891
        ts = dt.datetime.strptime(video_id+'000','%Y%m%d%H%M%S%f')
        ts = ts + dt.timedelta(milliseconds=int(ms))
        return ts

    df.reset_index(inplace=True)
    df['ts'] = df.frame.apply(frameid2timestamp)
    df.set_index('frame',inplace=True)
    df = df[['video','ts','count']]
    
    try:
        df_daily = pd.read_csv('../weather-daily.csv', index_col='datetime', parse_dates=['sunrise','sunset'])
        df_hourly = pd.read_csv('../weather-hourly.csv', index_col='datetime', parse_dates=['datetime'])
    except FileNotFoundError:
        df_daily = pd.read_csv('weather-daily.csv', index_col='datetime', parse_dates=['sunrise','sunset'])
        df_hourly = pd.read_csv('weather-hourly.csv', index_col='datetime', parse_dates=['datetime'])
    
    daycats = ['   morning','  late-morning',' noon','afternoon','late-afternoon']
    def ts2daycat(ts):
        date = ts.strftime('%Y-%m-%d')
        sunrise = df_daily.loc[date].sunrise
        sunset = df_daily.loc[date].sunset
        sundiff = sunset-sunrise
        sunseg = sundiff/len(daycats)
        cats = ['night']+daycats
        for i in range(len(cats)):
            if ts<sunrise+i*sunseg:
                return cats[i]
        return 'night'

    def cloudcat(val):
        val = val/100
        if val<0.33:
            return 'clear'
        elif val<0.66:
            return 'partly-cloudy'
        elif val<0.88:
            return 'mostly-cloudy'
        else:
            return 'overcast'
    def ts2cloudcat(ts,hourly=True):
        if hourly:
            ts = ts - dt.timedelta(minutes=ts.minute, 
                                   seconds=ts.second, 
                                   microseconds=ts.microsecond)
            val = df_hourly.loc[ts].cloudcover
        else:
            date = ts.strftime('%Y-%m-%d')
            val = df_daily.loc[date].cloudcover
        try: return cloudcat(val)
        except ValueError: 
            print('Warning: Duplicate "{ts}" ts entry. Using first found instance.')
            return cloudcat(val.iloc[0])

    def mooncat(val):
        val = val*100
        if val<12.5: return 'new'
        elif val<12.5*2: return 'crescent'
        elif val<12.5*3: return 'quarter'
        elif val<12.5*4: return 'gibbous'
        elif val<12.5*5: return 'full'
        elif val<12.5*6: return 'gibbous'
        elif val<12.5*7: return 'quarter'
        else: return 'crescent'
    def ts2mooncat(ts):
        date = ts.strftime('%Y-%m-%d')
        val = df_daily.loc[date].moonphase
        return mooncat(val)
    
    df = df.assign(cloudcat = df.ts.apply(ts2cloudcat),
                   daycat   = df.ts.apply(ts2daycat),
                   mooncat  = df.ts.apply(ts2mooncat))
    return df

    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('DETECTIONS', help='Accepts a directory of txt label files, or a text file listing txt label files')
    parser.add_argument('--input-labels', help='Compares DETECTIONS against known labels. Accepts a directory of txt label files, or a text file listing txt label files')
    parser.add_argument('--zooniverse', action='store_true', help='Compares DETECTIONS against zooniverse csv')
    parser.add_argument('--outdir','-o')
    parser.add_argument('--model-name', help='only used for confmats')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--metadata', action='store_true')
    parser.add_argument('--outfile')
    args = parser.parse_args()
    
    # create detections dataframe: video, frame, model_count
    label_files = get_labelfiles_from_txtdir(args.DETECTIONS)
    
    if not label_files:
        print('Error: No Frames found.')
        if args.outdir:
            fname_full = args.outfile or 'results.full.csv'
            fname_full = os.path.join(args.outdir,fname_full)
            print('Writing:',fname_full)
            with open(fname_full,'w') as f:
                header='video,frame,ts,count,cloudcat,daycat,mooncat'
                f.write(header+'\n')
        sys.exit('DONE')
    
    df_detect = create_df_from_labels(label_files)
 
    if args.outdir:
        os.makedirs(args.outdir,exist_ok=True)
            
    # compare against zooniverse
    if args.zooniverse:
        df, df_sum = do_zooniverse(df_detect)
        if args.model_name: confmats = proc_confmats(df, args.outdir, args.model_name)
        
    # compare df_detect against input_labels' data
    elif args.input_labels:
        df, df_sum = do_testset(df_detect, args.input_labels)
        if args.model_name: confmats = proc_confmats(df, args.outdir, args.model_name)

    else:
        df = df_detect
        if args.summary:
            df_sum = df_detect.groupby('video').sum()
            df_sum['frames_total'] = df.groupby('video').count()['presence']
            df_sum.loc['TOTALS',:]= df.sum(axis=0)
    
    # add metadata: cloudcat daycat mooncat
    if args.metadata:
        df = add_metadata(df)
        df.sort_values(by='ts', inplace=True)
    
    # output 
    if args.outdir:
        
        fname_full = args.outfile or 'results.full.csv'
        fname_sum = 'results.sum.csv'
        fname_full = os.path.join(args.outdir,fname_full)
        fname_sum = os.path.join(args.outdir,fname_sum)
        
        print('Writing:',fname_full)
        quote_video(df).to_csv(fname_full)
        if args.summary:
            print('Writing:',fname_sum)
            quote_video(df_sum).to_csv(fname_sum)
    
    

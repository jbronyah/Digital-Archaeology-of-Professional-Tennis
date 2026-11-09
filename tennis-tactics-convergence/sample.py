# %% [markdown]
# # Build and audit the Grand Slam sample
# This stage uses the bundled extracts. Run it before the tactical analysis.

# %%
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from IPython.display import display, Markdown

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'data' / 'snapshot.json').is_file())
OUT = ROOT / 'build' / 'sample'
for sub in ['tables', 'figures']:
    (OUT / sub).mkdir(parents=True, exist_ok=True)
MCP = ROOT / 'data'
snapshot = json.loads((MCP / 'snapshot.json').read_text())
for filename, expected in snapshot['files'].items():
    if hashlib.sha256((MCP / filename).read_bytes()).hexdigest() != expected:
        raise ValueError(f'Input differs from the preserved snapshot: {filename}')
SEED = 20261103
rng = np.random.default_rng(SEED)
plt.rcParams.update({'figure.dpi': 115, 'savefig.dpi': 180, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titlesize': 12, 'font.size': 10})
COLORS = ['#176B87', '#C75B39', '#6D597A', '#4B8753']
plt.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=COLORS)
pd.set_option('display.max_columns', 20)
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')
def read(path, **kwargs):
    return pd.read_csv(path if path.exists() else path.with_suffix('.csv.gz'), low_memory=False, **kwargs)
def table(df, name):
    df.to_csv(OUT / 'tables' / f'{name}.csv', index=False)
    display(df)
def savefig(name):
    plt.savefig(OUT / 'figures' / f'{name}.png', bbox_inches='tight')
    plt.show()
def note(text):
    display(Markdown(text))
def period(year):
    return pd.cut(year, [1994, 1999, 2004, 2009, 2014, 2019, 2025],
                  labels=['1995–99', '2000–04', '2005–09', '2010–14', '2015–19', '2020–25'])

# %% [markdown]
# ## Grand Slam sample and descriptive results
#
# ### Coverage first
#
# Restrict MCP to the four Slams and main-draw singles rounds. Keep ATP and WTA separate. Use five-year periods and a six-year final period (2020–25), since individual early years are sparse. Also export annual counts. A blank cell means no charted matches; it never means a tactic occurred zero times. Coverage ratios use main-draw tour matches as the denominator and can differ from exact linkage coverage because the repositories are independently curated.
#
# %%
round_order = ['R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F']
slams=['Australian Open','Roland Garros','Wimbledon','US Open']
metas=[]
for sex,t in [('m','ATP'),('w','WTA')]:
    d=read(MCP/f'charting-{sex}-matches.csv',encoding='utf-8-sig')
    d['tour']=t;d['date']=pd.to_datetime(d.Date.astype(str),format='%Y%m%d',errors='coerce');d['year']=d.date.dt.year
    d=d[d.Tournament.isin(slams)&d.year.between(1995,2025)&d.Round.isin(round_order)].copy()
    metas.append(d)
meta=pd.concat(metas,ignore_index=True).drop_duplicates('match_id')
meta['period']=period(meta.year)
meta['late_round']=meta.Round.isin(['QF','SF','F'])
coverage=meta.groupby(['tour','Tournament','period'],observed=False).agg(
    matches=('match_id','size'),late_round_share=('late_round','mean')).reset_index()
denom=read(MCP/'archive_slam_counts.csv')
coverage=coverage.merge(denom,on=['tour','Tournament','period'],how='left')
coverage['coverage_pct']=100*coverage.matches/coverage.archive_matches
table(coverage,'q3_charted_coverage')
table(meta.groupby(['tour','Tournament','year']).size().rename('matches').reset_index(),'q3_annual_coverage')
fig,axes=plt.subplots(1,2,figsize=(14,4))
for ax,(t,g) in zip(axes,coverage.groupby('tour')):
    pivot=g.pivot(index='Tournament',columns='period',values='matches').reindex(slams)
    ax.imshow(pivot,aspect='auto',cmap='Blues')
    for (i,j),v in np.ndenumerate(pivot.to_numpy()): ax.text(j,i,str(int(v)),ha='center',va='center',color='white' if v>pivot.to_numpy().max()*.55 else 'black')
    ax.set_xticks(range(len(pivot.columns)),pivot.columns,rotation=30);ax.set_yticks(range(4),slams);ax.set_title(t+' — charted main-draw matches')
fig.tight_layout();savefig('05_mcp_coverage')

# %% [markdown]
# ### Decode points and define serve-plus-one
#
# The supplied point CSVs contain raw strings, so we reconstruct successful-shot rally length. Use the second-serve sequence when present, otherwise the first. Count the serve and shot-type letters, then subtract the terminal attempted shot for an error. Exclude penalties, unknown entire points, malformed strings, and incorrect challenges. `q` is a known shot of unknown type: it counts for rally length, but cannot classify forehand/backhand.
#
# **Serve-plus-one is not serve-and-volley.** Here an opportunity means the server attempted the next shot after a return (the second non-serve shot token). We report (a) opportunities/all valid points, (b) forehand-groundstroke share of known plus-one shot types, (c) clean plus-one winners/opportunities, and (d) immediate plus-one successes, including a subsequent forced returner error. Terminal plus-one errors remain in the opportunity denominator. These are observable patterns, not evidence of tactical intent. A point won after a long rally is not labeled a plus-one success.
#
# For net play we use upstream `NetPoints`, `Approach`, and `ApproachRallies` totals rather than assume every net visit has a `+`. Their sums across both players, divided by total match points, measure **player net involvements/approaches per 100 points**. Both players may approach in the same point; these are not percentages of distinct points. `Approach` includes serve-and-volley; `ApproachRallies` excludes it. An absent subtype row is treated as zero only when that match is represented in the net-points file; completely absent match records remain missing. Serve-and-volley is measured directly from a `+` immediately after the serve code on valid non-ace points, divided by all valid points. This avoids conditioning on matches with a nonzero serve-and-volley aggregate. Optional position notation still limits comparability: zero means no recorded attempt, not necessarily no actual attempt.
#
# Forehand share uses **all known plus-one shot types** as denominator, including volleys. A rise can therefore reflect the replacement of volleys by groundstrokes, not just a greater forehand preference among groundstrokes.
#
# %%
SHOT_LETTERS='fbrsvzopuylmhijktq'
shot_re=re.compile('['+SHOT_LETTERS+']')
allowed=re.compile(r'^[0-9cfbrsvzopuylmhijktqnwdxeg!+\-=;^*#@]+$')
def parse_point(first,second,server,winner):
    text=str(second).strip() if pd.notna(second) and str(second).strip() else str(first).strip()
    if not allowed.fullmatch(text) or not re.match(r'^c*[0456]',text): return None
    shots=shot_re.findall(text)
    error=text.endswith(('@','#'))
    # A fault without a rally (only second-serve faults are valid point endings).
    fault=not shots and bool(re.search(r'[nwdxeg!]$',text))
    if fault:
        if pd.isna(second) or not str(second).strip(): return None
        length=0
    elif text.endswith(('*','#','@')):
        length=1+len(shots)-(1 if error and shots else 0)
    else: return None
    if server not in [1,2] or winner not in [1,2]: return None
    opportunity=len(shots)>=2
    plus_type=shots[1] if opportunity else ''
    clean=opportunity and len(shots)==2 and text.endswith('*') and server==winner
    forced=opportunity and len(shots)==3 and text.endswith('#') and server==winner
    snv=bool(re.match(r'^c*[0456]\+',text)) and not fault and not (not shots and text.endswith('*'))
    return (length,opportunity,plus_type!='q' and opportunity,plus_type in 'frul' if opportunity else False,
            clean,clean or forced,server==winner,snv)

# Meaningful parser checks: ace, double fault, return error, plus-one winner/error/forced error.
assert parse_point('4*',np.nan,1,1)[0]==1
assert parse_point('4n','6w',1,2)[0]==0
assert parse_point('6f#',np.nan,1,1)[0]==1
assert parse_point('6b27f3*',np.nan,1,1)[:2]==(3,True)
assert parse_point('6b27f3n@',np.nan,1,2)[0]==2
assert parse_point('6b27f3b#',np.nan,1,1)[5]
assert parse_point('S',np.nan,1,1) is None
assert parse_point('6bC',np.nan,1,1) is None
assert parse_point('6+b27v*',np.nan,1,1)[7]
assert not parse_point('6+*',np.nan,1,1)[7] # aces excluded from S&V numerator

point_frames=[]; point_audit=[]
target_ids=set(meta.match_id)
for sex in ['m','w']:
    for path in sorted(MCP.glob(f'charting-{sex}-points-*.csv.gz')):
        d=read(path,usecols=['match_id','Pt','1st','2nd','Svr','PtWinner'])
        d=d[d.match_id.isin(target_ids)].copy()
        duplicate=d.duplicated(['match_id','Pt']).sum()
        d=d.drop_duplicates(['match_id','Pt'])
        parsed=[parse_point(a,b,c,e) for a,b,c,e in d[['1st','2nd','Svr','PtWinner']].itertuples(index=False,name=None)]
        keep=np.array([p is not None for p in parsed])
        vals=pd.DataFrame([p for p in parsed if p is not None],columns=['rally','plus_opportunity','plus_known','plus_fh','plus_winner','plus_success','server_won','snv'])
        vals['match_id']=d.loc[keep,'match_id'].to_numpy();vals['Pt']=d.loc[keep,'Pt'].to_numpy()
        point_frames.append(vals)
        per=d.groupby('match_id').size().rename('raw_points').to_frame()
        per['parsed_points']=vals.groupby('match_id').size()
        point_audit.append(per.reset_index())
points=pd.concat(point_frames,ignore_index=True)
assert not points.duplicated(['match_id','Pt']).any()
pa=pd.concat(point_audit).groupby('match_id',as_index=False)[['raw_points','parsed_points']].sum()
pa['parsed_share']=pa.parsed_points/pa.raw_points
table(pa.merge(meta[['match_id','tour','year','Tournament']],on='match_id').groupby(['tour','year']).agg(
    matches=('match_id','size'),raw_points=('raw_points','sum'),parsed_points=('parsed_points','sum')).reset_index(),'q3_parser_coverage')
points['rally_bin']=pd.cut(points.rally,[-1,0,3,6,9,np.inf],labels=['0 (double fault)','1–3','4–6','7–9','10+'])
pm=points.groupby('match_id').agg(parsed_points=('rally','size'),mean_rally=('rally','mean'),
      plus_opportunities=('plus_opportunity','sum'),plus_known=('plus_known','sum'),plus_fh=('plus_fh','sum'),
      plus_winners=('plus_winner','sum'),plus_successes=('plus_success','sum'),snv_points=('snv','sum')).reset_index()
counts=pd.crosstab(points.match_id,points.rally_bin).reset_index()
pm=pm.merge(counts,on='match_id',validate='one_to_one')
pm=pm.merge(pa[['match_id','raw_points','parsed_share']],on='match_id',validate='one_to_one')

# Cross-check parser lengths against independently supplied match-level bins.
upstream=[];net=[];snv=[]
for sex in ['m','w']:
    r=read(MCP/f'charting-{sex}-stats-Rally.csv')
    upstream.append(r[r.match_id.isin(target_ids)&r.row.isin(['Total','1-3','4-6','7-9','10'])])
    n=read(MCP/f'charting-{sex}-stats-NetPoints.csv')
    net.append(n[n.match_id.isin(target_ids)&n.row.isin(['NetPoints','Approach','ApproachRallies'])])
    v=read(MCP/f'charting-{sex}-stats-SnV.csv');snv.append(v[v.match_id.isin(target_ids)])
up_rows=pd.concat(upstream).drop_duplicates()
# Repeated identical exports are harmless; conflicting versions are quarantined.
conflicting_ids=set(up_rows.loc[up_rows.duplicated(['match_id','row'],keep=False),'match_id'])
up_rows=up_rows[~up_rows.match_id.isin(conflicting_ids)]
up=up_rows.pivot(index='match_id',columns='row',values='pts').reset_index()
up[['1-3','4-6','7-9','10']]=up[['1-3','4-6','7-9','10']].fillna(0)
pm=pm.merge(up.rename(columns={'Total':'upstream_points','1-3':'up_short','4-6':'up_mid','7-9':'up_long','10':'up_10'}),on='match_id',how='left')
pm['rally_bin_difference']=(pm['0 (double fault)']+pm['1–3']-pm.up_short).abs()+(pm['4–6']-pm.up_mid).abs()+(pm['7–9']-pm.up_long).abs()+(pm['10+']-pm.up_10).abs()
net_data=pd.concat(net).drop_duplicates()
net_conflicts=set(net_data.loc[net_data.duplicated(['match_id','player','row'],keep=False),'match_id'])
net_data=net_data[~net_data.match_id.isin(net_conflicts)]
net_counts=net_data.groupby(['match_id','row']).agg(n_players=('player','nunique'),net_pts=('net_pts','sum')).reset_index()
net_wide=net_counts.pivot(index='match_id',columns='row',values='net_pts').fillna(0).reset_index()
pm=pm.merge(net_wide,on='match_id',how='left')
snv_data=pd.concat(snv).drop_duplicates()
snv_conflicts=set(snv_data.loc[snv_data.duplicated(['match_id','player','row'],keep=False),'match_id'])
snv_data=snv_data[~snv_data.match_id.isin(snv_conflicts)]
print('Serve-and-volley source columns:',list(snv_data.columns))

# %%
snv_counts=snv_data[snv_data.row.eq('SnV')].groupby('match_id').snv_pts.sum().reindex(snv_data.match_id.unique(),fill_value=0).rename('upstream_snv_points').reset_index()
pm=pm.merge(snv_counts,on='match_id',how='left').merge(meta,on='match_id',validate='one_to_one')
# ≥98% parser coverage, total within 2%, and summed absolute bin discrepancy ≤2%.
pm['eligible']=pm.parsed_share.ge(.98)&((pm.upstream_points-pm.raw_points).abs()/pm.raw_points).le(.02)&(pm.rally_bin_difference/pm.raw_points).le(.02)
pm['rally_approach_points']=pm.ApproachRallies
quality=pm.groupby('tour').agg(matches=('match_id','size'),eligible=('eligible','sum'),
    median_parsed_share=('parsed_share','median'),median_bin_abs_difference=('rally_bin_difference','median')).reset_index()
table(quality,'q3_quality_checks')
note(f'Excluded {len(pm)-int(pm.eligible.sum())} matches after coverage/consistency checks. '
     f'Conflicting aggregate versions quarantined: {len(conflicting_ids)} rally matches, '
     f'{len(net_conflicts)} net matches, {len(snv_conflicts)} serve-and-volley matches. '
     'The upstream bin labeled 1–3 includes double faults; it is checked against reconstructed 0–3 counts.')
table(pm[['match_id','raw_points','parsed_points','parsed_share','upstream_points','rally_bin_difference','eligible']], 'q3_match_quality')
q=pm[pm.eligible].copy()
metrics={'mean_rally':('mean_rally',None),'net_per100':('NetPoints','upstream_points'),
         'approach_per100':('Approach','upstream_points'),'rally_approach_per100':('rally_approach_points','upstream_points'),
         'snv_per100':('snv_points','parsed_points'),
         'plus_opportunity_pct':('plus_opportunities','parsed_points'),
         'plus_fh_pct':('plus_fh','plus_known'),'plus_winner_pct':('plus_winners','plus_opportunities'),
         'plus_success_pct':('plus_successes','plus_opportunities')}
for metric,(num,den) in metrics.items():
    if den:q[metric]=100*q[num]/q[den].replace(0,np.nan)
for label in ['0 (double fault)','1–3','4–6','7–9','10+']:
    q['share_'+label]=100*q[label]/q.parsed_points
table(q[['match_id','tour','Tournament','year','period','Round','parsed_points',*metrics]],'q3_match_metrics')

# Equal match weighting: long matches and best-of-five do not dominate the historical distribution.
summary=[]
for keys,g in q.groupby(['tour','Tournament','period'],observed=True):
    row=dict(zip(['tour','Tournament','period'],keys));row['matches']=len(g);row['points']=g.parsed_points.sum()
    for metric in [*metrics,*['share_'+x for x in ['0 (double fault)','1–3','4–6','7–9','10+']]]:
        values=g[metric].dropna().to_numpy();row[metric]=np.mean(values) if len(values) else np.nan
        row[metric+'_n']=len(values)
        if len(values)>=10:
            draws=rng.integers(0,len(values),(1000,len(values)));lo,hi=np.quantile(values[draws].mean(1),[.025,.975])
        else:lo,hi=np.nan,np.nan
        row[metric+'_low']=lo;row[metric+'_high']=hi
    summary.append(row)
tactics=pd.DataFrame(summary)
table(tactics[['tour','Tournament','period','matches','mean_rally','net_per100','approach_per100','rally_approach_per100','snv_per100',
              'plus_opportunity_pct','plus_fh_pct','plus_winner_pct','plus_success_pct']],'q3_tactics_summary')
tactics.to_csv(OUT/'tables/q3_tactics_with_intervals.csv',index=False)

# %% [markdown]
# ### Read the charts
#
# Each match receives equal weight. Bars split rally length into double faults (0), short (1–3), medium (4–6), longer (7–9), and long (10+) points. Line intervals are 95% bootstrap intervals resampling **matches**, not independent points. Cells with fewer than ten matches are suppressed in charts; their estimates and sample sizes remain in the CSV. These intervals quantify sample variation only: they cannot fix selection bias or repeated-player dependence. Coverage counts are indispensable to any historical interpretation.
#
# %%
periods=['1995–99','2000–04','2005–09','2010–14','2015–19','2020–25']
fig,axes=plt.subplots(2,4,figsize=(17,8),sharey=True)
rally_labels=['0 (double fault)','1–3','4–6','7–9','10+']
for i,t in enumerate(['ATP','WTA']):
    for j,slam in enumerate(slams):
        ax=axes[i,j];g=tactics[(tactics.tour==t)&(tactics.Tournament==slam)].set_index('period').reindex(periods)
        bottom=np.zeros(6)
        for color,label in zip(['#B9C1C9','#176B87','#5A9BB5','#E5AA70','#B7533F'],rally_labels):
            vals=g['share_'+label].where(g.matches>=10)
            ax.bar(range(6),vals,bottom=bottom,label=label,color=color);bottom+=vals.fillna(0)
        ax.set(title=t+' · '+slam,ylim=(0,100));ax.set_xticks(range(6),periods,rotation=60)
        if j==0:ax.set_ylabel('Mean within-match point share (%)')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=5)
fig.tight_layout(rect=(0,0,1,.95));savefig('06_rally_distributions')

def tactic_plot(selected,filename):
    fig,axes=plt.subplots(2,len(selected),figsize=(5*len(selected),8),squeeze=False)
    for i,t in enumerate(['ATP','WTA']):
        for j,(metric,title) in enumerate(selected):
            ax=axes[i,j]
            for slam in slams:
                g=tactics[(tactics.tour==t)&(tactics.Tournament==slam)].set_index('period').reindex(periods)
                y=g[metric].where(g[metric+'_n']>=10)
                ax.plot(range(6),y,'o-',label=slam,markersize=4)
                ax.fill_between(range(6),g[metric+'_low'].where(y.notna()),g[metric+'_high'].where(y.notna()),alpha=.10)
            ax.set(title=t+' · '+title);ax.set_xticks(range(6),periods,rotation=35)
            ax.set_ylabel('Per 100 points' if 'per100' in metric else ('Successful shots' if metric=='mean_rally' else '%'))
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4)
    fig.tight_layout(rect=(0,0,1,.94));savefig(filename)
tactic_plot([('mean_rally','Mean rally length'),('net_per100','Net involvements'),('rally_approach_per100','Approaches excluding S&V')],'07_rallies_and_net')
tactic_plot([('plus_opportunity_pct','Serve +1 opportunities / points'),('plus_fh_pct','Forehand share of known +1 shots'),
             ('plus_success_pct','Immediate +1 successes / opportunities')],'08_serve_plus_one')
tactic_plot([('snv_per100','Serve-and-volley points'),('plus_winner_pct','Clean +1 winners / opportunities')],'09_snv_and_plus_one_winners')

# %% [markdown]
# ### Sensitivity to composition and weighting
#
# Compare early (1995–2004) and late (2015–2025) periods within each Slam/tour. These endpoint windows are deliberately wider than chart periods. Report raw equal-match means, **late-round-only means (QF/SF/F)** to partly address the archival preference for famous late matches, and point/opportunity-weighted rates. Late-round restriction does not control player composition. Sparse endpoint cells remain unreliable. “Net approach frequency” and “forehand preference on serve +1” describe separate things, so neither is a substitute for the other.
#
# %%
endpoint=[]
for keys,g in q.groupby(['tour','Tournament']):
    for selection in ['All charted rounds','QF/SF/F only']:
        z=g if selection=='All charted rounds' else g[g.late_round]
        for metric in ['mean_rally','net_per100','approach_per100','rally_approach_per100','snv_per100','plus_fh_pct','plus_success_pct']:
            early=z[z.year.between(1995,2004)][metric].dropna().to_numpy()
            late=z[z.year.between(2015,2025)][metric].dropna().to_numpy()
            if len(early) and len(late):
                lo=hi=np.nan
                if min(len(early),len(late))>=10:
                    diff=late[rng.integers(0,len(late),(1000,len(late)))].mean(1)-early[rng.integers(0,len(early),(1000,len(early)))].mean(1)
                    lo,hi=np.quantile(diff,[.025,.975])
                endpoint.append({'tour':keys[0],'Tournament':keys[1],'sample':selection,'metric':metric,
                    'early_n':len(early),'late_n':len(late),'early_mean':early.mean(),'late_mean':late.mean(),
                    'change':late.mean()-early.mean(),'low':lo,'high':hi})
endpoint=pd.DataFrame(endpoint);table(endpoint,'q3_endpoint_changes')
weighted=[]
for keys,g in q.groupby(['tour','Tournament','period'],observed=True):
    row=dict(zip(['tour','Tournament','period'],keys))
    row['mean_rally']=np.average(g.mean_rally,weights=g.parsed_points)
    for metric,(num,den) in metrics.items():
        if den:
            z=g.dropna(subset=[num,den]);row[metric]=100*z[num].sum()/z[den].sum() if z[den].sum() else np.nan
    weighted.append(row)
table(pd.DataFrame(weighted),'q3_point_weighted_sensitivity')

# %% [markdown]
# ### 5C. Net frequency, success, and how points end
#
# The original question-3 analysis **already uses both `charting-m-stats-NetPoints.csv` and `charting-w-stats-NetPoints.csv`**. It sums `NetPoints` and `ApproachRallies` across the two players, with match points as denominator. This extension exposes the player-level records and adds effectiveness: `pts_won/net_pts`, net winners, induced forced errors, net unforced errors, and times passed.
#
# Frequency and effectiveness answer different questions. Players can approach less often but win a greater share of those selected approaches. Success rates are conditional on deciding to approach; they do not estimate what would happen if a player approached more often. Outcomes listed here are not an exhaustive partition of all net points, especially serve-and-volley points ending before a volley. We retain the same high-coverage Slam match sample as the main tactics analysis, and use equal-match averages and match bootstrap intervals.
#
# %%
net_detail=[]
for sex,t in [('m','ATP'),('w','WTA')]:
    n=read(MCP/f'charting-{sex}-stats-NetPoints.csv').drop_duplicates()
    bad=set(n.loc[n.duplicated(['match_id','player','row'],False),'match_id'])
    n=n[~n.match_id.isin(bad)&n.row.isin(['NetPoints','ApproachRallies'])].copy()
    n['tour']=t;net_detail.append(n)
net_detail=pd.concat(net_detail).merge(q[['match_id','Tournament','year','period','upstream_points']],on='match_id',validate='many_to_one')
net_detail['net_frequency_per100']=100*net_detail.net_pts/net_detail.upstream_points
net_detail['net_win_pct']=100*net_detail.pts_won/net_detail.net_pts.replace(0,np.nan)
table(net_detail[['match_id','tour','Tournament','year','player','row','net_pts','pts_won','net_frequency_per100','net_win_pct',
                  'net_winner','induced_forced','net_unforced','passed_at_net']],'q3_mcp_player_net_detail')
net_match=net_detail.groupby(['match_id','tour','Tournament','period','row'],observed=True).agg(
    net_pts=('net_pts','sum'),pts_won=('pts_won','sum'),net_winner=('net_winner','sum'),
    induced_forced=('induced_forced','sum'),net_unforced=('net_unforced','sum'),passed_at_net=('passed_at_net','sum')).reset_index()
for field in ['pts_won','net_winner','induced_forced','net_unforced','passed_at_net']:
    net_match[field+'_pct']=100*net_match[field]/net_match.net_pts.replace(0,np.nan)
net_summary=[]
for keys,g in net_match.groupby(['tour','Tournament','period','row'],observed=True):
    row=dict(zip(['tour','Tournament','period','row'],keys));row['matches']=len(g);row['net_involvements']=g.net_pts.sum()
    for field in ['pts_won','net_winner','induced_forced','net_unforced','passed_at_net']:
        v=g[field+'_pct'].dropna().to_numpy();row[field+'_pct']=v.mean() if len(v) else np.nan
        if len(v)>=10:
            draws=rng.integers(0,len(v),(1000,len(v)));lo,hi=np.quantile(v[draws].mean(1),[.025,.975])
        else:lo,hi=np.nan,np.nan
        row[field+'_low']=lo;row[field+'_high']=hi
    net_summary.append(row)
net_summary=pd.DataFrame(net_summary);table(net_summary,'q3_mcp_net_outcome_summary')
fig,axes=plt.subplots(2,2,figsize=(12,8))
for i,t in enumerate(['ATP','WTA']):
    for j,(kind,title) in enumerate([('NetPoints','All net involvements'),('ApproachRallies','Non-serve approaches')]):
        ax=axes[i,j]
        for slam in slams:
            g=net_summary[(net_summary.tour==t)&(net_summary.Tournament==slam)&net_summary.row.eq(kind)].set_index('period').reindex(periods)
            y=g.pts_won_pct.where(g.matches>=10)
            ax.plot(range(6),y,'o-',label=slam)
            ax.fill_between(range(6),g.pts_won_low.where(y.notna()),g.pts_won_high.where(y.notna()),alpha=.10)
        ax.set(title=t+' · '+title,ylabel='Net points won (%)');ax.set_xticks(range(6),periods,rotation=30)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4)
fig.tight_layout(rect=(0,0,1,.94));savefig('12_mcp_net_effectiveness')

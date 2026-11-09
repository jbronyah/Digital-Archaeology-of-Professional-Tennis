# %% [markdown]
# # Tactical evolution or archival selection? Four investigations
#
# **Grand Slam singles, ATP and WTA, 1995–2025.** This follows the original `tennis_trends.ipynb` with four questions: (1) how net entry changed; (2) whether serving advantage shifted toward the next shot; (3) whether Slam styles converged; and (4) how archival composition shapes those answers.
#
# This section uses the audited match sample and rereads the filtered MCP points. Run the sample stage first. All code is visible here. It saves new figures and CSVs under `build/analysis/` without replacing the earlier results. Bootstrap intervals resample matches unless explicitly stated otherwise; they measure variation within the observed archive, not population representativeness. These exploratory tests are not causal or confirmatory.
#
# Sources: [Match Charting Project](https://github.com/JeffSackmann/tennis_MatchChartingProject), [Sackmann archive](https://github.com/Aneeshers/tennis-sackmann-archive), local MCP dictionary and MatchChart Instructions, [charting guide](https://www.tennisabstract.com/blog/2015/09/23/the-match-charting-project-quick-start-guide/), and [tactics glossary](https://www.tennisabstract.com/blog/2019/08/17/match-charting-project-tactics-stats-glossary/). Data: Jeff Sackmann / Tennis Abstract and volunteer contributors, **CC BY-NC-SA 4.0**. Retain attribution and the license when redistributing derived data.

# %%
from pathlib import Path
import re, json, hashlib, platform
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from IPython.display import display, Markdown

ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'data'/'snapshot.json').is_file())
BASE=ROOT/'build'/'sample'; OUT=ROOT/'build'/'analysis'; MCP=ROOT/'data'
for folder in ['figures','tables']:(OUT/folder).mkdir(parents=True,exist_ok=True)
SEED=20261108; rng=np.random.default_rng(SEED); B=1000
SLAMS=['Australian Open','Roland Garros','Wimbledon','US Open']
PERIODS=['1995–99','2000–04','2005–09','2010–14','2015–19','2020–25']
INPUTS=set()
def read(path,**kw):
    path = Path(path)
    path = path if path.exists() else path.with_suffix('.csv.gz')
    INPUTS.add(path)
    return pd.read_csv(path, low_memory=False, **kw)
def table(d,name,show=True):
    d.to_csv(OUT/'tables'/f'{name}.csv',index=False)
    if show:display(d)
def note(s):display(Markdown(s))
def save(name):
    plt.savefig(OUT/'figures'/f'{name}.png',dpi=180,bbox_inches='tight');plt.show()
def period(y):return pd.cut(y,[1994,1999,2004,2009,2014,2019,2025],labels=PERIODS)
def boot_mean(v):
    v=np.asarray(v,dtype=float);v=v[np.isfinite(v)]
    if len(v)<10:return (v.mean() if len(v) else np.nan,np.nan,np.nan,len(v))
    draws=rng.integers(0,len(v),(B,len(v)));lo,hi=np.quantile(v[draws].mean(1),[.025,.975])
    return v.mean(),lo,hi,len(v)
def boot_change(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float);a=a[np.isfinite(a)];b=b[np.isfinite(b)]
    if not len(a) or not len(b):return np.nan,np.nan,np.nan
    delta=b.mean()-a.mean()
    if min(len(a),len(b))<10:return delta,np.nan,np.nan
    diff=b[rng.integers(0,len(b),(B,len(b)))].mean(1)-a[rng.integers(0,len(a),(B,len(a)))].mean(1)
    return delta,*np.quantile(diff,[.025,.975])
plt.rcParams.update({'figure.dpi':110,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                    'axes.prop_cycle':plt.cycler(color=['#176B87','#C75B39','#6D597A','#4B8753'])})
pd.set_option('display.max_columns',16)
q=read(BASE/'tables/q3_match_metrics.csv')
meta=pd.concat([read(MCP/f'charting-{s}-matches.csv').assign(tour=t) for s,t in [('m','ATP'),('w','WTA')]],ignore_index=True)
meta=meta.drop_duplicates('match_id')
q=q.merge(meta[['match_id','Player 1','Player 2','Charted by']],on='match_id',validate='one_to_one')
q['late_round']=q.Round.isin(['QF','SF','F']); q['era']=np.where(q.year<=2004,'Early',np.where(q.year>=2015,'Late','Middle'))
assert not q.match_id.duplicated().any()
note(f'Using **{len(q):,} audited matches**: '+', '.join(f'{t} {len(g):,}' for t,g in q.groupby('tour'))+'.')

# %% [markdown]
# ## 1. Decode the sequence, and distinguish observation from inference
#
# Rally length includes the serve and excludes the terminal error. Penalties, unknown whole points, incorrect challenges and malformed sequences are excluded, as in the first notebook. First-serve faults are replaced by the second-serve rally when present.
#
# A **recorded non-serve approach** is a shot with a `+` position marker followed by an opponent shot. A terminal approach winner/error is excluded because it does not establish a subsequent net visit. Count only the first such entry per player per point, and exclude the server's entries on serve-and-volley points. This marker-based measure is narrower than upstream `ApproachRallies`, which can infer net visits from other shots. Report both; absence of `+` is not proof of no approach.
#
# Approach shot number 2 is a return approach; number 3 is the server's plus-one approach; number 4+ is a later entry. The immediately preceding **opponent shot** is classified as a groundstroke, dropshot, lob, or net shot. These codes do not reveal whether a player intentionally constructed an opening, was forced forward, or faced an easy ball.
#
# Serve outcomes form a mutually exclusive partition: double fault; unreturned serve (including return errors); clean plus-one winner; plus-one followed by a forced opponent error; another server win; another returner win. A plus-one can be a volley. We separately measure forehand share among baseline shot types (`f,r,u,l,b,s,y,m`), excluding an explicit net-position `-` modifier, to avoid conflating fewer volleys with forehand preference. Shot type is not a precise court-position measurement. The volley/overhead category describes shot types, including rare overheads hit from the baseline. Other server wins include unforced errors after plus-one as well as later rallies; they are not all long-rally wins.

# %%
SHOTS='fbrsvzopuylmhijktq'; token_re=re.compile('(['+SHOTS+'])([^'+SHOTS+']*)')
allowed=re.compile(r'^[0-9cfbrsvzopuylmhijktqnwdxeg!+\-=;^*#@]+$')
def decode(first,second,server,winner):
    second_used=pd.notna(second) and bool(str(second).strip())
    text=str(second).strip() if second_used else str(first).strip()
    if not allowed.fullmatch(text) or not re.match(r'^c*[0456]',text) or server not in [1,2] or winner not in [1,2]:return None
    tokens=token_re.findall(text);n=len(tokens);fault=not n and bool(re.search(r'[nwdxeg!]$',text))
    if fault:
        if not second_used:return None
        length=0
    elif text.endswith(('*','#','@')):length=1+n-int(n>0 and text.endswith(('#','@')))
    else:return None
    sw=server==winner;plus=n>=2
    clean=plus and n==2 and text.endswith('*') and sw
    forced=plus and n==3 and text.endswith('#') and sw
    snv=bool(re.match(r'^c*[0456]\+',text)) and not fault and not (n==0 and text.endswith('*'))
    outcome=('Double fault' if fault else 'Unreturned serve' if length==1 and sw else
             'Clean +1 winner' if clean else '+1 induced error' if forced else 'Other server win' if sw else 'Other returner win')
    direction={'4':'Wide','5':'Body','6':'T','0':'Unknown'}[re.match(r'^c*([0456])',text).group(1)]
    plus_type=tokens[1][0] if plus else ''
    depth_codes=set(re.findall('[789]',tokens[0][1])) if plus else set()
    depth={'7':'Shallow','8':'Deep','9':'Very deep'}[next(iter(depth_codes))] if len(depth_codes)==1 else 'Unknown'
    base=plus_type in 'frulbsym' and '-' not in tokens[1][1] if plus else False
    entries=[];seen=set()
    for i,(shot,suffix) in enumerate(tokens[:-1]):
        player=server if i%2 else 3-server
        if '+' not in suffix or player in seen or (snv and player==server):continue
        seen.add(player)
        before=tokens[i-1][0] if i else 'serve'
        before_class=('Serve' if i==0 else 'Drop shot' if before in 'uy' else 'Lob' if before in 'lm'
                      else 'Groundstroke' if before in 'fbrs' else 'Net shot' if before in 'vzophijk' else 'Other/unknown')
        entries.append({'approacher':player,'role':'Server' if player==server else 'Returner','shot_number':i+2,
                        'preceding_type':before_class,'approach_type':shot,'approach_won':player==winner})
    row={'rally':length,'Svr':server,'server_won':sw,'first_serve':not second_used,'serve_direction':direction,
         'return_depth':depth,'plus':plus,'plus_baseline':base,'plus_fh_baseline':base and plus_type in 'frul',
         'plus_net':plus_type in 'vzophijk' if plus else False,'plus_success':clean or forced,
         'outcome':outcome,'snv':snv,'explicit_approaches':len(entries)}
    return row,entries
assert decode('4*',np.nan,1,1)[0]['outcome']=='Unreturned serve'
assert decode('4n','6w',1,2)[0]['rally']==0
assert decode('6b27f3n@',np.nan,1,2)[0]['plus_baseline']
assert decode('6b27f+3b#',np.nan,1,1)[1][0]['shot_number']==3
assert decode('6b27f+3*',np.nan,1,1)[1]==[]
assert decode('6+b27v+3b#',np.nan,1,1)[1]==[]
assert decode('6s+27f3b*',np.nan,1,2)[1][0]['role']=='Returner'
assert not decode('6s27f-3*',np.nan,1,1)[0]['plus_baseline']
assert decode('6s187f3*',np.nan,1,1)[0]['return_depth']=='Unknown'

frames=[];entries=[];parse_audit=[];ids=set(q.match_id);conflicting_matches=set()
for sex in ['m','w']:
    for path in sorted(MCP.glob(f'charting-{sex}-points-*.csv.gz')):
        d=read(path,usecols=['match_id','Pt','1st','2nd','Svr','PtWinner']);d=d[d.match_id.isin(ids)].drop_duplicates()
        repeated=d[d.duplicated(['match_id','Pt'],False)].copy()
        repeated['decoded_version']=[repr(decode(a,b,c,e)) for a,b,c,e in repeated[['1st','2nd','Svr','PtWinner']].itertuples(index=False,name=None)]
        disagreement=repeated.groupby(['match_id','Pt']).decoded_version.nunique()
        bad=set(disagreement[disagreement>1].index.get_level_values('match_id'))
        conflicting_matches.update(bad)
        d=d[~d.match_id.isin(bad)].drop_duplicates(['match_id','Pt'])
        records=[]
        for mid,pt,first,second,svr,winner in d[['match_id','Pt','1st','2nd','Svr','PtWinner']].itertuples(index=False,name=None):
            parsed=decode(first,second,svr,winner)
            if parsed is None:continue
            row,approaches=parsed;row.update(match_id=mid,Pt=pt);records.append(row)
            for entry in approaches:entry.update(match_id=mid,Pt=pt);entries.append(entry)
        frames.append(pd.DataFrame(records));parse_audit.append({'file':path.stem,'input_points':len(d),'decoded_points':len(records),
                                                               'repeated_variant_rows':len(repeated),'conflicting_matches_excluded':len(bad)})
points=pd.concat(frames,ignore_index=True)
q=q[~q.match_id.isin(conflicting_matches)].copy()
note(f'Quarantined {len(conflicting_matches)} matches with conflicting decoded point versions. '
     'Duplicate versions differing only in unused fault notation are collapsed; the decoder output must agree.')
assert not points.duplicated(['match_id','Pt']).any()
match_keys=['match_id','tour','Tournament','year','period','era','late_round','Player 1','Player 2']
points=points.merge(q[match_keys],on='match_id',validate='many_to_one')
points['player']=np.where(points.Svr.eq(1),points['Player 1'],points['Player 2'])
approaches=pd.DataFrame(entries).merge(q[match_keys],on='match_id',validate='many_to_one')
approaches['timing']=pd.cut(approaches.shot_number,[1,2,3,np.inf],labels=['Return (shot 2)','Serve +1 (shot 3)','Later (shot 4+)'])
table(pd.DataFrame(parse_audit),'01_point_parser_audit')
recount=points.groupby('match_id').rally.agg(['size','mean']).join(q.set_index('match_id')[['parsed_points','mean_rally']])
assert (recount['size']==recount.parsed_points).all() and np.allclose(recount['mean'],recount.mean_rally)
table(approaches.drop(columns=['Player 1','Player 2']),'02_recorded_approach_events',show=False)

# %% [markdown]
# ## 2. Net play: fewer automatic entries, more later approaches?
#
# Primary frequency measures use upstream net aggregates already validated in the first notebook. Marker-based timing and context describe only explicitly recorded approaches. Equal-match averages prevent long matches dominating; conditional success/timing requires at least one recorded approach in the match. Confidence intervals require at least ten matches. Endpoint contrasts use 1995–2004 and 2015–2025, with the full intermediate trajectory retained.

# %%
match_extra=points.groupby('match_id').agg(n=('rally','size'),plus_n=('plus','sum'),baseline_n=('plus_baseline','sum'),
    baseline_fh=('plus_fh_baseline','sum'),net_plus_n=('plus_net','sum'),explicit=('explicit_approaches','sum')).reset_index()
match_extra['baseline_fh_pct']=100*match_extra.baseline_fh/match_extra.baseline_n.replace(0,np.nan)
match_extra['plus_net_pct']=100*match_extra.net_plus_n/match_extra.plus_n.replace(0,np.nan)
match_extra['explicit_per100']=100*match_extra.explicit/match_extra.n
am=approaches.groupby('match_id').agg(approaches=('Pt','size'),approach_success=('approach_won','mean'),
    mean_approach_shot=('shot_number','mean'),later_share=('shot_number',lambda x:(x>=4).mean()),
    after_drop_share=('preceding_type',lambda x:x.eq('Drop shot').mean())).reset_index()
for c in ['approach_success','later_share','after_drop_share']:am[c]*=100
matches=q.merge(match_extra,on='match_id',validate='one_to_one').merge(am,on='match_id',how='left',validate='one_to_one')
timing_counts=pd.crosstab(approaches.match_id,approaches.timing).reindex(q.match_id,fill_value=0)
for label,metric in [('Return (shot 2)','return_approach_rate'),('Serve +1 (shot 3)','plus1_approach_rate'),('Later (shot 4+)','later_approach_rate')]:
    matches[metric]=100*matches.match_id.map(timing_counts[label])/matches.n
metric_labels={'snv_per100':'Serve-and-volley /100 points','rally_approach_per100':'Non-serve approaches /100 points',
               'explicit_per100':'Explicit non-serve approaches /100 points','later_share':'Later entry share (%)',
               'approach_success':'Explicit approach success (%)','mean_approach_shot':'Mean approach shot number',
               'after_drop_share':'Entries after drop shots (%)','baseline_fh_pct':'Forehand share among baseline +1 (%)',
               'plus_net_pct':'Volley/overhead share of +1 (%)','return_approach_rate':'Return approaches /100 points',
               'plus1_approach_rate':'Plus-one approaches /100 points','later_approach_rate':'Later approaches /100 points'}
summary=[]
for keys,g in matches.groupby(['tour','Tournament','period'],observed=True):
    for metric in metric_labels:
        mean,lo,hi,n=boot_mean(g[metric]);summary.append(dict(zip(['tour','Tournament','period'],keys),metric=metric,mean=mean,low=lo,high=hi,n=n))
summary=pd.DataFrame(summary);table(summary,'03_tactical_period_estimates',show=False)
def trend_panels(metrics,filename):
    fig,axes=plt.subplots(2,len(metrics),figsize=(5*len(metrics),8),squeeze=False)
    for i,t in enumerate(['ATP','WTA']):
        for j,metric in enumerate(metrics):
            ax=axes[i,j]
            for slam in SLAMS:
                z=summary[(summary.tour==t)&(summary.Tournament==slam)&(summary.metric==metric)].set_index('period').reindex(PERIODS)
                y=z['mean'].where(z.n>=10);ax.plot(range(6),y,'o-',label=slam,markersize=4)
                ax.fill_between(range(6),z.low.where(y.notna()),z.high.where(y.notna()),alpha=.1)
            ax.set_title(t+' · '+metric_labels[metric]);ax.set_xticks(range(6),PERIODS,rotation=40)
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4)
    fig.tight_layout(rect=(0,0,1,.94));save(filename)
trend_panels(['snv_per100','rally_approach_per100','later_share'],'01_net_entry_transformation')
trend_panels(['mean_approach_shot','after_drop_share','approach_success'],'02_approach_context')
endpoint=[]
for keys,g in matches.groupby(['tour','Tournament']):
    for metric in metric_labels:
        a=g[g.era=='Early'][metric].dropna();b=g[g.era=='Late'][metric].dropna();delta,lo,hi=boot_change(a,b)
        endpoint.append(dict(zip(['tour','Tournament'],keys),metric=metric,early_n=len(a),late_n=len(b),
                             early=a.mean(),late=b.mean(),change=delta,low=lo,high=hi))
endpoint=pd.DataFrame(endpoint);table(endpoint,'04_tactical_endpoint_changes')
# Explicit approach-marker coverage is compared with upstream counts in rate units, not assumed identical.
coverage_markers=matches.groupby(['tour','period'],observed=True).agg(matches=('match_id','size'),
    explicit_rate=('explicit_per100','mean'),upstream_approach_rate=('rally_approach_per100','mean')).reset_index()
coverage_markers['ratio_of_mean_rates']=coverage_markers.explicit_rate/coverage_markers.upstream_approach_rate
table(coverage_markers,'05_approach_marker_comparability')
context=approaches.groupby(['tour','Tournament','era','role','timing','preceding_type'],observed=True).agg(
    entries=('Pt','size'),wins=('approach_won','sum')).reset_index()
context['success_pct']=100*context.wins/context.entries;table(context,'06_approach_context_counts',show=False)

# %% [markdown]
# ## 3. From serve to next shot: mechanism, direction, and return depth
#
# Outcome shares below use **all valid service points**, including double faults. They describe how points ended, not the causal contribution of a serve. Forehand preference among groundstrokes and the share of plus-one volleys address different changes.
#
# Direction and return depth are optional codes. Audit their coverage before comparing tactical outcomes; conflicting depth codes on one return are treated as unknown. For matched comparisons use first-serve points and compare categories **within the same server–match**, requiring at least five eligible points in each category. Wide-versus-T compares forehand share among baseline plus-one attempts. Shallow-versus-very-deep compares immediate plus-one success among opportunities, only in matches with ≥80% depth coverage on first-serve plus-one opportunities. Equal server–match weighting avoids high-volume players dominating. Intervals resample whole matches, retaining both servers together. Neither comparison controls point score or deuce/ad court; these are descriptive conditional associations.

# %%
OUTCOMES=['Double fault','Unreturned serve','Clean +1 winner','+1 induced error','Other server win','Other returner win']
oc=pd.crosstab(points.match_id,points.outcome).reindex(columns=OUTCOMES,fill_value=0)
oc=oc.div(oc.sum(axis=1),axis=0)*100
assert np.allclose(oc.sum(axis=1),100)
oc=oc.reset_index().merge(q[match_keys[:7]],on='match_id',validate='one_to_one')
outcome_summary=oc.groupby(['tour','Tournament','period'],observed=True)[OUTCOMES].mean().reset_index()
outcome_summary=outcome_summary.merge(oc.groupby(['tour','Tournament','period'],observed=True).size().rename('matches').reset_index(),on=['tour','Tournament','period'])
table(outcome_summary,'07_service_outcome_partition',show=False)
oc['Immediate +1 success']=oc['Clean +1 winner']+oc['+1 induced error']
serve_endpoint=[]
for keys,g in oc.groupby(['tour','Tournament']):
    for metric in [*OUTCOMES,'Immediate +1 success']:
        a=g[g.era=='Early'][metric];b=g[g.era=='Late'][metric];delta,lo,hi=boot_change(a,b)
        serve_endpoint.append(dict(zip(['tour','Tournament'],keys),metric=metric,early_n=len(a),late_n=len(b),
                                   early=a.mean(),late=b.mean(),change=delta,low=lo,high=hi))
serve_endpoint=pd.DataFrame(serve_endpoint);table(serve_endpoint,'20_service_outcome_endpoint_changes')
fig,axes=plt.subplots(2,4,figsize=(17,8),sharey=True)
colors=['#C8C8C8','#176B87','#C75B39','#E5AA70','#6D597A','#92B69A']
for i,t in enumerate(['ATP','WTA']):
    for j,slam in enumerate(SLAMS):
        g=outcome_summary[(outcome_summary.tour==t)&(outcome_summary.Tournament==slam)].set_index('period').reindex(PERIODS)
        n=matches[(matches.tour==t)&(matches.Tournament==slam)].groupby('period',observed=True).size().reindex(PERIODS)
        bottom=np.zeros(6);ax=axes[i,j]
        for col,color in zip(OUTCOMES,colors):
            y=g[col].where(n>=10);ax.bar(range(6),y,bottom=bottom,label=col,color=color);bottom+=y.fillna(0)
        ax.set(title=t+' · '+slam,ylim=(0,100));ax.set_xticks(range(6),PERIODS,rotation=60)
        if j==0:ax.set_ylabel('Mean within-match point share (%)')
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3)
fig.tight_layout(rect=(0,0,1,.92));save('03_service_outcome_partition')
trend_panels(['baseline_fh_pct','plus_net_pct'],'04_plus_one_shot_selection')
coding=points.groupby(['tour','period'],observed=True).agg(points=('Pt','size'),
    known_direction=('serve_direction',lambda x:x.ne('Unknown').sum()),opportunities=('plus','sum'),
    known_depth=('return_depth',lambda x:x.ne('Unknown').sum())).reset_index()
coding['direction_pct']=100*coding.known_direction/coding.points
coding['depth_pct']=100*coding.known_depth/coding.opportunities
table(coding,'08_optional_coding_coverage')
first=points[points.first_serve].copy()
dc=first[first.plus].groupby('match_id').return_depth.apply(lambda s:s.ne('Unknown').mean())
direction=first[first.plus_baseline&first.serve_direction.isin(['Wide','T'])].groupby(
    ['match_id','player','serve_direction']).agg(n=('Pt','size'),value=('plus_fh_baseline','mean')).reset_index()
depth=first[first.plus&first.match_id.isin(dc[dc>=.8].index)&first.return_depth.isin(['Shallow','Very deep'])].groupby(
    ['match_id','player','return_depth']).agg(n=('Pt','size'),value=('plus_success','mean')).reset_index()
def contrast_rows(data,column,a,b,label):
    data=data[data.n>=5]
    wide=data.pivot(index=['match_id','player'],columns=column,values='value')
    if a not in wide or b not in wide:return pd.DataFrame(columns=['match_id','player','contrast','difference'])
    wide=wide[[a,b]].dropna();wide['difference']=100*(wide[a]-wide[b]);wide['contrast']=label
    return wide.reset_index()[['match_id','player','contrast','difference']]
paired=pd.concat([contrast_rows(direction,'serve_direction','Wide','T','Wide minus T: baseline FH share'),
                  contrast_rows(depth,'return_depth','Shallow','Very deep','Shallow minus very deep: +1 success')],ignore_index=True)
paired=paired.merge(q[['match_id','tour','Tournament','period']],on='match_id',validate='many_to_one')
conditional=[]
for keys,g in paired.groupby(['tour','Tournament','contrast'],observed=True):
    clusters=g.groupby('match_id').difference.agg(['sum','count']).to_numpy()
    if len(clusters)>=10:
        draw=rng.integers(0,len(clusters),(B,len(clusters)))
        boot=clusters[draw,0].sum(1)/clusters[draw,1].sum(1);lo,hi=np.quantile(boot,[.025,.975])
    else:lo=hi=np.nan
    conditional.append(dict(zip(['tour','Tournament','contrast'],keys),server_matches=len(g),matches=len(clusters),
                            difference_pp=g.difference.mean(),low=lo,high=hi))
conditional=pd.DataFrame(conditional);table(conditional,'09_within_match_direction_depth_contrasts')
table(paired,'10_direction_depth_matched_units',show=False)
fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,(label,g) in zip(axes,conditional.groupby('contrast')):
    g=g[g.matches>=10].copy();names=g.tour+' · '+g.Tournament
    ax.errorbar(g.difference_pp,range(len(g)),xerr=[g.difference_pp-g.low,g.high-g.difference_pp],fmt='o',capsize=3)
    ax.set_yticks(range(len(g)),names);ax.axvline(0,color='grey',lw=1);ax.set(title=label,xlabel='Percentage-point difference')
fig.tight_layout();save('05_direction_and_depth')

# %% [markdown]
# ## 4. Do the Slams converge? Distributional and player-matched checks
#
# Use Jensen–Shannon **distance** (base 2, bounded 0–1) between the mean within-match rally-length distributions: 0, 1–3, 4–6, 7–9, 10+. A lower value means more similar distributions. This is different from comparing mean rally lengths. Also report absolute differences in net frequency and serve-and-volley, in percentage points. A shrinking rally distance does not imply convergence on every tactical dimension.
#
# Early/late contrasts bootstrap matches separately within each Slam and window. The same-player check compares Wimbledon with another Slam in the **same season**, using the player's own service-point rally length, plus-one baseline forehand share, and recorded serve-and-volley rate. Average matches within each player–year–Slam first. Players must appear at both Slams in that season; this is a selected subset, not a population adjustment. Intervals resample players to keep their seasons together. Opponents and rounds can still differ.

# %%
points['rally_bin']=pd.cut(points.rally,[-1,0,3,6,9,np.inf],labels=['0','1–3','4–6','7–9','10+'])
dist=pd.crosstab(points.match_id,points.rally_bin).reindex(columns=['0','1–3','4–6','7–9','10+'],fill_value=0)
dist=dist.div(dist.sum(axis=1),axis=0).reset_index().merge(q[['match_id','tour','Tournament','period','era']],on='match_id')
bins=['0','1–3','4–6','7–9','10+']
distance_rows=[]
for (t,p),g in dist.groupby(['tour','period'],observed=True):
    for other in [x for x in SLAMS if x!='Wimbledon']:
        a=g[g.Tournament=='Wimbledon'];b=g[g.Tournament==other]
        if min(len(a),len(b))<10:continue
        distance_rows.append({'tour':t,'period':p,'comparison':'Wimbledon vs '+other,'wimbledon_n':len(a),'other_n':len(b),
                              'rally_distance':jensenshannon(a[bins].mean(),b[bins].mean(),base=2)})
distances=pd.DataFrame(distance_rows);table(distances,'11_rally_distance_by_period')
from itertools import combinations
all_pairs=[]
for t in ['ATP','WTA']:
    for a_name,b_name in combinations(SLAMS,2):
        estimates=[];draws_by_era=[];ns=[]
        for era in ['Early','Late']:
            a=dist[(dist.tour==t)&(dist.Tournament==a_name)&(dist.era==era)][bins].to_numpy()
            b=dist[(dist.tour==t)&(dist.Tournament==b_name)&(dist.era==era)][bins].to_numpy();ns.extend([len(a),len(b)])
            estimates.append(jensenshannon(a.mean(0),b.mean(0),base=2))
            aa=a[rng.integers(0,len(a),(B,len(a)))].mean(1);bb=b[rng.integers(0,len(b),(B,len(b)))].mean(1)
            draws_by_era.append(jensenshannon(aa,bb,base=2,axis=1))
        lo,hi=np.quantile(draws_by_era[1]-draws_by_era[0],[.025,.975]) if min(ns)>=10 else (np.nan,np.nan)
        all_pairs.append({'tour':t,'slam_a':a_name,'slam_b':b_name,'early_distance':estimates[0],'late_distance':estimates[1],
                          'change':estimates[1]-estimates[0],'low':lo,'high':hi,'early_a_n':ns[0],'early_b_n':ns[1],
                          'late_a_n':ns[2],'late_b_n':ns[3]})
all_pairs=pd.DataFrame(all_pairs);table(all_pairs,'21_all_slam_pair_convergence')
# Reuse the same bootstrap estimates in the Wimbledon subset and the all-pairs table.
distance_change=[]
for r in all_pairs[all_pairs.slam_a.eq('Wimbledon')|all_pairs.slam_b.eq('Wimbledon')].itertuples():
    w_a=r.slam_a=='Wimbledon'
    distance_change.append({'tour':r.tour,'other_slam':r.slam_b if w_a else r.slam_a,
        'early_wimbledon_n':r.early_a_n if w_a else r.early_b_n,'early_other_n':r.early_b_n if w_a else r.early_a_n,
        'late_wimbledon_n':r.late_a_n if w_a else r.late_b_n,'late_other_n':r.late_b_n if w_a else r.late_a_n,
        'early_distance':r.early_distance,'late_distance':r.late_distance,'change':r.change,'low':r.low,'high':r.high})
distance_change=pd.DataFrame(distance_change);table(distance_change,'12_rally_convergence_endpoint')
fig,axes=plt.subplots(2,2,figsize=(11,9))
for i,t in enumerate(['ATP','WTA']):
    for j,(col,label) in enumerate([('early_distance','1995–2004'),('late_distance','2015–2025')]):
        mat=np.zeros((4,4))
        for r in all_pairs[all_pairs.tour==t].itertuples():
            a,b=SLAMS.index(r.slam_a),SLAMS.index(r.slam_b);mat[a,b]=mat[b,a]=getattr(r,col)
        ax=axes[i,j];im=ax.imshow(mat,vmin=0,vmax=.25,cmap='Blues')
        for (a,b),v in np.ndenumerate(mat):ax.text(b,a,f'{v:.3f}',ha='center',va='center',color='white' if v>.15 else 'black')
        ax.set_xticks(range(4),['AO','RG','WIM','US']);ax.set_yticks(range(4),['AO','RG','WIM','US']);ax.set_title(t+' · '+label)
fig.suptitle('Rally-distribution distance: all Grand Slam pairs (0 = identical)',y=.99)
fig.tight_layout(rect=(0,0,1,.96));save('10_all_slam_distances')
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,t in zip(axes,['ATP','WTA']):
    for label,g in distances[distances.tour==t].groupby('comparison'):
        z=g.set_index('period').reindex(PERIODS);ax.plot(range(6),z.rally_distance,'o-',label=label.replace('Wimbledon vs ',''))
    ax.set(title=t+' · Wimbledon rally distribution distance',ylabel='Jensen–Shannon distance');ax.set_xticks(range(6),PERIODS,rotation=35);ax.legend()
fig.tight_layout();save('06_rally_convergence')
style_dist=[]
for keys,g in matches.groupby(['tour','period'],observed=True):
    for other in [x for x in SLAMS if x!='Wimbledon']:
        a=g[g.Tournament=='Wimbledon'];b=g[g.Tournament==other]
        for metric in ['net_per100','snv_per100','rally_approach_per100']:
            style_dist.append(dict(zip(['tour','period'],keys),other_slam=other,metric=metric,wimbledon_n=len(a),other_n=len(b),
                                   absolute_gap=abs(a[metric].mean()-b[metric].mean())))
table(pd.DataFrame(style_dist),'13_tactical_distance_by_period',show=False)
ps=points.groupby(['match_id','tour','Tournament','year','period','player'],observed=True).agg(
    service_points=('Pt','size'),service_rally=('rally','mean'),snv=('snv','sum'),baseline=('plus_baseline','sum'),fh=('plus_fh_baseline','sum')).reset_index()
ps['snv_pct']=100*ps.snv/ps.service_points;ps['baseline_fh_pct']=100*ps.fh/ps.baseline.replace(0,np.nan)
py=ps.groupby(['tour','player','year','period','Tournament'],observed=True)[['service_rally','snv_pct','baseline_fh_pct']].mean().reset_index()
matched=[]
for other in [x for x in SLAMS if x!='Wimbledon']:
    a=py[py.Tournament=='Wimbledon'];b=py[py.Tournament==other]
    z=a.merge(b,on=['tour','player','year','period'],suffixes=('_w','_o'),validate='one_to_one')
    for metric in ['service_rally','snv_pct','baseline_fh_pct']:
        r=z[['tour','player','year','period']].copy();r['metric']=metric;r['other_slam']=other
        r['difference']=z[metric+'_w']-z[metric+'_o'];matched.append(r)
matched=pd.concat(matched).dropna(subset=['difference'])
same_player=[]
for keys,g in matched.groupby(['tour','period','other_slam','metric'],observed=True):
    clusters=g.groupby('player').difference.agg(['sum','count']).to_numpy()
    lo=hi=np.nan
    if len(clusters)>=10:
        draw=rng.integers(0,len(clusters),(B,len(clusters)))
        lo,hi=np.quantile(clusters[draw,0].sum(1)/clusters[draw,1].sum(1),[.025,.975])
    same_player.append(dict(zip(['tour','period','other_slam','metric'],keys),player_seasons=len(g),players=len(clusters),
                            wimbledon_minus_other=g.difference.mean(),low=lo,high=hi))
same_player=pd.DataFrame(same_player);table(same_player,'14_same_player_season_comparisons')
table(matched,'15_same_player_season_units',show=False)
fig,axes=plt.subplots(2,2,figsize=(12,8))
for i,t in enumerate(['ATP','WTA']):
    for j,metric in enumerate(['service_rally','snv_pct']):
        ax=axes[i,j]
        for other in [x for x in SLAMS if x!='Wimbledon']:
            z=same_player[(same_player.tour==t)&(same_player.metric==metric)&(same_player.other_slam==other)].set_index('period').reindex(PERIODS)
            y=z.wimbledon_minus_other.where(z.players>=10);ax.plot(range(6),y,'o-',label=other)
            ax.fill_between(range(6),z.low.where(y.notna()),z.high.where(y.notna()),alpha=.1)
        ax.axhline(0,color='grey',lw=1);ax.set(title=t+' · '+metric.replace('_',' '),ylabel='Wimbledon minus comparison Slam')
        ax.set_xticks(range(6),PERIODS,rotation=35)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3)
fig.tight_layout(rect=(0,0,1,.94));save('07_same_player_checks')

# %% [markdown]
# ## 5. The archive as an object of study
#
# Concentration uses **player appearances** (two per match), not distinct people per match. Contributor concentration uses the entire `Charted by` credit string as a label; multiple people can share a credit and the same person can use spelling variants. It is not a verified contributor-identity measure. Missing credits are reported separately. Broad archive coverage ratios come from the first notebook.
#
# Sensitivity estimates compare 1995–2004 with 2015–2025 for each Slam/tour. (a) All eligible matches; (b) QF/SF/F only; (c) equal weight to each round present with ≥3 matches in both eras; (d) exclude matches involving the five most-charted players across the two eras; (e) equal weight to each contributor-credit label with ≥3 matches in both eras; (f) players with ≥2 matches in each era, weighting each bridge player equally. The last design can be tiny and confounds period with aging. None reconstructs a random historical sample. Bootstrap intervals for (c/e) resample within stratum/era; for (f), resample bridge players; otherwise resample matches.

# %%
composition=[]
for keys,g in matches.groupby(['tour','Tournament','period'],observed=True):
    people=pd.concat([g['Player 1'],g['Player 2']]).value_counts();credits=g['Charted by'].fillna('').str.strip()
    counts=credits[credits.ne('')].value_counts();shares=counts/counts.sum()
    composition.append(dict(zip(['tour','Tournament','period'],keys),matches=len(g),
        late_round_pct=100*g.late_round.mean(),unique_players=len(people),top5_player_appearance_pct=100*people.head(5).sum()/people.sum(),
        credit_labels=len(counts),missing_credit_pct=100*credits.eq('').mean(),
        top3_credit_pct=100*shares.head(3).sum(),credit_hhi=(shares**2).sum()))
composition=pd.DataFrame(composition);table(composition,'16_archive_composition')
archive_cov=read(BASE/'tables/q3_charted_coverage.csv')
table(archive_cov.merge(composition,on=['tour','Tournament','period'],suffixes=('_charted','_analysis')),'17_archive_coverage_and_selection',show=False)
fig,axes=plt.subplots(2,2,figsize=(12,8))
for i,t in enumerate(['ATP','WTA']):
    for j,metric in enumerate(['late_round_pct','top5_player_appearance_pct']):
        ax=axes[i,j]
        for slam in SLAMS:
            z=composition[(composition.tour==t)&(composition.Tournament==slam)].set_index('period').reindex(PERIODS)
            ax.plot(range(6),z[metric],'o-',label=slam)
        ax.set(title=t+' · '+metric.replace('_',' '),ylabel='%');ax.set_xticks(range(6),PERIODS,rotation=35)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=4)
fig.tight_layout(rect=(0,0,1,.94));save('08_archive_composition')
robust=[];top_players=[]
test_metrics=['snv_per100','rally_approach_per100','mean_rally','baseline_fh_pct']
for keys,g0 in matches[matches.era.isin(['Early','Late'])].groupby(['tour','Tournament']):
    people=pd.concat([g0['Player 1'],g0['Player 2']]).value_counts();top=set(people.head(5).index)
    for player,count in people.head(5).items():top_players.append(dict(zip(['tour','Tournament'],keys),player=player,appearances=count))
    choices={'All eligible':g0,'Late rounds only':g0[g0.late_round],
             'Exclude top 5':g0[~g0['Player 1'].isin(top)&~g0['Player 2'].isin(top)]}
    for method,g in choices.items():
        for metric in test_metrics:
            a=g[g.era=='Early'][metric].dropna();b=g[g.era=='Late'][metric].dropna();delta,lo,hi=boot_change(a,b)
            robust.append(dict(zip(['tour','Tournament'],keys),method=method,metric=metric,early_n=len(a),late_n=len(b),
                               units=np.nan,change=delta,low=lo,high=hi))
    for metric in test_metrics:
        # Balance contributor-credit labels independently of the round-balancing check below.
        credit_deltas=[];credit_boot=[];credit_a=credit_b=0
        credit_data=g0[g0['Charted by'].fillna('').str.strip().ne('')]
        for credit,g in credit_data.groupby('Charted by'):
            a=g[g.era=='Early'][metric].dropna().to_numpy();b=g[g.era=='Late'][metric].dropna().to_numpy()
            if min(len(a),len(b))<3:continue
            credit_deltas.append(b.mean()-a.mean());credit_a+=len(a);credit_b+=len(b)
            credit_boot.append(b[rng.integers(0,len(b),(B,len(b)))].mean(1)-a[rng.integers(0,len(a),(B,len(a)))].mean(1))
        if credit_deltas:
            lo,hi=np.quantile(np.mean(credit_boot,axis=0),[.025,.975]) if min(credit_a,credit_b)>=10 else (np.nan,np.nan)
            robust.append(dict(zip(['tour','Tournament'],keys),method='Equal common credits',metric=metric,early_n=credit_a,late_n=credit_b,
                               units=len(credit_deltas),change=np.mean(credit_deltas),low=lo,high=hi))
        common=[];bootstrap=[];n_a=n_b=0
        for round_name,g in g0.groupby('Round'):
            a=g[g.era=='Early'][metric].dropna().to_numpy();b=g[g.era=='Late'][metric].dropna().to_numpy()
            if min(len(a),len(b))<3:continue
            common.append(b.mean()-a.mean());n_a+=len(a);n_b+=len(b)
            bootstrap.append(b[rng.integers(0,len(b),(B,len(b)))].mean(1)-a[rng.integers(0,len(a),(B,len(a)))].mean(1))
        if common:
            lo,hi=np.quantile(np.mean(bootstrap,axis=0),[.025,.975]) if min(n_a,n_b)>=10 else (np.nan,np.nan)
            robust.append(dict(zip(['tour','Tournament'],keys),method='Equal common rounds',metric=metric,early_n=n_a,late_n=n_b,
                               units=len(common),change=np.mean(common),low=lo,high=hi))
        roster=pd.concat([g0.assign(player=g0[c]) for c in ['Player 1','Player 2']])
        stats=roster.groupby(['player','era'])[metric].agg(['mean','count']).reset_index()
        stats=stats[stats['count']>=2].pivot(index='player',columns='era',values='mean')
        if 'Early' in stats and 'Late' in stats:
            stats=stats[['Early','Late']].dropna();v=stats.Late-stats.Early
            mean,lo,hi,n=boot_mean(v)
            robust.append(dict(zip(['tour','Tournament'],keys),method='Bridge players',metric=metric,early_n=np.nan,late_n=np.nan,
                               units=n,change=mean,low=lo,high=hi))
robust=pd.DataFrame(robust);table(robust,'18_selection_sensitivity')
table(pd.DataFrame(top_players),'19_excluded_top_players')
fig,axes=plt.subplots(1,2,figsize=(13,5))
for ax,metric in zip(axes,['snv_per100','rally_approach_per100']):
    z=robust[(robust.tour=='ATP')&(robust.Tournament=='Wimbledon')&(robust.metric==metric)].copy()
    ax.scatter(z.change,range(len(z)));valid=z.low.notna()
    ax.errorbar(z.loc[valid,'change'],np.flatnonzero(valid),xerr=[z.loc[valid,'change']-z.loc[valid,'low'],z.loc[valid,'high']-z.loc[valid,'change']],fmt='none',capsize=3)
    labels=[f'Bridge players (n={int(r.units)}; descriptive)' if r.method=='Bridge players' else r.method for r in z.itertuples()]
    ax.set_yticks(range(len(z)),labels);ax.axvline(0,color='grey',lw=1)
    ax.set(title='ATP Wimbledon · '+metric_labels[metric],xlabel='Late minus early (per 100 points)')
fig.tight_layout();save('09_wimbledon_selection_sensitivity')

# %% [markdown]
# ## 6. Findings and limits
#
# All contrasts below are computed from this snapshot. Numerical changes alone do not show intention, causal surface effects, or how the full tennis population changed. The sensitivity tables show which comparisons weaken under alternative sample definitions. Sparse cells and absent coding are substantive archival findings, not values to fill with invented observations.

# %%
lines=['# Follow-up findings: tactical evolution and archival selection','',
       'Windows: 1995–2004 versus 2015–2025 unless otherwise specified. Sources: Jeff Sackmann / Tennis Abstract and MCP contributors (CC BY-NC-SA 4.0).','',
       '## Interpretation','',
       '- **Net play changed form:** men’s Wimbledon has far fewer serve-and-volley attempts and more non-serve approaches. '
       'However, the share of explicitly recorded approaches occurring on shot 4 or later does not clearly increase. '
       'The evidence supports a change in entry type more strongly than a claim that players now wait longer to construct an approach.',
       '- **The plus-one story depends on the denominator:** the large increase in overall forehand plus-one share is largely '
       'associated with replacing net shots. Restricting to baseline shot types leaves a much smaller, uncertain Wimbledon change.',
       '- **Convergence is dimension-specific:** Wimbledon rally distributions become more similar to Australian Open and Roland Garros '
       'in this selected sample; the US Open contrast is inconclusive. Modern same-player comparisons still show shorter Wimbledon service rallies.',
       '- **Selection is substantial:** early charting concentrates on late rounds and a few players. The men’s Wimbledon net-entry '
       'changes survive late-round restriction, round balancing, and excluding the five most-charted players. '
       'Only one player meets the strict Wimbledon bridge-player criterion, so that check cannot establish a general within-player historical effect.',
       '',
       '## 1. Net-entry transformation','']
for t in ['ATP','WTA']:
    for slam in SLAMS:
        z=endpoint[(endpoint.tour==t)&(endpoint.Tournament==slam)].set_index('metric')
        chunks=[]
        for m,label in [('snv_per100','S&V /100 points'),('rally_approach_per100','non-serve approaches /100 points'),('later_share','later share of explicit approaches (%)')]:
            r=z.loc[m];chunks.append(f'{label}: {r.early:.1f} → {r.late:.1f} (change {r.change:+.1f}, CI {r.low:+.1f} to {r.high:+.1f})')
        lines.append(f'- **{t}, {slam}:** '+'; '.join(chunks)+'.')
lines+=['','Timing estimates use explicit approach markers, a narrower and potentially changing subset of net visits. '
        'They cannot establish intent or whether a player was drawn forward by an opponent.',
        '', '## 2. Serving mechanisms','']
for t in ['ATP','WTA']:
    z=endpoint[(endpoint.tour==t)&(endpoint.Tournament=='Wimbledon')].set_index('metric')
    for m in ['baseline_fh_pct','plus_net_pct']:
        r=z.loc[m];lines.append(f'- **{t}, Wimbledon — {metric_labels[m]}:** {r.early:.1f} → {r.late:.1f}; change {r.change:+.1f} pp (CI {r.low:+.1f} to {r.high:+.1f}).')
    for r in serve_endpoint[(serve_endpoint.tour==t)&(serve_endpoint.Tournament=='Wimbledon')&serve_endpoint.metric.isin(['Unreturned serve','Immediate +1 success','Other server win'])].itertuples():
        lines.append(f'- **{t}, Wimbledon — {r.metric} / all service points:** {r.early:.1f}% → {r.late:.1f}%; '
                     f'change {r.change:+.1f} pp (CI {r.low:+.1f} to {r.high:+.1f}).')
for r in conditional[conditional.Tournament.eq('Wimbledon')].itertuples():
    lines.append(f'- **{r.tour}, Wimbledon — {r.contrast}:** {r.difference_pp:+.1f} pp '
                 f'(CI {r.low:+.1f} to {r.high:+.1f}; {r.server_matches} server–matches in {r.matches} matches).')
lines+=['','Direction/depth comparisons are exported with sample sizes and match-cluster intervals. They condition on first-serve '
        'points and observed plus-one opportunities; they do not isolate a causal effect of serving wide or returning deep.',
        '', '## 3. Rally-distribution convergence','']
for r in distance_change.itertuples():
    lines.append(f'- **{r.tour}, Wimbledon versus {r.other_slam}:** distance {r.early_distance:.3f} → {r.late_distance:.3f}; '
                 f'change {r.change:+.3f} (CI {r.low:+.3f} to {r.high:+.3f}). Negative means greater similarity.')
lines+=['','Same-player, same-season comparisons of service-point tactics are in table 14. Those pairs reduce player-composition '
        'differences but remain selected and do not control opponent or round. Missing early intervals indicate fewer than ten distinct players.',
        'All six Slam pairs, including comparisons not involving Wimbledon, are in table 21 and figure 10. '
        'Intervals throughout are pointwise and exploratory; no family-wise significance claim is made.',
        '', '## 4. Selection sensitivity: ATP Wimbledon','']
for r in robust[(robust.tour=='ATP')&(robust.Tournament=='Wimbledon')&robust.metric.isin(['snv_per100','rally_approach_per100'])].itertuples():
    lines.append(f'- **{r.method}, {r.metric}:** late-minus-early {r.change:+.2f} (CI {r.low:+.2f} to {r.high:+.2f}); '
                 f'early matches {r.early_n}, late matches {r.late_n}, strata/bridge players {r.units}.')
lines+=['','NaN intervals mean insufficient observations, not no effect. Bridge-player contrasts confound era with aging and may be tiny. '
        'Bootstrap intervals do not repair volunteer selection, surviving-video bias, or repeated-player dependence except where player clustering is specified.',
        '', '## Reproduce','',
        'Run the notebook from its first cell using the project Python environment. Use the bundled snapshot to rebuild the audited sample. New tables and figures are in this folder.']
report='\n'.join(lines);(OUT/'FINDINGS.md').write_text(report);note(report)

# %%
assert points.rally.ge(0).all()
assert (points.plus_fh_baseline<=points.plus_baseline).all()
assert approaches.shot_number.ge(2).all()
assert not approaches.duplicated(['match_id','Pt','approacher']).any()
assert np.isfinite(distances.rally_distance).all() and distances.rally_distance.between(0,1).all()
provenance={'seed':SEED,'bootstrap_replicates':B,'python':platform.python_version(),
            'snapshot_manifest_sha256':hashlib.sha256((MCP/'snapshot.json').read_bytes()).hexdigest(),
            'inputs':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(INPUTS)]}
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
print(f'Validated {len(matches):,} matches, {len(points):,} points, and {len(approaches):,} explicit approach events.')

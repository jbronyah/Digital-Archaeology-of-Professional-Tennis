from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'build' / 'analysis'
OUT = ROOT / 'report' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
P=['1995–99','2000–04','2005–09','2010–14','2015–19','2020–25']
plt.rcParams.update({'font.size':11,'axes.titlesize':12,'figure.dpi':150,'savefig.dpi':240,
 'axes.spines.top':False,'axes.spines.right':False,'axes.prop_cycle':plt.cycler(color=['#176B87','#C75B39','#6D597A','#4B8753'])})
def save(name):
 plt.tight_layout();plt.savefig(OUT/name,bbox_inches='tight');plt.close()
s=pd.read_csv(BASE/'tables/03_tactical_period_estimates.csv')
fig,axes=plt.subplots(1,2,figsize=(8,3.1))
for ax,t in zip(axes,['ATP','WTA']):
 for m,label in [('snv_per100','Serve and volley'),('rally_approach_per100','Non serve approach')]:
  g=s[(s.tour==t)&(s.Tournament=='Wimbledon')&(s.metric==m)].set_index('period').reindex(P)
  y=g['mean'].where(g.n>=10);ax.plot(range(6),y,'o-',label=label);ax.fill_between(range(6),g.low.where(y.notna()),g.high.where(y.notna()),alpha=.12)
 ax.set_title(t+' Wimbledon');ax.set_xticks(range(6),P,rotation=40);ax.set_ylabel('Player entries per 100 points');ax.legend(fontsize=8)
save('02_net_play.png')
fig,axes=plt.subplots(1,2,figsize=(8,3.1))
for ax,m,title in [(axes[0],'baseline_fh_pct','Forehands among baseline plus one'),(axes[1],'plus_net_pct','Volleys and overheads among plus one')]:
 for t in ['ATP','WTA']:
  g=s[(s.tour==t)&(s.Tournament=='Wimbledon')&(s.metric==m)].set_index('period').reindex(P)
  y=g['mean'].where(g.n>=10);ax.plot(range(6),y,'o-',label=t);ax.fill_between(range(6),g.low.where(y.notna()),g.high.where(y.notna()),alpha=.12)
 ax.set_title(title,fontsize=11);ax.set_xticks(range(6),P,rotation=40);ax.set_ylabel('Share (%)');ax.legend(fontsize=9)
save('03_plus_one.png')
d=pd.read_csv(BASE/'tables/21_all_slam_pair_convergence.csv')
slams=['Australian Open','Roland Garros','Wimbledon','US Open'];fig,axes=plt.subplots(1,2,figsize=(8,3.4))
for ax,(col,title) in zip(axes,[('early_distance','ATP 1995 to 2004'),('late_distance','ATP 2015 to 2025')]):
 mat=np.zeros((4,4))
 for z in d[d.tour=='ATP'].itertuples():
  a,b=slams.index(z.slam_a),slams.index(z.slam_b);mat[a,b]=mat[b,a]=getattr(z,col)
 ax.imshow(mat,vmin=0,vmax=.25,cmap='Blues');ax.set_title(title)
 for (a,b),v in np.ndenumerate(mat):ax.text(b,a,f'{v:.3f}',ha='center',va='center',color='white' if v>.15 else 'black',fontsize=10)
 ax.set_xticks(range(4),['AO','RG','WIM','US']);ax.set_yticks(range(4),['AO','RG','WIM','US'])
save('04_convergence.png')
c=pd.read_csv(BASE/'tables/16_archive_composition.csv');fig,axes=plt.subplots(1,2,figsize=(8,3.1))
for ax,col,title in [(axes[0],'late_round_pct','Quarterfinals through finals'),(axes[1],'top5_player_appearance_pct','Top five player appearances')]:
 for t,g in c[c.Tournament=='Wimbledon'].groupby('tour'):
  g=g.set_index('period').reindex(P);ax.plot(range(6),g[col],'o-',label=t)
 ax.set_title(title,fontsize=11);ax.set_xticks(range(6),P,rotation=40);ax.set_ylabel('Share (%)');ax.set_ylim(0,105);ax.legend(fontsize=9)
save('05_archive.png')

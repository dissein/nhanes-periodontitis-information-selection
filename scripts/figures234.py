from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Patch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import roc_curve,roc_auc_score

import sys
PACKAGE=Path(__file__).resolve().parents[1]
O=Path(sys.argv[1]).resolve();O.mkdir(parents=True,exist_ok=False)
(O/'figures').mkdir();(O/'data').mkdir()
R=PACKAGE
S=R/'core'
PREV=PACKAGE/'results'
sources={}
def read(p):
    sources[p.relative_to(PACKAGE).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    return pd.read_csv(p,float_precision='round_trip')
def save(d,name):d.to_csv(O/'data'/name,index=False,encoding='utf-8-sig')
flow=read(S/'descriptives/inclusion_flow_sequential.csv')
cohort=read(S/'descriptives/Table1_cohort_summary.csv')
desc=read(S/'descriptives/Table1_all_variables_long.csv')
rank=read(S/'source_final/attribution_rankings.csv').replace({'algorithm':{'XGBoost':'TreeSHAP'}})
freq=read(PREV/'source_tables/S8_bootstrap500_frequencies.csv')
oof=read(PREV/'source_tables/S2_all_budget_OOF.csv')
random=read(PACKAGE/'random/random500_summary_all_budgets.csv')
pred=read(S/'temporal/paired_predictions.csv.gz')
ci=read(PREV/'source_tables/S5_temporal_metric_intervals.csv')
cal=read(S/'temporal/calibration_groups.csv')
dca=read(PACKAGE/'dca/all_budgets_DCA_pointwise_intervals.csv')

BLUE='#286F8E';ORANGE='#C67A45';GREEN='#587E68';RANDOM='#A4ABB3';INK='#26323B';CORE='#8A8097';LIGHT='#D4DCE1'
COL={'EBM':BLUE,'TreeSHAP':ORANGE,'Exhaustive':GREEN,'Random':RANDOM}
plt.rcParams.update({'font.family':'Arial','font.size':8.4,'axes.labelsize':8.5,'axes.titlesize':9.2,'xtick.labelsize':7.8,'ytick.labelsize':7.8,'axes.edgecolor':'#82909A','axes.linewidth':.65,'axes.spines.top':False,'axes.spines.right':False,'xtick.color':INK,'ytick.color':INK,'text.color':INK,'axes.labelcolor':INK,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','legend.fontsize':8,'legend.frameon':False,'savefig.facecolor':'white'})
FIGS=[]
def export(fig,stem):
    for ext in ['pdf','svg','png']:
        kwargs={'dpi':600 if ext=='tiff' else 300,'bbox_inches':'tight','pad_inches':.035}
        if ext=='tiff':kwargs['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(O/'figures'/f'{stem}.{ext}',**kwargs)
    FIGS.append((fig,stem))
def panel(ax,letter,title):ax.set_title(f'{letter}  {title}',loc='left',fontweight='bold',pad=12)
def grid(ax):ax.grid(axis='y',color='#E7EBED',lw=.5,zorder=0);ax.set_axisbelow(True);ax.tick_params(length=3,width=.6)

# Figure 2. All 18 budgets have complete 500-replicate random OOF results.
fig,axs=plt.subplots(1,2,figsize=(7.4,3.25));fig.subplots_adjust(left=.08,right=.985,bottom=.17,top=.78,wspace=.29)
handles=[]
for ax,metric,letter,title in zip(axs,['auc','brier'],['A','B'],['Discrimination','Prediction error']):
    for method,marker in [('EBM','o'),('TreeSHAP','^'),('Exhaustive','s')]:
        z=oof[oof.method.eq(method)].sort_values('k');assert set(z.k)==set(range(18))
        ln=ax.plot(z.k,z[metric],color=COL[method],lw=1.5,marker=marker,ms=2.5,markevery=1,label=method,zorder=4 if method=='TreeSHAP' else 3)[0]
        if metric=='auc':handles.append(ln)
    z=random[random.metric.eq(metric)].sort_values('k');assert z.k.tolist()==list(range(18))
    ax.fill_between(z.k,z.q025,z.q975,color=RANDOM,alpha=.19,lw=0,zorder=1)
    ln=ax.plot(z.k,z['median'],ls='--',color=RANDOM,marker='D',ms=3,lw=1.2,label='Random',zorder=2)[0]
    if metric=='auc':handles.append(ln)
    panel(ax,letter,title);ax.set(xlim=(-.3,17.3),xticks=list(range(0,18,2)),xlabel='Information units',ylabel='Weighted OOF AUC' if metric=='auc' else 'Weighted OOF Brier score')
    ax.set_xticks(range(1,18,2),minor=True)
    ax.tick_params(axis='x',which='minor',length=1.8,width=.45,color='#B4BEC5')
    ax.tick_params(axis='x',labelsize=7.4)
    ax.set_ylim((.66,.83) if metric=='auc' else (.165,.226));grid(ax)
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,1.005),ncol=4,columnspacing=1.5,handlelength=2.1)
export(fig,'Figure2_budget_performance')
save(oof,'Figure2_OOF_metrics.csv');save(random,'Figure2_random500_evaluated_budgets.csv')

# Figure 3. Shared row order makes method agreement and resampling frequency
# readable without repeating 17 labels in adjacent panels.
labels={'DMDEDUC2':'Education','INDFMIN2':'Family income','SmokingStatus':'Smoking status','DIQ010':'Diabetes','HUQ010':'General health','OHQ835':'Perceived gum disease','OHQ845':'Teeth/gum health','OHQ865':'Tooth appearance','OHQ855':'Tooth mobility','OHQ620':'Oral pain','OHQ640':'Work/school disruption','OHQ680':'Oral self-consciousness','OHQ870':'Interdental cleaning','OHQ875':'Therapeutic mouthwash','OHQ030':'Time since dental visit','OHQ850':'Periodontal treatment','OHQ860':'Bone loss reported'}
rp=rank.pivot(index='unit',columns='algorithm',values='rank');rp['mean']=rp[['EBM','TreeSHAP']].mean(axis=1)
units=rp.sort_values(['mean','EBM']).index.tolist();yy=np.arange(17)
fig=plt.figure(figsize=(7.4,4.95))
a=fig.add_axes([.245,.15,.19,.77]);b=fig.add_axes([.49,.15,.20,.77]);c=fig.add_axes([.77,.15,.22,.77])
imp=rank.pivot(index='unit',columns='algorithm',values='importance_fraction').loc[units]*100
a.barh(yy-.16,imp.EBM,height=.26,color=BLUE)
a.barh(yy+.16,imp.TreeSHAP,height=.26,color=ORANGE)
a.set(yticks=yy,yticklabels=[labels[u] for u in units],ylim=(16.6,-.6),xlim=(0,17),xticks=[0,8,16],xlabel='Attribution (%)')
a.tick_params(axis='y',length=0,labelsize=7.55,pad=7);a.tick_params(axis='x',length=3);a.spines['left'].set_visible(False)
mat=np.empty((17,4))
for j,(method,k) in enumerate([('EBM',3),('EBM',5),('TreeSHAP',3),('TreeSHAP',5)]):
    z=freq[freq.method.eq(method)&freq.k.eq(k)].set_index('unit').loc[units]
    assert abs(z.frequency.sum()-k)<1e-10
    mat[:,j]=z.frequency.to_numpy()*100
cmap=LinearSegmentedColormap.from_list('frequency',['#F7F9FA','#DCE5E9','#7C9AA8','#355A6D'])
b.imshow(mat[:,:2],aspect='auto',cmap=cmap,vmin=0,vmax=100,extent=(-.5,1.5,16.5,-.5),zorder=1)
warm=LinearSegmentedColormap.from_list('warm_frequency',['#FFFCF5','#F5E3C8','#DEB16C','#A9632B'])
b.imshow(mat[:,2:],aspect='auto',cmap=warm,vmin=0,vmax=100,extent=(1.5,3.5,16.5,-.5),zorder=1)
b.set_xlim(-.5,3.5)
for i in range(17):
    for j in range(4):
        v=mat[i,j]
        b.text(j,i,'0' if v==0 else f'{v:.1f}',ha='center',va='center',fontsize=6.6,color='white' if v>=70 else '#8D9BA4' if v==0 else INK,zorder=2)
b.axvline(1.5,color='white',lw=3)
b.set(xticks=range(4),xticklabels=['Top-3','Top-5','Top-3','Top-5'],yticks=[],ylim=(16.6,-.6))
b.xaxis.tick_bottom();b.tick_params(axis='x',length=0,pad=6,labelsize=6.8)
for spine in b.spines.values():spine.set_visible(False)
for label,color in zip(b.get_xticklabels(),[BLUE,BLUE,ORANGE,ORANGE]):label.set_color(color)
for m,k,mk in [('EBM',10,'o'),('TreeSHAP',8,'^')]:
    z=rank[rank.algorithm.eq(m)].sort_values('rank');c.plot(z['rank'],z.cumulative_fraction*100,color=COL[m],lw=1.6)
    row=z[z['rank'].eq(k)].iloc[0];value=row.cumulative_fraction*100
    c.scatter([k],[value],s=25,color=COL[m],marker=mk,zorder=4)
    c.plot([k,k],[0,value],color=COL[m],lw=.85,ls=(0,(3,2)),zorder=2)
c.axhline(80,color='#A4ADB3',ls='--',lw=.85);c.text(16.9,79,'80%',ha='right',va='top',fontsize=7.2,color='#84929C')
c.set(xlim=(0,17.5),ylim=(0,104),xticks=[0,8,10,17],yticks=[0,20,40,60,80,100],xlabel='Information units',ylabel='Attribution (%)');grid(c)
fig.text(.245,.955,'A  Attribution',fontweight='bold',fontsize=9.0)
fig.text(.49,.955,'B  Inclusion (%)',fontweight='bold',fontsize=9.0)
fig.text(.77,.955,'C  Cumulative',fontweight='bold',fontsize=9.0)
fig.canvas.draw()
# Center the shared key on the full visible figure, including the unit labels.
visible_bounds=fig.get_tightbbox(fig.canvas.get_renderer())
legend_center=(visible_bounds.x0+visible_bounds.x1)/(2*fig.get_figwidth())
fig.legend(handles=[Patch(facecolor=BLUE,label='EBM'),Patch(facecolor=ORANGE,label='TreeSHAP')],loc='upper center',bbox_to_anchor=(legend_center,1.035),ncol=2,handlelength=.9,handleheight=.9,handletextpad=.6,columnspacing=2,fontsize=8)
export(fig,'Figure3_priorities_stability')
save(rank,'Figure3_full_development_attribution.csv')
fr=pd.DataFrame({'unit':units,'label':[labels[u] for u in units],'EBM_rank':rp.loc[units,'EBM'],'TreeSHAP_rank':rp.loc[units,'TreeSHAP']}).reset_index(drop=True)
for j,name in enumerate(['EBM_Top3_percent','EBM_Top5_percent','TreeSHAP_Top3_percent','TreeSHAP_Top5_percent']):fr[name]=mat[:,j]
save(fr,'Figure3_ranking_and_frequencies.csv')

# Figure 4. The same model colors and line patterns apply to all three panels.
models=[('3-unit','EBM',3,CORE,(0,(2,1.6))),('TreeSHAP 8-unit','XGBoost',8,ORANGE,(0,(5,2))),('EBM 10-unit','EBM',10,BLUE,'-'),('Full 17-unit','EBM',17,INK,(0,(7,1.6,1.5,1.6)))]
fig,axs=plt.subplots(1,3,figsize=(7.4,3.05));fig.subplots_adjust(left=.07,right=.99,bottom=.19,top=.76,wspace=.39)
handles=[];rocdata=[]
w=pred.weight.to_numpy();y=pred.y.to_numpy();prevalence=float(np.average(y,weights=w))
for name,m,k,color,ls in models[::-1]:
    p=pred[f'{m}__k{k}'].to_numpy();fpr,tpr,_=roc_curve(y,p,sample_weight=w)
    axs[0].plot(fpr,tpr,color=color,ls=ls,lw=1.35,zorder=3 if k in [8,10] else 2)
    rocdata+=pd.DataFrame({'model':name,'FPR':fpr,'TPR':tpr}).to_dict('records')
    if k!=3:
        z=cal[cal.algorithm.eq(m)&cal.k.eq(k)].sort_values('mean_prediction')
        axs[1].plot(z.mean_prediction,z.observed,color=color,ls=ls,lw=1.35)
    mm='TreeSHAP' if m=='XGBoost' else m
    z=dca[dca.method.eq(mm)&dca.units.eq(k)&dca.threshold_probability.between(.05,.4000001)].sort_values('threshold_probability')
    assert len(z)>50
    axs[2].plot(z.threshold_probability*100,z.net_benefit,color=color,ls=ls,lw=1.4)
    ref=ci[ci.algorithm.eq(mm)&ci.k.eq(k)&ci.metric.eq('auc')].iloc[0]
    assert abs(float(roc_auc_score(y,p,sample_weight=w))-ref.estimate)<1e-10
for name,m,k,color,ls in models:handles.append(Line2D([],[],color=color,ls=ls,lw=1.6,label=name))
axs[0].plot([0,1],[0,1],color='#C6CED3',lw=.8,ls='--',zorder=0)
axs[0].set(xlim=(0,1),ylim=(0,1.02),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='1 − specificity',ylabel='Sensitivity')
axs[1].plot([0,1],[0,1],color='#C6CED3',lw=.8,ls='--',zorder=0)
axs[1].set(xlim=(0,1),ylim=(0,1.02),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='Predicted risk',ylabel='Observed risk')
ts=np.linspace(.05,.4,100);nb_all=prevalence-(1-prevalence)*ts/(1-ts)
axs[2].plot(ts*100,nb_all,color='#929EA7',ls='--',lw=1)
axs[2].axhline(0,color='#B7C0C6',lw=.85,ls=':')
axs[2].text(39.5,nb_all[-1]-.009,'Examine all',ha='right',va='top',fontsize=6.6,color='#82909A')
axs[2].text(6.0,.004,'Examine none',ha='left',va='bottom',fontsize=6.6,color='#82909A')
axs[2].set(xlim=(5,40),ylim=(-.13,.34),xticks=[5,20,40],yticks=[-.1,0,.1,.2,.3],xlabel='Risk threshold (%)',ylabel='Net benefit')
for ax,letter,title in zip(axs,['A','B','C'],['ROC','Calibration','Decision curve']):panel(ax,letter,title);grid(ax)
fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.54,1.01),ncol=4,columnspacing=1.2,handlelength=2.0)
export(fig,'Figure4_temporal_validation')
save(pd.DataFrame(rocdata),'Figure4_weighted_ROC_coordinates.csv')
save(cal[(cal.algorithm.eq('EBM')&cal.k.isin([10,17]))|(cal.algorithm.eq('XGBoost')&cal.k.eq(8))].drop(columns=['evaluation']),'Figure4_calibration_deciles.csv')
save(dca[(dca.method.eq('EBM')&dca.units.isin([3,10,17]))|(dca.method.eq('TreeSHAP')&dca.units.eq(8))],'Figure4_DCA_source.csv')

with PdfPages(O/'Main_figures.pdf') as pdf:
    for fig,stem in FIGS:pdf.savefig(fig,bbox_inches='tight',pad_inches=.04)
for fig,_ in FIGS:plt.close(fig)
(O/'chart_source_hashes.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
print('Updated Figures 2–4')

"""Descriptive tables and source-document provenance; no prediction fitting."""
import json,re
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import formal_common as c

def main():
 cfg=c.config();root=c.ROOT;out=root/'descriptives';out.mkdir(exist_ok=True)
 dictionary=pd.read_csv(root/'frozen_candidate_dictionary.csv');labels=dictionary.set_index('unit').label_en.to_dict();rows=[]
 for cy in ['G','H']:
  raw=pd.read_csv(root/f'data/raw_analysis_fields_{cy}.csv.gz');d=c.load(cy);assert np.array_equal(raw.ID.astype(str),d.ID)
  w=d.weight.to_numpy();den=w.sum();n=len(d)
  def cat(v,values):
   for level in pd.unique(values):
    ix=np.asarray(values==level);rows.append({'cycle':cy,'variable':v,'label':labels.get(v,v),'level':str(level),'n':int(ix.sum()),'weighted_percent':100*float(w[ix].sum()/den),'weighted_mean':None,'weighted_SD':None,'denominator_n':n,'denominator':'all included participants; missing response states retained'})
  age=d.RIDAGEYR.to_numpy();mean=np.average(age,weights=w)
  rows.append({'cycle':cy,'variable':'RIDAGEYR','label':'Age','level':'mean_SD','n':n,'weighted_percent':None,'weighted_mean':mean,'weighted_SD':np.sqrt(np.average((age-mean)**2,weights=w)),'denominator_n':n,'denominator':'all included participants; population weighted SD, not SE'})
  cat('RIAGENDR',d.RIAGENDR);cat('CDC_AAP_class',raw.cdc_class);cat('CDC_AAP_mod_severe',d.y)
  for u in cfg['units']:cat(u,d[u].fillna('NUMERIC_MISSING'))
 long=pd.DataFrame(rows)
 options={}
 for row in dictionary.itertuples():
  options[row.unit]={k:v.strip() for k,v in re.findall(r'(?:^|;\s*)(\d+)=([^;]*)',str(row.official_options_G))}
 def level_label(row):
  if row.variable=='RIAGENDR':return {'CAT_1':'Male','CAT_2':'Female'}[row.level]
  if row.variable=='CDC_AAP_mod_severe':return {'0':'No/mild periodontitis','1':'Moderate/severe periodontitis'}[row.level]
  if row.level.startswith('CAT_'):return options.get(row.variable,{}).get(row.level[4:],row.level)
  if row.variable in ['OHQ870','OHQ875'] and row.level!='NUMERIC_MISSING':return row.level+' days'
  return row.level
 long['level_label']=[level_label(row) for row in long.itertuples()]
 long.to_csv(out/'Table1_all_variables_long.csv',index=False,encoding='utf-8-sig')
 def display(row):
  return f'{row.weighted_mean:.1f} ({row.weighted_SD:.1f})' if row.level=='mean_SD' else f'{row.n} ({row.weighted_percent:.1f}%)'
 long['display']=[display(r) for r in long.itertuples()]
 wide=long.pivot(index=['variable','label','level','level_label'],columns='cycle',values='display').reset_index().rename(columns={'G':'2011–2012 development','H':'2013–2014 validation'})
 wide.to_csv(out/'Table1_all_variables_display.csv',index=False,encoding='utf-8-sig')
 # Sequential exclusions are reconstructed from original DEMO, not subtracted from unrelated denominators.
 import pyreadstat
 audit=json.loads((root/'pretraining_data_audit.json').read_text(encoding='utf-8'));flow=[]
 for cy in ['G','H']:
  demo=next(p for p in audit['original_sources_sha256'] if p.endswith(f'DEMO_{cy}.xpt'));a=pyreadstat.read_xport(str(c.ROOT.parent/demo))[0]
  old=pd.read_csv(root/'data/inclusion_flow.csv');counts=old[old.cycle==cy].set_index('stage').n.to_dict()
  previous=len(a)
  stages=[('All DEMO participants',len(a)),('Age >=30 years',int(a.RIDAGEYR.ge(30).sum())),('Positive MEC weight',int((a.RIDAGEYR.ge(30)&a.WTMEC2YR.gt(0)).sum())),('Complete periodontal examination (OHDPDSTS=1)',counts['complete_periodontal_exam']),('Classifiable CDC/AAP outcome; included',counts['CDC_classifiable_included'])]
  for stage,n in stages:flow.append({'cycle':cy,'stage':stage,'included_n':n,'excluded_at_step':previous-n});previous=n
 pd.DataFrame(flow).to_csv(out/'inclusion_flow_sequential.csv',index=False,encoding='utf-8-sig')
 plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False})
 fig,axs=plt.subplots(1,2,figsize=(12,8))
 for ax,cy,title in zip(axs,['G','H'],['2011–2012 development','2013–2014 validation']):
  ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title(title)
  for j,r in enumerate([r for r in flow if r['cycle']==cy]):
   y=.87-j*.18;box=FancyBboxPatch((.06,y-.055),.88,.105,boxstyle='round,pad=.01',facecolor='#edf4fa',edgecolor='#4a718a');ax.add_patch(box)
   ax.text(.5,y,f"{r['stage']}\nn = {r['included_n']:,}",ha='center',va='center',fontsize=10)
   if j:ax.annotate('',xy=(.5,y+.06),xytext=(.5,y+.12),arrowprops={'arrowstyle':'->','color':'#4a718a'});ax.text(.53,y+.09,f"Excluded at this step {r['excluded_at_step']:,}",va='center',fontsize=9)
 fig.tight_layout()
 for ext in ['png','svg']:fig.savefig(out/f'participant_flow.{ext}',dpi=180)
 plt.close(fig)
 (out/'descriptive_notes.md').write_text('Survey-weighted descriptive summaries use all included participants as percentage denominators. Age uses weighted mean and population SD. Refused, unknown, and system-missing states remain separate. Variable-specific cycle tests are reproduced separately by scripts/table1.py.',encoding='utf-8')
 print('DESCRIPTIVES_COMPLETE',len(long),flush=True)

if __name__=='__main__':main()

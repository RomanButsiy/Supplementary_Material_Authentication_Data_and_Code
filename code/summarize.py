from pathlib import Path
import numpy as np,pandas as pd,json,sys,platform,sklearn,scipy
from scipy.stats import beta
from sklearn.metrics import roc_curve
from run_analysis import ROOT,OUT,load,group,partition,MODELS,calc,score_items
def read_csv_exact(*args, **kwargs):
    kwargs.setdefault('float_precision', 'round_trip')
    return pd.read_csv(*args, **kwargs)

METRICS=['Accuracy','Balanced_Accuracy','F1_genuine','FAR','FRR','HTER','AUC','EER']

def summary(df,by,metrics=METRICS):
 rows=[]
 for key,g in df.groupby(by,sort=False,dropna=False):
  if not isinstance(key,tuple):key=(key,)
  vals=g.groupby('subject')[ [m for m in metrics if m in g] ].mean()
  a=vals.to_numpy();rng=np.random.default_rng(271828);inds=rng.integers(0,len(a),(20000,len(a)))
  reps=a[inds].mean(1);lo,hi=np.quantile(reps,[.025,.975],axis=0)
  row=dict(zip(by,key));row['N_subjects']=len(vals)
  for j,col in enumerate(vals):row[col]=a[:,j].mean();row[col+'_lo']=lo[j];row[col+'_hi']=hi[j]
  rows.append(row)
 return pd.DataFrame(rows)

def main():
 d=load();a=read_csv_exact(OUT/'primary_all.csv');s=summary(a,['K','model','operating']);s.to_csv(OUT/'summary_primary.csv',index=False)
 print(s[s.operating=='default'][['K','model','FAR','FRR','HTER','HTER_lo','HTER_hi','Accuracy','F1_genuine']].to_string(index=False))
 # Endpoint display as directions first; two directions count once per claimant in bootstrap.
 summary(a,['K','model','operating','direction']).to_csv(OUT/'summary_direction.csv',index=False)
 # Counts: repeated comparison of the same impostor traces is not independent sampling.
 counts=a.groupby(['K','model','operating'])[['TP','TN','FP','FN']].sum().reset_index()
 for kind,err,total in [('FAR','FP',['FP','TN']),('FRR','FN',['FN','TP'])]:
  n=counts[total].sum(axis=1);e=counts[err]
  counts[kind+'_N']=n;counts[kind+'_upper95_independence_reference']=np.where(e==n,1,beta.ppf(.95,e+1,n-e))
 counts.to_csv(OUT/'counts_and_bounds.csv',index=False)
 # validation-constrained threshold analysis, not test-set threshold fitting
 operating=[]
 for tag,z in score_items():
  p,session,kk=tag.split('_');k=int(kk[1:])
  for m in ['Linear SVM','MLP','SIC','Nearest Neighbors']:
   sv=z['v_'+m.replace(' ','_')];ss=z['s_'+m.replace(' ','_')];yv=z['yv'];yt=z['y']
   fr,tr,th=roc_curve(yv,sv,drop_intermediate=False)
   for target in [.01,.001]:
    ids=np.flatnonzero(fr<=target);ii=ids[np.argmax(tr[ids])]
    operating.append(dict(subject=int(p),direction=session,K=k,model=m,target_FAR=target,threshold=th[ii],validation_FAR=fr[ii],validation_FRR=1-tr[ii],**calc(yt,ss,th[ii])))
 op=pd.DataFrame(operating);op.to_csv(OUT/'security_operating_points.csv',index=False);summary(op,['K','model','target_FAR']).to_csv(OUT/'summary_security_points.csv',index=False)
 # claimant-paired differences K5 minus K1
 deltas=[]
 for (m,oper),g in a.groupby(['model','operating']):
  one=g[g.K==1].groupby('subject')[METRICS].mean();five=g[g.K==5].groupby('subject')[METRICS].mean()
  diff=five-one;diff['model']=m;diff['operating']=oper;diff['subject']=diff.index;deltas.append(diff)
 di=pd.concat(deltas,ignore_index=True);summary(di,['model','operating']).to_csv(OUT/'summary_paired_differences.csv',index=False)
 # Counts for each partition and descriptor length.
 countrows=[]
 for key,x in d.items():
  p,session=key.split('_')
  for k in [1,5]:countrows.append(dict(subject=int(p),session=session,K=k,cycles=len(x),fit=len(partition(x,k,'fit')),calibration=len(partition(x,k,'cal')),test=len(group(x,k))))
 pd.DataFrame(countrows).to_csv(OUT/'sample_counts.csv',index=False)
 if (OUT/'secondary_all.csv').exists():
  sec=read_csv_exact(OUT/'secondary_all.csv');summary(sec,['K','model','experiment','setting'],[m for m in METRICS if m in sec]+['width']).to_csv(OUT/'summary_secondary.csv',index=False)
 if (OUT/'unseen_all.csv').exists():
  u=read_csv_exact(OUT/'unseen_all.csv');summary(u,['K','model','condition']).to_csv(OUT/'summary_unseen.csv',index=False)

def extra_summaries():
 s=read_csv_exact(OUT/'secondary_all.csv');out=[]
 for (m,k,di,p),g in s[s.experiment=='zone'].groupby(['model','K','direction','subject']):
  b=g[g.setting=='full'].iloc[0]
  for _,row in g[g.setting!='full'].iterrows():
   out.append(dict(subject=p,direction=di,model=m,K=k,setting=row.setting,HTER=row.HTER-b.HTER,FAR=row.FAR-b.FAR,FRR=row.FRR-b.FRR))
 summary(pd.DataFrame(out),['model','K','setting'],['FAR','FRR','HTER']).to_csv(OUT/'summary_zone_differences.csv',index=False)
 u=read_csv_exact(OUT/'unseen_all.csv');out=[]
 for (m,k,di,p,f),g in u.groupby(['model','K','direction','subject','fold']):
  a=g[g.condition=='known'].iloc[0];b=g[g.condition=='unseen'].iloc[0]
  out.append(dict(subject=p,direction=di,fold=f,model=m,K=k,FAR=b.FAR-a.FAR,HTER=b.HTER-a.HTER))
 summary(pd.DataFrame(out),['model','K'],['FAR','HTER']).to_csv(OUT/'summary_identity_differences.csv',index=False)
 e=s[s.experiment=='equal_count'];out=[]
 for (p,di),g in e.groupby(['subject','direction']):
  a=g[g.setting=='single_cycle_intervals'].iloc[0];b=g[g.setting=='five_cycle_intervals'].iloc[0]
  out.append(dict(subject=p,direction=di,width_ratio=b.width/a.width,FRR=b.FRR-a.FRR,HTER=b.HTER-a.HTER))
 summary(pd.DataFrame(out).assign(experiment='equal_count'),['experiment'],['width_ratio','FRR','HTER']).to_csv(OUT/'summary_equal_count_differences.csv',index=False)

if __name__=='__main__':
 main()
 extra_summaries()


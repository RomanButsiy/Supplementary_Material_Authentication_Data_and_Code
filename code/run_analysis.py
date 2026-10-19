"""Reproducible session-disjoint ECG reanalysis; genuine=1, impostor=0."""
from pathlib import Path
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']: os.environ[key]='1'
import json,hashlib,warnings,time,argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve,roc_auc_score,confusion_matrix
from sklearn.pipeline import make_pipeline
from joblib import Parallel,delayed
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('ECG_RESULTS_DIR', str(ROOT/'results'))).resolve()
OUT.mkdir(parents=True, exist_ok=True)
CACHE=ROOT/'data'/'cycles.npz'; SEED=20261012
SIC_CONFIG = {1: {'kappa': 5.7, 'eta': 348 / 360, 'required_count': 348},
              5: {'kappa': 8.4, 'eta': 328 / 360, 'required_count': 328}}
# Selected during exploratory assessment of this cohort; fixed across claimants and directions.
MODELS=['SIC','AdaBoost','MLP','Random Forest','Decision Tree','Gaussian Process','Linear SVM','Nearest Neighbors']

def load():
 # The supplied rows are already aligned, mean-centered cardiac cycles.
 with np.load(CACHE, allow_pickle=False) as z:
  return {key:z[key] for key in z.files}

def score_items(k=None):
 """Read compact published scores or individual scores during a new run."""
 if (OUT/'scores.npz').exists():
  with np.load(OUT/'scores.npz',allow_pickle=False) as z:
   tags=sorted({key.split('__',1)[0] for key in z.files})
   for tag in tags:
    if k is not None and not tag.endswith(f'_K{k}'):continue
    prefix=tag+'__'
    yield tag,{key[len(prefix):]:z[key] for key in z.files if key.startswith(prefix)}
 else:
  pattern='*.npz' if k is None else f'*_K{k}.npz'
  for path in sorted((OUT/'scores').glob(pattern)):
   with np.load(path,allow_pickle=False) as z:
    yield path.stem,{key:z[key] for key in z.files}

def group(a,k):return a[:len(a)//k*k].reshape(-1,k,a.shape[1]).mean(1)
def partition(a,k,part):
 cut=int(np.floor(.7*len(a)));return group(a[:cut] if part=='fit' else a[cut:],k)
def model(name,seed):
 clfs={'Nearest Neighbors':lambda:KNeighborsClassifier(n_neighbors=3,weights='uniform',metric='minkowski',p=2,n_jobs=1),'Linear SVM':lambda:SVC(kernel='linear',C=.025,probability=False,random_state=seed),'Gaussian Process':lambda:GaussianProcessClassifier(1.*RBF(1.),n_restarts_optimizer=0,max_iter_predict=100,random_state=seed),'Decision Tree':lambda:DecisionTreeClassifier(max_depth=5,random_state=seed),'Random Forest':lambda:RandomForestClassifier(max_depth=5,n_estimators=10,max_features=1,random_state=seed,n_jobs=1),'MLP':lambda:MLPClassifier(hidden_layer_sizes=(100,),activation='relu',solver='adam',alpha=1,max_iter=1000,random_state=seed),'AdaBoost':lambda:AdaBoostClassifier(n_estimators=50,learning_rate=1,random_state=seed)}
 return make_pipeline(StandardScaler(),clfs[name]())
def balanced_neg(d,ids,session,k,n,seed):
 rng=np.random.default_rng(seed);ids=list(ids);q,r=divmod(n,len(ids));rows=[]
 for j,p in enumerate(ids):
  a=partition(d[f'{p:03}_{session}'],k,'fit');count=q+(j<r);assert count<=len(a)
  rows.append(a[rng.choice(len(a),count,replace=False)])
 return np.vstack(rows)
def calc(y,s,t):
 pred=(s>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();far=fp/(tn+fp);frr=fn/(tp+fn)
 return dict(TP=int(tp),TN=int(tn),FP=int(fp),FN=int(fn),Accuracy=(tp+tn)/len(y),Balanced_Accuracy=1-(far+frr)/2,F1_genuine=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,FAR=far,FRR=frr,HTER=(far+frr)/2,AUC=roc_auc_score(y,s))
def eer(y,s):
 fpr,tpr,thr=roc_curve(y,s,drop_intermediate=False);frr=1-tpr;diff=fpr-frr;i=np.searchsorted(diff,0)
 if i==0:return float((fpr[0]+frr[0])/2)
 if i==len(diff):return float((fpr[-1]+frr[-1])/2)
 frac=-diff[i-1]/(diff[i]-diff[i-1]) if diff[i]!=diff[i-1] else 0.
 return float(fpr[i-1]+frac*(fpr[i]-fpr[i-1]))
def cal_threshold(y,s):
 fpr,tpr,thr=roc_curve(y,s,drop_intermediate=False);risk=(fpr+1-tpr)/2
 idx=np.flatnonzero(np.isclose(risk,risk.min(),atol=1e-12));return float(thr[idx[0]])
def fit_scorer(name,x,y,seed,k=1):
 if name=='SIC':
  genuine=x[y==1];mu=genuine.mean(0);sd=genuine.std(0,ddof=1)
  cfg=SIC_CONFIG[k]; kappa=cfg['kappa']; eta=cfg['eta']
  def score(a):return np.mean(np.abs(a-mu)<=kappa*sd,axis=1)
  return score,eta,dict(mu=mu,sd=sd,width=float(np.mean(2*kappa*sd)),kappa=kappa,eta=eta),None
 clf=model(name,seed)
 with warnings.catch_warnings(record=True) as ws:
  warnings.simplefilter('always');clf.fit(x,y)
 def score(a):
  if name=='Linear SVM':return clf.decision_function(a)
  return clf.predict_proba(a)[:,list(clf.classes_).index(1)]
 return score,(0 if name=='Linear SVM' else .5),{'warnings':[str(w.message) for w in ws]},clf

def design(d,p,session,k,background=None,mask=None):
 ids=sorted({int(key[:3]) for key in d});bg=[j for j in ids if j!=p] if background is None else list(background)
 test='S2' if session=='S1' else 'S1';sd=SEED+p*101+(session=='S2')*10001+k*13
 gf=partition(d[f'{p:03}_{session}'],k,'fit');nf=balanced_neg(d,bg,session,k,len(gf),sd)
 vg=partition(d[f'{p:03}_{session}'],k,'cal');vn=np.vstack([partition(d[f'{j:03}_{session}'],k,'cal') for j in bg])
 tg=group(d[f'{p:03}_{test}'],k);tn=np.vstack([group(d[f'{j:03}_{test}'],k) for j in bg])
 x=np.vstack([gf,nf]);y=np.r_[np.ones(len(gf),int),np.zeros(len(nf),int)]
 xv=np.vstack([vg,vn]);yv=np.r_[np.ones(len(vg),int),np.zeros(len(vn),int)]
 xt=np.vstack([tg,tn]);yt=np.r_[np.ones(len(tg),int),np.zeros(len(tn),int)]
 if mask is not None:x=x[:,mask];xv=xv[:,mask];xt=xt[:,mask]
 order=np.random.default_rng(sd).permutation(len(y));return x[order],y[order],xv,yv,xt,yt,sd

def primary_one(p,session,k):
 with threadpool_limits(limits=1):
  d=load();x,y,xv,yv,xt,yt,sd=design(d,p,session,k);rows=[];preds={};vals={};sic={}
  for name in MODELS:
   st=time.perf_counter();score,th,info,clf=fit_scorer(name,x,y,sd,k=k);fit_s=time.perf_counter()-st
   sv=score(xv);ss=score(xt);ct=cal_threshold(yv,sv)
   for typ,t in [('default',th),('calibrated',ct)]:rows.append(dict(subject=p,direction=session+'_'+('S2' if session=='S1' else 'S1'),K=k,model=name,operating=typ,threshold=t,fit_seconds=fit_s,n_fit=len(y),n_val=len(yv),**calc(yt,ss,t),EER=eer(yt,ss)))
   preds[name]=ss;vals[name]=sv
   if name=='SIC':sic=info
  tag=f'{p:03}_{session}_K{k}';dest=OUT/'scores';dest.mkdir(exist_ok=True)
  np.savez_compressed(dest/(tag+'.npz'),y=yt,yv=yv,**{'s_'+n.replace(' ','_'):a for n,a in preds.items()},**{'v_'+n.replace(' ','_'):a for n,a in vals.items()},mu=sic['mu'],sd=sic['sd'],sic_kappa=sic['kappa'],sic_eta=sic['eta'])
  pd.DataFrame(rows).to_csv(OUT/('primary_'+tag+'.csv'),index=False);print('DONE',tag,flush=True);return rows

def secondary_one(p,session,k):
 with threadpool_limits(limits=1):
  d=load();x,y,xv,yv,xt,yt,sd=design(d,p,session,k);rows=[];axes=np.arange(360);u=axes*275/359
  masks={'full':np.ones(360,bool),'omit_P_R':~(u<48),'omit_R_T':~((u>=48)&(u<131)),'omit_T_P':~(u>=131)}
  for name in ['Linear SVM','MLP']:
   for label,mask in masks.items():
    scorer,th,info,clf=fit_scorer(name,x[:,mask],y,sd);ss=scorer(xt[:,mask])
    rows.append(dict(subject=p,direction=session,K=k,model=name,experiment='zone',setting=label,**calc(yt,ss,th)))
    if label!='full':continue
    for amplitude in [1,2,4,8]:
     rng=np.random.default_rng(sd+amplitude*31);shifts=rng.uniform(-amplitude,amplitude,len(xt))
     a=np.array([np.interp(axes-shift,axes,v) for v,shift in zip(xt,shifts)]);a-=a.mean(1,keepdims=True)
     rows.append(dict(subject=p,direction=session,K=k,model=name,experiment='jitter',setting=amplitude,**calc(yt,scorer(a),th)))
  if k==5:
   gf=partition(d[f'{p:03}_{session}'],1,'fit');gc=gf[:len(gf)//5*5].reshape(-1,5,360)[:,0,:]
   for setting,gg in [('single_cycle_intervals',gc),('five_cycle_intervals',group(gf,5))]:
    score,th,info,_=fit_scorer('SIC',gg,np.ones(len(gg),int),sd,k=5)
    rows.append(dict(subject=p,direction=session,K=5,model='SIC',experiment='equal_count',setting=setting,width=info['width'],**calc(yt,score(xt),th)))
  pd.DataFrame(rows).to_csv(OUT/f'secondary_{p:03}_{session}_K{k}.csv',index=False);print('SECONDARY',p,session,k,flush=True);return rows

def unseen_one(p,session,k):
 with threadpool_limits(limits=1):
  d=load();ids=sorted({int(key[:3]) for key in d});other=[i for i in ids if i!=p];rows=[];ts='S2' if session=='S1' else 'S1'
  for fold in [0,1]:
   bg=other[fold::2];hold=other[1-fold::2];x,y,xv,yv,xt,yt,sd=design(d,p,session,k,bg)
   ug=group(d[f'{p:03}_{ts}'],k);un=np.vstack([group(d[f'{j:03}_{ts}'],k) for j in hold]);xu=np.vstack([ug,un]);yu=np.r_[np.ones(len(ug),int),np.zeros(len(un),int)]
   for name in ['Linear SVM','MLP']:
    score,th,info,clf=fit_scorer(name,x,y,sd,k=k)
    for cond,xx,yy in [('known',xt,yt),('unseen',xu,yu)]:rows.append(dict(subject=p,direction=session,K=k,fold=fold,model=name,condition=cond,**calc(yy,score(xx),th)))
  pd.DataFrame(rows).to_csv(OUT/f'unseen_{p:03}_{session}_K{k}.csv',index=False);print('UNSEEN',p,session,k,flush=True);return rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['primary','secondary','unseen','all'],default='all');ap.add_argument('--jobs',type=int,default=3);args=ap.parse_args()
 d=load();ids=sorted({int(key[:3]) for key in d});jobs=[(i,s,k) for i in ids for s in ['S1','S2'] for k in [1,5]]
 for stage,fun in [('primary',primary_one),('secondary',secondary_one),('unseen',unseen_one)]:
  if args.stage not in ['all',stage]:continue
  todo=[j for j in jobs if not (OUT/f'{stage}_{j[0]:03}_{j[1]}_K{j[2]}.csv').exists()];print('STAGE',stage,'jobs',len(todo),flush=True)
  Parallel(n_jobs=args.jobs,backend='loky')(delayed(fun)(*j) for j in todo)
  pd.concat([pd.read_csv(p,float_precision='round_trip') for p in sorted(OUT.glob(stage+'_[0-9]*.csv'))],ignore_index=True).to_csv(OUT/(stage+'_all.csv'),index=False)
if __name__=='__main__':main()

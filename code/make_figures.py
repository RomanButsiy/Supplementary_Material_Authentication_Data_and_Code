"""Publication figures from the supplied descriptors and measured evaluation outputs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import roc_curve
import os
from run_analysis import ROOT,OUT,SIC_CONFIG,score_items
def read_csv_exact(*args, **kwargs):
    kwargs.setdefault('float_precision', 'round_trip')
    return pd.read_csv(*args, **kwargs)

F=Path(os.environ.get('ECG_FIGURES_DIR', str(OUT))).resolve();F.mkdir(parents=True, exist_ok=True)
NAMES=['SIC','AdaBoost','MLP','Random Forest','Decision Tree','Gaussian Process','Linear SVM','Nearest Neighbors']
DISPLAY={'MLP':'Neural network (MLP)','Nearest Neighbors':'Nearest neighbors'}
def save(fig,name):
    fig.tight_layout(pad=1.2)
    fig.savefig(F/(name+'.pdf'),bbox_inches='tight')
    plt.close(fig)
def tidy(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.grid(axis='y',alpha=.20)
def matched_count_figure():
    a=read_csv_exact(OUT/'summary_secondary.csv')
    a=a[a.experiment=='equal_count'].set_index('setting')
    keys=['single_cycle_intervals','five_cycle_intervals']
    labels=['Individual cycles','Five-cycle means']
    colors=['#295B86','#AD5933']
    fig,(left,right)=plt.subplots(1,2,figsize=(7.2,3.5),gridspec_kw={'width_ratios':[.8,1.6]})
    for j,key in enumerate(keys):
        z=a.loc[key]
        left.errorbar(j,z.width,yerr=[[z.width-z.width_lo],[z.width_hi-z.width]],fmt=['o','s'][j],color=colors[j],ms=6,capsize=4,lw=1.3)
        left.text(j,z.width_hi+.012,f'{z.width:.3f}',ha='center',fontsize=9)
        vals=np.array([z.FAR,z.FRR,z.HTER])*100
        low=np.array([z.FAR_lo,z.FRR_lo,z.HTER_lo])*100
        high=np.array([z.FAR_hi,z.FRR_hi,z.HTER_hi])*100
        right.errorbar(np.arange(3)+[-.13,.13][j],vals,yerr=np.vstack([vals-low,high-vals]),fmt=['o','s'][j],color=colors[j],label=labels[j],capsize=4,ms=6,lw=1.3)
    left.set_xticks([0,1],['Individual\ncycles','Five-cycle\nmeans'])
    left.set(xlim=(-.5,1.5),ylim=(0,.30),ylabel='Mean envelope width (recording units)')
    left.set_title('(a) Envelope width',loc='left',fontsize=11)
    right.set_xticks(range(3),['FAR','FRR','HTER'])
    right.set(ylim=(-1,58),ylabel='Error rate (%)')
    right.set_title('(b) Verification errors',loc='left',fontsize=11)
    right.legend(frameon=False,fontsize=9,loc='upper right')
    for ax in (left,right):tidy(ax)
    save(fig,'figureS2_sic_matched_count')

def main():
    d=np.load(ROOT/'data'/'cycles.npz')
    a=d['002_S1'][:3];fig,ax=plt.subplots(figsize=(7.2,3.7))
    for j,v in enumerate(a):
        xx=j+np.arange(360)/359
        ax.plot(xx,v,lw=1.4,label='Observed, zone-aligned cycles' if j==0 else None)
        for h,q in enumerate([0,48/275,131/275]):
            pos=j+q;ax.axvline(pos,ls=['-','--',':'][h],lw=.65,alpha=.55)
            val=np.interp(q,np.arange(360)/359,v)
            ax.annotate(['P','R','T'][h],(pos,val),xytext=(3,10),textcoords='offset points',fontsize=9)
        for lo,hi,txt in [(0,48/275,'P–R'),(48/275,131/275,'R–T'),(131/275,1,'T–P')]:
            ax.text(j+(lo+hi)/2,-.36,txt,ha='center',va='center',fontsize=9)
    ax.axvline(3,lw=.65,alpha=.55)
    ax.set(xlim=(-.02,3.03),ylim=(-.43,1.14),xlabel='Cycle index + normalized phase (not physical time)',ylabel='Centered ECG amplitude (recording units)')
    ax.set_xticks(np.arange(0,3.1,.5));tidy(ax);save(fig,'figure1_zone_cycles')
    s=read_csv_exact(OUT/'summary_primary.csv');s=s[s.operating=='default']
    fig,ax=plt.subplots(figsize=(7.2,4.6))
    for k,delta,marker in [(1,-.13,'o'),(5,.13,'s')]:
        t=s[s.K==k].set_index('model').loc[NAMES]
        v=t.HTER.to_numpy()*100;err=np.vstack([v-t.HTER_lo.to_numpy()*100,t.HTER_hi.to_numpy()*100-v])
        ax.errorbar(v,np.arange(8)+delta,xerr=err,fmt=marker,ms=5,capsize=3,lw=1.2,label=f'K = {k}')
    ax.set_yticks(np.arange(8),[DISPLAY.get(n,n) for n in NAMES]);ax.invert_yaxis()
    ax.set_xscale('symlog',linthresh=1);ax.set_xticks([0,.5,1,2,5,10,20,40]);ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set(xlim=(-.01,42),xlabel='Half total error rate (%) · symmetric-log scale')
    ax.legend(frameon=False,loc='lower right');tidy(ax);ax.grid(axis='x',alpha=.2);save(fig,'figure2_verification')
    # Rejection uses strict h < eta; display the margin on a shared boundary.
    grid=np.unique(np.r_[np.linspace(-1,.10,900),0.0]);fig,ax=plt.subplots(figsize=(7.2,3.8))
    for k,cl,ls in [(1,1,'-'),(5,1,'--'),(1,0,':'),(5,0,'-.')]:
        curves=[]
        for tag,z in score_items(k):
            v=np.sort(z['s_SIC'][z['y']==cl]-SIC_CONFIG[k]['eta'])
            curves.append(np.searchsorted(v,grid,side='left')/len(v))
        ax.plot(grid,np.mean(curves,axis=0),ls=ls,lw=1.8,label=f'{"Genuine" if cl else "Impostor"}, K = {k}')
    ax.axvline(0,ls='--',lw=1,alpha=.65)
    ax.annotate('Accept at margin ≥ 0',(0,.45),xytext=(-.62,.69),arrowprops={'arrowstyle':'->','lw':.9},fontsize=10)
    ax.set(xlim=(-1.01,.105),ylim=(-.015,1.025),xlabel='SIC coverage margin: h(z) − η(K)',ylabel='Mean fraction with margin < abscissa')
    ax.set_xticks([-1,-.8,-.6,-.4,-.2,0,.1])
    ax.legend(frameon=False,loc='upper left',fontsize=9);tidy(ax);save(fig,'figure3_sic_scores')
    sec=read_csv_exact(OUT/'summary_secondary.csv');zone=sec[sec.experiment=='zone']
    fig,ax=plt.subplots(figsize=(7.2,3.9))
    settings=['full','omit_P_R','omit_R_T','omit_T_P']
    for m,k,delta,marker in [('Linear SVM',1,-.24,'o'),('Linear SVM',5,-.08,'s'),('MLP',1,.08,'^'),('MLP',5,.24,'D')]:
        t=zone[(zone.model==m)&(zone.K==k)].set_index('setting').loc[settings];v=t.HTER.to_numpy()*100
        ax.errorbar(np.arange(4)+delta,v,yerr=np.vstack([v-t.HTER_lo.to_numpy()*100,t.HTER_hi.to_numpy()*100-v]),fmt=marker,ms=5,capsize=3,lw=1.1,label=f'{m}, K = {k}')
    ax.set_xticks(range(4),['Full cycle','Omit P–R','Omit R–T','Omit T–P']);ax.set(ylabel='Half total error rate (%)',ylim=(-.2,8))
    ax.legend(frameon=False,ncols=2,fontsize=9,loc='upper left');tidy(ax);save(fig,'figure4_zone_omission')
    fig,ax=plt.subplots(figsize=(7.2,3.9))
    for m,k,marker,ls in [('Linear SVM',1,'o','-'),('Linear SVM',5,'s','--'),('MLP',1,'^',':'),('MLP',5,'D','-.')]:
        t=sec[(sec.model==m)&(sec.K==k)&(sec.experiment=='jitter')].copy();t['amp']=t.setting.astype(int);t=t.sort_values('amp')
        base=zone[(zone.model==m)&(zone.K==k)&(zone.setting=='full')].iloc[0]
        x=np.r_[0,t.amp.to_numpy()];v=np.r_[base.HTER,t.HTER.to_numpy()]*100
        lo=np.r_[base.HTER_lo,t.HTER_lo.to_numpy()]*100;hi=np.r_[base.HTER_hi,t.HTER_hi.to_numpy()]*100
        ax.plot(x,v,marker=marker,ls=ls,lw=1.7,ms=5,label=f'{m}, K = {k}')
        # No overlaid ribbons: keep all four curves readable in monochrome.
    ax.set_xticks([0,1,2,4,8]);ax.set(xlabel='Maximum residual phase displacement (samples of 360)',ylabel='Half total error rate (%)',ylim=(-.3,20))
    ax.legend(frameon=False,fontsize=9);tidy(ax);save(fig,'figure5_phase_sensitivity')
    u=read_csv_exact(OUT/'summary_unseen.csv');fig,ax=plt.subplots(figsize=(7.2,3.9));keys=[('Linear SVM',1),('Linear SVM',5),('MLP',1),('MLP',5)]
    for cond,delta,marker in [('known',-.11,'o'),('unseen',.11,'s')]:
        t=u[u.condition==cond].set_index(['model','K']).loc[keys];v=t.FAR.to_numpy()*100
        ax.errorbar(np.arange(4)+delta,v,yerr=np.vstack([v-t.FAR_lo.to_numpy()*100,t.FAR_hi.to_numpy()*100-v]),fmt=marker,ms=6,capsize=3,lw=1.3,label='Training-background identities' if cond=='known' else 'Held-out impostor identities')
    ax.set_xticks(range(4),['Linear SVM\nK = 1','Linear SVM\nK = 5','MLP\nK = 1','MLP\nK = 5']);ax.set(ylabel='False acceptance rate (%)',ylim=(-.5,32));ax.legend(frameon=False,fontsize=9);tidy(ax);save(fig,'figure6_identity_transfer')
    fig,ax=plt.subplots(figsize=(7.2,4.2));fgrid=np.r_[0,np.logspace(-5,0,400)]
    for m,k,ls in [('Linear SVM',1,'-'),('Linear SVM',5,'--'),('MLP',1,':'),('MLP',5,'-.'),('SIC',1,'-'),('SIC',5,'--')]:
        curves=[]
        for tag,z in score_items(k):
            fp,tp,_=roc_curve(z['y'],z['s_'+m.replace(' ','_')],drop_intermediate=False)
            frame=pd.DataFrame({'fp':fp,'tp':tp}).groupby('fp').tp.max()
            curves.append(1-np.interp(fgrid,frame.index,frame.to_numpy()))
        ax.plot(fgrid[1:]*100,np.mean(curves,0)[1:]*100,ls=ls,lw=1.5,label=f'{m}, K = {k}')
    ax.set_xscale('log');ax.set(xlim=(.001,100),ylim=(-1,100),xlabel='False acceptance rate (%)',ylabel='Mean false rejection rate (%)');ax.legend(frameon=False,fontsize=9,ncols=2);tidy(ax);save(fig,'figureS1_operating_curves')
    matched_count_figure()
    print('Created eight PDF figures.')
if __name__=='__main__':main()

"""Create vector figures from the verified six-budget scoring tables."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_feedback import BUDGETS, BRIER, ECE, SUBJECTS, BENCHMARKS, COUNTS, WEIGHTS, ALC_DISPLAY

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'figures'
OUT.mkdir(parents=True,exist_ok=True)
BLUE='#21618C'
ORANGE='#B66124'
GRAY='#5A6470'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#69717A','axes.linewidth':.7,'grid.color':'#E1E5E8','grid.linewidth':.6,'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})

def save(fig,name):
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.04)
 fig.savefig(OUT/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.04)
 plt.close(fig)

x=np.arange(6)
mb=BRIER.mean(0);me=ECE.mean(0)
fig,axes=plt.subplots(1,2,figsize=(6.5,2.5))
for ax,values,color,title,ymax in [(axes[0],mb,BLUE,'A. Prediction error',.29),(axes[1],me,ORANGE,'B. Calibration error',.32)]:
 ax.set_title(title,loc='left',weight='bold',pad=10)
 ax.plot(x,values,marker='o',markersize=4,color=color,lw=2,label='Pair macro mean')
 ax.set_xticks(x,[str(v) for v in BUDGETS]);ax.set_xlim(-.15,5.15);ax.set_ylim(0,ymax)
 ax.set_xlabel('Revealed labels (log budget spacing)')
 ax.set_ylabel('Brier score' if ax is axes[0] else 'Expected calibration error')
 ax.grid(axis='y');ax.set_axisbelow(True)
 for j in [0,5]: ax.annotate(f'{values[j]:.4f}',(x[j],values[j]),xytext=(2,8),textcoords='offset points',ha='left' if j==0 else 'right',fontsize=8,color=color,weight='bold')
axes[0].axhline(.25,color=GRAY,ls='--',lw=1)
axes[0].text(2.7,.256,'Fixed p = 0.5',fontsize=8,color=GRAY)
axes[0].axhline(mb[0],color=GRAY,ls=':',lw=1)
axes[0].text(1.5,mb[0]+.007,'Frozen zero-label reference',fontsize=7.5,color=GRAY)
axes[0].fill_between(x,mb,0,color=BLUE,alpha=.08)
axes[0].text(.18,.034,'Brier ALC = 0.160822',color=BLUE,weight='bold',fontsize=9)
fig.tight_layout(w_pad=1.6)
save(fig,'learning_curves')

fig,axes=plt.subplots(3,3,figsize=(6.5,4.65),sharex=True,sharey=True)
for i,ax in enumerate(axes.flat):
 ax.plot(x,BRIER[i],color=BLUE,marker='o',markersize=3,lw=1.6)
 ax.scatter([0],[BRIER[i,0]],s=20,facecolor='white',edgecolor=BLUE,zorder=4)
 ax.axhline(.25,color=GRAY,ls=':',lw=.65)
 ax.set_title(f'S{SUBJECTS[i]} / B{BENCHMARKS[i]}',loc='left',fontsize=8.0,weight='bold',pad=4)
 ax.text(.98,.94,f'm = {COUNTS[i]}   ALC {ALC_DISPLAY[i]:.3f}',transform=ax.transAxes,ha='right',va='top',fontsize=7,color=GRAY)
 ax.set_ylim(0,.305);ax.set_xlim(-.15,5.15);ax.set_xticks(x,[str(v) for v in BUDGETS]);ax.set_yticks([0,.1,.2,.3]);ax.grid(axis='y')
 if i%3==0:ax.set_ylabel('Brier score')
 if i>=6:ax.set_xlabel('Revealed labels')
fig.tight_layout(w_pad=1,h_pad=1.1)
save(fig,'pair_learning_curves')

fig,axes=plt.subplots(1,2,figsize=(6.5,3.0),gridspec_kw={'width_ratios':[1.15,1]})
order=np.argsort(BRIER[:,0]-BRIER[:,-1])
gain=BRIER[:,0]-BRIER[:,-1]
axes[0].barh(np.arange(9),gain[order],color=BLUE,height=.6)
axes[0].set_yticks(np.arange(9),['S'+SUBJECTS[i] for i in order]);axes[0].set_xlim(0,.19)
for j,i in enumerate(order):axes[0].text(gain[i]+.003,j,f'{gain[i]:.4f}',va='center',fontsize=7.5)
axes[0].set_xlabel('Brier reduction: budget 0 minus budget 31')
axes[0].set_title('A. Brier gain by pair',loc='left',weight='bold',fontsize=9.5)
axes[0].grid(axis='x');axes[0].set_axisbelow(True)
ax=axes[1]
for i in range(9):
 ax.annotate('',xy=(ECE[i,-1],BRIER[i,-1]),xytext=(ECE[i,0],BRIER[i,0]),arrowprops=dict(arrowstyle='->',color=GRAY,lw=.8,alpha=.65))
ax.scatter(ECE[:,0],BRIER[:,0],facecolors='white',edgecolors=ORANGE,s=27,label='Budget 0',zorder=3)
ax.scatter(ECE[:,-1],BRIER[:,-1],c=BLUE,s=24,label='Budget 31',zorder=4)
for i in [4,7,6]:
 dx,dy={4:(7,-12),7:(7,4),6:(7,0)}[i]
 ax.annotate('S'+SUBJECTS[i],(ECE[i,-1],BRIER[i,-1]),xytext=(dx,dy),textcoords='offset points',fontsize=6.8)
ax.set_xlim(0,.45);ax.set_ylim(0,.31);ax.set_xlabel('Expected calibration error');ax.set_ylabel('Brier score')
ax.set_title('B. Calibration versus error',loc='left',weight='bold',fontsize=9.5)
ax.legend(loc='lower right',frameon=False);ax.grid(alpha=.8);ax.set_axisbelow(True)
fig.tight_layout(w_pad=1.5)
save(fig,'adaptation_calibration')

fig,ax=plt.subplots(figsize=(6.5,2.9))
im=ax.imshow(ECE,aspect='auto',vmin=0,vmax=.42,cmap='Blues')
ax.set_xticks(x,[str(v) for v in BUDGETS]);ax.set_yticks(range(9),['S'+s for s in SUBJECTS])
ax.set_xlabel('Revealed labels');ax.tick_params(length=0)
for i in range(9):
 for j in range(6):
  ax.text(j,i,f'{ECE[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if ECE[i,j]>.245 else '#182633')
ax.set_title('Expected calibration error by pair and budget',loc='left',weight='bold',pad=9)
c=fig.colorbar(im,ax=ax,fraction=.035,pad=.025);c.set_label('ECE');c.outline.set_visible(False)
fig.tight_layout()
save(fig,'ece_heatmap')
print('Figure outputs:',*[str(p.relative_to(ROOT)) for p in sorted(OUT.glob('*.pdf'))],sep='\n')

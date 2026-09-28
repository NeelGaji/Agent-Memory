#!/usr/bin/env python3
"""Recreate StateMem's 5 publication figures from frozen aggregate figure_data.json.

Requirements: matplotlib >=3.6 and its normal dependencies. Outputs PNG, PDF, SVG.
No source data downloads, VLM inference or parameter fitting are performed here.
"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
D = json.loads((HERE/'figure_data.json').read_text())
C = {'Latest':'#52657B','Recency-Calibrated':'#D39A31','StateMem-Calibrated':'#0C8275',
     'ink':'#1A2838','muted':'#596B79','rule':'#D6E0E6','paper':'#FFFFFF','light':'#F3F7F9',
     'orange_light':'#FFF4E2','teal_light':'#E4F4EF','blue_light':'#EBF1F8'}
plt.rcParams.update({
 'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':15,'axes.titleweight':'bold',
 'axes.labelsize':10,'xtick.labelsize':9,'ytick.labelsize':9,
 'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
 'axes.spines.right':False,'axes.spines.top':False,'axes.titlepad':15,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
})

def save(fig,stem):
 for ext,opts in [('png',{'dpi':300}),('pdf',{}),('svg',{})]:
  fig.savefig(HERE/f'{stem}.{ext}',bbox_inches='tight',pad_inches=.2,**opts)
 plt.close(fig)

def heading(ax,title,subtitle=None):
 ax.set_title(title,loc='left',color=C['ink'])
 if subtitle:
  ax.text(0,1.025,subtitle,transform=ax.transAxes,ha='left',va='bottom',fontsize=9,
          color=C['muted'])

def fig01_architecture():
 fig,ax=plt.subplots(figsize=(13.6,6.15))
 ax.set_xlim(0,14.1);ax.set_ylim(0,6.4);ax.axis('off')
 fig.text(.03,.965,'StateMem: separate evidence from current belief',fontsize=18,
          weight='bold',color=C['ink'],ha='left',va='top')
 fig.text(.03,.905,'Proposed system; solid = evaluated offline core, dashed = future online integration',
          fontsize=10,color=C['muted'],ha='left',va='top')
 def block(x,y,w,h,title,body,edge,face,borderstyle='solid',small=9.4):
  ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.18,rounding_size=0.16',
               facecolor=face,edgecolor=edge,lw=1.8,linestyle=borderstyle))
  ax.text(x+.22,y+h-.32,title,color=C['ink'],weight='bold',fontsize=11.1,ha='left',va='top')
  ax.text(x+.22,y+h-.90,body,color=C['muted'],fontsize=small,ha='left',va='top',linespacing=1.45)
 def arrow(x,y,xx,yy,color=C['muted'],dashed=False,mutation=16):
  ax.add_patch(FancyArrowPatch((x,y),(xx,yy),arrowstyle='-|>',mutation_scale=mutation,
                              lw=1.65,color=color,linestyle='--' if dashed else '-'))
 block(.3,2.6,2.25,1.52,'RGB / VLM','Timestamped RGB\nA/B neutral judgments',C['Latest'],C['blue_light'])
 block(3.08,2.6,2.55,1.52,'Episodic evidence','Provenance + time\nSupport + contradictions',C['Latest'],C['blue_light'])
 block(6.17,2.6,3.02,1.52,'Temporal reconciler','Time-gap prediction\nReliability-based update',C['StateMem-Calibrated'],C['teal_light'])
 block(9.74,2.6,3.35,1.52,'World belief memory','P(current START / GOAL)\nEvidence-linked estimate',C['StateMem-Calibrated'],C['teal_light'])
 arrow(2.64,3.36,2.94,3.36)
 arrow(5.71,3.36,6.02,3.36)
 arrow(9.27,3.36,9.59,3.36,color=C['StateMem-Calibrated'])
 block(6.23,4.79,2.85,1.05,'Train-only calibration','Judgment-specific reliability',C['Recency-Calibrated'],C['orange_light'],small=8.7)
 arrow(7.66,4.67,7.66,4.27,color=C['Recency-Calibrated'])
 block(9.93,.30,2.98,1.40,'Query / planner','Offline QA done\nOnline Habitat pending',C['Latest'],C['light'],borderstyle='--',small=9)
 arrow(11.31,2.42,11.31,1.89,dashed=True)
 ax.text(1.49,2.24,'Implemented: counterbalanced local RGB perception',ha='center',va='top',fontsize=8.8,color=C['muted'])
 ax.text(7.1,2.03,'Implemented: frozen state filtering on selected frames',ha='center',va='top',fontsize=9,color=C['StateMem-Calibrated'])
 fig.text(.05,.057,'Evaluated states: START / GOAL receptacles; not an unrestricted world model',
         fontsize=9,color=C['muted'],ha='left',va='bottom')
 save(fig,'fig01_architecture')

def fig02_full_accuracy():
 methods=['Latest','Recency-Calibrated','StateMem-Calibrated']
 pretty={'Latest':'Latest observation','Recency-Calibrated':'Calibrated recency','StateMem-Calibrated':'Calibrated StateMem'}
 vals=[100*D['main']['methods'][m]['accuracy'] for m in methods]
 fig,ax=plt.subplots(figsize=(9.4,4.9));fig.subplots_adjust(left=.25,right=.95,top=.73,bottom=.21)
 for y,(m,v) in enumerate(zip(methods,vals)):
  ax.hlines(y,68,v,lw=7,color=C[m],alpha=.85)
  ax.plot(v,y,'o',color=C[m],markersize=9,zorder=5)
  ax.text(v+.09,y,f'{v:.2f}%',va='center',ha='left',fontweight='bold',color=C['ink'])
 ax.set_yticks(range(len(methods)),[pretty[m] for m in methods])
 ax.invert_yaxis();ax.set_xlim(68,74);ax.set_xticks([68,69,70,71,72,73,74]);ax.set_xlabel('Overall state accuracy (%) - zoomed x-axis',labelpad=13)
 ax.grid(axis='x',alpha=.22);ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color(C['rule']);ax.tick_params(axis='y',length=0,pad=11)
 fig.suptitle('Full RGB stream: accuracy differences are small',x=.08,y=.98,ha='left',fontsize=17,fontweight='bold',color=C['ink'])
 fig.text(.08,.875,'FindingDory selected-frame protocol | 67 validation episodes | 148 transitions | 963 state-evaluation steps',
          ha='left',color=C['muted'],fontsize=9.3)
 fig.text(.08,.055,'Paired episode-cluster 95% CIs: StateMem - Latest [+0.73 pp; -0.32 to +1.98];  '
          'StateMem - Recency [+0.52 pp; -0.55 to +1.79].',ha='left',color=C['muted'],fontsize=8.8)
 fig.text(.08,.015,'Both CIs cross zero. Oracle annotation-assisted RGB frame selection; not online Habitat success.',
          ha='left',color=C['muted'],fontsize=8.6)
 save(fig,'fig02_full_accuracy')

def fig03_sparsity():
 regimes=['sparse_33','sparse_50','sparse_75','full'];x=[33,50,75,100]
 d={(r['regime'],r['method']):float(r['accuracy'])*100 for r in D['stress']}
 fig,ax=plt.subplots(figsize=(9.8,5.8));fig.subplots_adjust(left=.105,right=.86,top=.77,bottom=.22)
 label={'Latest':'Latest','Recency-Calibrated':'Recency','StateMem-Calibrated':'StateMem'}
 for m in ['Latest','Recency-Calibrated','StateMem-Calibrated']:
  ys=[d[(reg,m)] for reg in regimes]
  ax.plot(x,ys,'-o',lw=2.65,markersize=7.3,label=label[m],color=C[m])
  sparse_label_shift = {'Latest':(-3,-13),'Recency-Calibrated':(0,2),'StateMem-Calibrated':(3,13)}[m]
  ax.annotate(f'{ys[0]:.2f}%',(x[0],ys[0]),xytext=sparse_label_shift,textcoords='offset points',ha='center',fontsize=8.7,color=C[m],fontweight='bold')
  if m == 'StateMem-Calibrated':
   ax.annotate(f'{ys[-1]:.2f}%',(x[-1],ys[-1]),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8.7,color=C[m],fontweight='bold')
 ax.set_xticks(x,[f'{k}%' for k in x]);ax.set_xlim(28,105);ax.set_ylim(51,79);ax.set_xlabel('Retained observations after first START anchor');ax.set_ylabel('Overall state accuracy (%)')
 ax.grid(axis='y',alpha=.22);ax.spines['left'].set_color(C['rule']);ax.spines['bottom'].set_color(C['rule'])
 ax.legend(loc='lower right',frameon=False,ncol=1,bbox_to_anchor=(1.22,.1),fontsize=9.5)
 fig.suptitle('Temporal robustness as observations become sparse',x=.07,y=.98,ha='left',fontsize=17,fontweight='bold',color=C['ink'])
 fig.text(.07,.89,'Frozen memory parameters; 20 deterministic masks at each sparse level; same 67-episode validation split',fontsize=9.1,color=C['muted'])
 fig.text(.07,.055,'StateMem - Latest: +0.73 pp (full)  |  +0.94 pp (75%)  |  +2.26 pp (50%)  |  +3.57 pp (33%).',fontsize=9.3,color=C['ink'])
 fig.text(.07,.018,'Post-hoc exploratory stress replay; source observations were selected using FindingDory annotations.',fontsize=8.7,color=C['muted'])
 save(fig,'fig03_sparsity')

def fig04_world_qa():
 measures=['object_state_accuracy','world_exact','count_exact','goal_set_f1']
 labels=['Object state','Whole world\nexact','GOAL count\nexact','GOAL-set F1']
 fig,ax=plt.subplots(figsize=(11.1,5.8));fig.subplots_adjust(left=.095,right=.95,top=.74,bottom=.2)
 offsets=[-.255,0,.255];width=.22
 for j,m in enumerate(['Latest','Recency-Calibrated','StateMem-Calibrated']):
  vals=[100*D['qa_summary'][m][q] for q in measures]
  pos=[i+offsets[j] for i in range(len(measures))]
  bars=ax.bar(pos,vals,width=width,color=C[m],label={'Latest':'Latest','Recency-Calibrated':'Recency','StateMem-Calibrated':'StateMem'}[m],zorder=3)
  for b,v in zip(bars,vals):
   ax.text(b.get_x()+b.get_width()/2,v+1.25,f'{v:.1f}',ha='center',va='bottom',fontsize=8.4,color=C['ink'])
 ax.set_xticks(range(len(measures)),labels);ax.set_ylim(0,95);ax.set_yticks(range(0,101,20));ax.set_ylabel('Score (%)');ax.grid(axis='y',alpha=.21,zorder=0)
 ax.legend(loc='upper right',bbox_to_anchor=(1.0,1.15),frameon=False,ncol=3,fontsize=9.5)
 ax.spines['left'].set_color(C['rule']);ax.spines['bottom'].set_color(C['rule'])
 fig.suptitle('Multi-object world-state QA (offline)',x=.07,y=.98,ha='left',fontsize=17,fontweight='bold',color=C['ink'])
 fig.text(.07,.885,'335 shared query times across 44 validation episodes; identical frozen RGB evidence and memory parameters',fontsize=9.1,color=C['muted'])
 fig.text(.07,.047,'All reported paired 95% CIs for StateMem vs Latest/Recency cross zero; no established downstream win.',fontsize=9.1,color=C['muted'])
 fig.text(.07,.012,'Task is structured START/GOAL state QA, not embodied navigation or official FindingDory task success.',fontsize=8.5,color=C['muted'])
 save(fig,'fig04_world_qa')

def fig05_effects():
 rgs=['full','sparse_75','sparse_50','sparse_33','delay_goal_1','delay_goal_2','delay_goal_3']
 rlabels=['Full evidence','Retain 75%','Retain 50%','Retain 33%','Delay 1 GOAL obs.','Delay 2 GOAL obs.','Delay 3 GOAL obs.']
 bs={(r['regime'],r['comparison']):r for r in D['stress_bootstrap'] if r['metric']=='overall_accuracy'}
 fig,ax=plt.subplots(figsize=(10.4,6.3));fig.subplots_adjust(left=.235,right=.94,top=.82,bottom=.16)
 for j,base in enumerate(['Latest','Recency-Calibrated']):
  key=f'StateMem-Calibrated - {base}';off=(-.125 if j==0 else .125)
  label='vs Latest' if j==0 else 'vs Recency'
  for i,rg in enumerate(rgs):
   r=bs[(rg,key)]
   v,lo,hi=[100*float(r[k]) for k in ['difference','ci95_low','ci95_high']]
   yy=i+off
   ax.hlines(yy,lo,hi,lw=2.25,color=C[base],zorder=3)
   ax.plot([lo,lo],[yy-.052,yy+.052],lw=1.5,color=C[base]);ax.plot([hi,hi],[yy-.052,yy+.052],lw=1.5,color=C[base])
   ax.plot(v,yy,'o',markersize=6.3,color=C[base],zorder=4)
 ax.axvline(0,color=C['ink'],lw=1,alpha=.5,ls='--')
 ax.set_yticks(range(len(rgs)),rlabels);ax.invert_yaxis();ax.set_xlim(-1.7,5.8);ax.set_xticks(range(-1,6));ax.set_xlabel('StateMem overall accuracy difference (percentage points)');ax.grid(axis='x',alpha=.17)
 ax.tick_params(axis='y',length=0,pad=8)
 ax.spines['left'].set_visible(False);ax.spines['bottom'].set_color(C['rule'])
 handles=[Line2D([0],[0],marker='o',linestyle='-',color=C['Latest'],label='vs Latest'),
          Line2D([0],[0],marker='o',linestyle='-',color=C['Recency-Calibrated'],label='vs Recency')]
 ax.legend(handles=handles,loc='lower right',frameon=False)
 fig.suptitle('Paired accuracy effects under evidence loss',x=.07,y=.98,ha='left',fontsize=17,fontweight='bold',color=C['ink'])
 fig.text(.07,.905,'StateMem minus baseline; episode-cluster 95% bootstrap confidence intervals (20,000 resamples)',fontsize=9.2,color=C['muted'])
 fig.text(.07,.044,'Full-stream intervals cross zero. Sparse/delayed cases are exploratory on the same previously examined split.',fontsize=8.8,color=C['muted'])
 fig.text(.07,.014,'Effects are for selected-frame binary state tracking, not end-to-end robot success.',fontsize=8.7,color=C['muted'])
 save(fig,'fig05_paired_effects')

if __name__=='__main__':
 fig01_architecture();fig02_full_accuracy();fig03_sparsity();fig04_world_qa();fig05_effects()
 print('Created 5 figures x PNG/PDF/SVG in',HERE)

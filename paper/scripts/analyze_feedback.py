"""Reproduce descriptive results from the supplied scoring output, submission 1033935."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data'
OUT.mkdir(parents=True, exist_ok=True)
BUDGETS = np.array([0, 1, 3, 7, 15, 31])
WEIGHTS = np.array([.1, .2, .2, .2, .2, .1])
SUBJECTS = ['198432','243003','265079','279917','341270','604683','912438','941033','968598']
BENCHMARKS = ['800730','800730','869474','213669','893575','342307','869474','119137','830060']
COUNTS = np.array([50,54,56,58,44,60,56,44,74])
BRIER = np.array([
 [.197036,.141744,.095946,.067952,.061828,.051231],
 [.195231,.138863,.091878,.061027,.053679,.040327],
 [.203901,.176585,.194296,.153350,.151432,.143456],
 [.202827,.160266,.123605,.101692,.097504,.098363],
 [.241865,.235420,.241557,.261820,.237886,.237775],
 [.206015,.267988,.266198,.204948,.154412,.135660],
 [.172442,.121322,.074291,.038093,.019099,.010232],
 [.283088,.249099,.237301,.250187,.237347,.237130],
 [.206145,.191146,.183869,.183852,.185207,.185867],
])
ECE = np.array([
 [.386220,.306276,.218805,.140396,.116296,.052273],
 [.399063,.320803,.236614,.158748,.133699,.067694],
 [.246940,.183921,.226751,.104399,.094978,.068300],
 [.332160,.260306,.176277,.095509,.070140,.076048],
 [.079923,.014334,.080866,.165216,.052031,.050887],
 [.273533,.369661,.367233,.271586,.154354,.096089],
 [.409337,.341324,.263773,.183032,.120792,.075945],
 [.214510,.109379,.015256,.114241,.015630,.049010],
 [.149906,.086268,.010770,.009931,.063112,.045460],
])
ALC_DISPLAY = np.array([.098321,.092645,.169868,.126733,.243301,.212877,.068828,.246809,.188016])
ECE_DISPLAY = np.array([.200204,.216648,.153534,.161267,.075570,.269529,.230312,.077253,.053553])

def main():
    rows=[]
    for i, s in enumerate(SUBJECTS):
     for j,b in enumerate(BUDGETS):
      rows.append(dict(submission_id=1033935,subject_id='subject_'+s,benchmark_id='benchmark_'+BENCHMARKS[i],unique_evaluation_pairs=int(COUNTS[i]),label_budget=int(b),brier=float(BRIER[i,j]),ece=float(ECE[i,j])))
    with (OUT/'feedback_by_budget.csv').open('w',newline='') as f:
     w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    
    pair_alc=BRIER@WEIGHTS
    pair_ece=ECE@WEIGHTS
    mean_brier=BRIER.mean(axis=0)
    mean_ece=ECE.mean(axis=0)
    gain=BRIER[:,0]-BRIER[:,-1]
    metrics=dict(
     submission_id=1033935,
     source='User-supplied Codabench scoring HTML and submission screenshot; decimal values rounded to six places.',
     displayed_brier_alc=.160822,
     label_budgets=BUDGETS.tolist(),
     alc_weights=WEIGHTS.tolist(),
     subject_benchmark_pairs=len(SUBJECTS),
     distinct_subjects=len(set(SUBJECTS)),
     distinct_benchmarks=len(set(BENCHMARKS)),
     unique_evaluation_pairs=int(COUNTS.sum()),
     macro_brier_by_budget=mean_brier.tolist(),
     macro_ece_by_budget=mean_ece.tolist(),
     macro_alc_from_budget_rows=float(mean_brier@WEIGHTS),
     macro_alc_from_summary_rows=float(ALC_DISPLAY.mean()),
     macro_ece_alc_from_budget_rows=float(mean_ece@WEIGHTS),
     macro_ece_alc_from_summary_rows=float(ECE_DISPLAY.mean()),
     fixed_half_probability_brier=.25,
     relative_alc_reduction_vs_fixed_half=float(1-(mean_brier@WEIGHTS)/.25),
     zero_to_31_absolute_reduction=float(mean_brier[0]-mean_brier[-1]),
     zero_to_31_relative_reduction=float(1-mean_brier[-1]/mean_brier[0]),
     zero_to_31_ece_relative_reduction=float(1-mean_ece[-1]/mean_ece[0]),
     frozen_zero_budget_reference_alc=float(mean_brier[0]),
     adaptation_alc_gain_vs_frozen_zero=float(mean_brier[0]-mean_brier@WEIGHTS),
     adaptation_relative_alc_gain_vs_frozen_zero=float(1-(mean_brier@WEIGHTS)/mean_brier[0]),
     pair_alc_reconstruction_max_abs_error=float(np.max(np.abs(pair_alc-ALC_DISPLAY))),
     pair_ece_reconstruction_max_abs_error=float(np.max(np.abs(pair_ece-ECE_DISPLAY))),
     count_weighted_alc_diagnostic=float(np.average(ALC_DISPLAY,weights=COUNTS)),
     pairs_improving_at_31=int(np.sum(gain>0)),
     pairs_with_any_brier_increase=int(np.sum(np.any(np.diff(BRIER,axis=1)>0,axis=1))),
     monotone_pair_count=int(np.sum(np.all(np.diff(BRIER,axis=1)<=0,axis=1))),
     checkpoint_increases=int(np.sum(np.diff(BRIER,axis=1)>0)),
     checkpoint_transitions=int(np.diff(BRIER,axis=1).size),
     lowest_pair_alc=float(ALC_DISPLAY.min()),
     highest_pair_alc=float(ALC_DISPLAY.max()),
     pairs=[dict(subject_id='subject_'+s,benchmark_id='benchmark_'+BENCHMARKS[i],unique_evaluation_pairs=int(COUNTS[i]),brier_alc=float(ALC_DISPLAY[i]),brier_alc_reconstructed=float(pair_alc[i]),ece_alc=float(ECE_DISPLAY[i]),brier_zero=float(BRIER[i,0]),brier_31=float(BRIER[i,-1]),absolute_gain=float(gain[i]),relative_gain=float(gain[i]/BRIER[i,0])) for i,s in enumerate(SUBJECTS)]
    )
    assert abs(metrics['macro_alc_from_budget_rows']-.160822)<.0000005
    assert metrics['pair_alc_reconstruction_max_abs_error']<.000001
    assert metrics['pair_ece_reconstruction_max_abs_error']<.000001
    (OUT/'computed_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    with (OUT/'feedback_pair_summary.csv').open('w',newline='') as f:
     w=csv.DictWriter(f,fieldnames=list(metrics['pairs'][0]));w.writeheader();w.writerows(metrics['pairs'])
    print(json.dumps({k:v for k,v in metrics.items() if k!='pairs'},indent=2))


if __name__ == "__main__":
    main()

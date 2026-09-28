#!/usr/bin/env python3
import argparse, csv, json, math, random
from dataclasses import dataclass
from pathlib import Path

STATES = ["counter", "sink", "table"]

@dataclass
class Observation:
    t: int
    true_state: str
    observed_state: str
    confidence: float
    corrupted: bool

def sample_confidence(correct, mode, rng):
    if mode == "reliable":
        return rng.uniform(0.70, 0.98) if correct else rng.uniform(0.10, 0.55)
    if mode == "overlapping":
        return rng.uniform(0.55, 0.95) if correct else rng.uniform(0.35, 0.85)
    if mode == "misleading":
        return rng.uniform(0.45, 0.85) if correct else rng.uniform(0.55, 0.95)
    raise ValueError(mode)

def generate_episode(steps, corruption_rate, switch_prob, confidence_mode, rng):
    true_state = rng.choice(STATES)
    out = []
    for t in range(steps):
        if t > 0 and rng.random() < switch_prob:
            true_state = rng.choice([s for s in STATES if s != true_state])
        corrupted = rng.random() < corruption_rate
        observed = rng.choice([s for s in STATES if s != true_state]) if corrupted else true_state
        conf = sample_confidence(not corrupted, confidence_mode, rng)
        out.append(Observation(t, true_state, observed, conf, corrupted))
    return out

def generate_dataset(episodes, steps, corruption_rate, switch_prob, confidence_mode, seed):
    rng = random.Random(seed)
    return [generate_episode(steps, corruption_rate, switch_prob, confidence_mode, rng)
            for _ in range(episodes)]

class HistogramCalibrator:
    def __init__(self, bins=10, alpha=2.0, beta=2.0):
        self.bins, self.alpha, self.beta = bins, alpha, beta
        self.correct = [0]*bins
        self.total = [0]*bins
        self.global_correct = 0
        self.global_total = 0

    def _bin(self, c):
        c = max(0.0, min(0.999999, c))
        return min(self.bins-1, int(c*self.bins))

    def fit_observation(self, c, is_correct):
        i = self._bin(c)
        self.total[i] += 1
        self.correct[i] += int(is_correct)
        self.global_total += 1
        self.global_correct += int(is_correct)

    def predict(self, c):
        i = self._bin(c)
        if self.total[i] == 0:
            if self.global_total == 0:
                return 0.75
            return (self.global_correct+self.alpha)/(self.global_total+self.alpha+self.beta)
        return (self.correct[i]+self.alpha)/(self.total[i]+self.alpha+self.beta)

    def table(self):
        rows = []
        for i in range(self.bins):
            center = (i+0.5)/self.bins
            rows.append((center, self.predict(center), self.total[i]))
        return rows

def fit_calibrators_by_mode(conditions, bins):
    modes = sorted({k[0] for k in conditions})
    out = {m: HistogramCalibrator(bins=bins) for m in modes}
    for (mode, _, _), dataset in conditions.items():
        cal = out[mode]
        for ep in dataset:
            for o in ep:
                cal.fit_observation(o.confidence, o.observed_state == o.true_state)
    return out

class RecencyTracker:
    def __init__(self, decay, use_confidence):
        self.factor = math.exp(-decay)
        self.use_confidence = use_confidence
        self.scores = {s: 0.0 for s in STATES}

    def update(self, state, confidence):
        for s in STATES:
            self.scores[s] *= self.factor
        self.scores[state] += confidence if self.use_confidence else 1.0
        return max(self.scores, key=self.scores.get)

class StateMem:
    def __init__(self, stay_probability, mode, fixed_reliability=0.75,
                 calibrator=None, adaptive_strength=0.30):
        self.base_stay = stay_probability
        self.mode = mode
        self.fixed_reliability = fixed_reliability
        self.calibrator = calibrator
        self.adaptive_strength = adaptive_strength
        self.belief = {s: 1/len(STATES) for s in STATES}

    def _reliability(self, raw):
        if self.mode == "raw": return raw
        if self.mode == "noconf": return self.fixed_reliability
        if self.mode in ("calibrated", "adaptive"):
            return self.calibrator.predict(raw)
        raise ValueError(self.mode)

    def _effective_stay(self, obs_state, reliability):
        if self.mode != "adaptive":
            return self.base_stay
        current = max(self.belief, key=self.belief.get)
        if obs_state == current:
            return min(0.98, self.base_stay + 0.08*max(0.0, reliability-0.5)*2)
        signed = (reliability-0.5)*2
        if signed >= 0:
            stay = self.base_stay - self.adaptive_strength*signed
        else:
            stay = self.base_stay + 0.20*(-signed)
        return max(0.35, min(0.98, stay))

    def update(self, obs_state, raw_conf):
        r = max(0.01, min(0.99, self._reliability(raw_conf)))
        stay = self._effective_stay(obs_state, r)
        n = len(STATES)
        ch = (1-stay)/(n-1)
        prior = {s: 0.0 for s in STATES}
        for prev, pp in self.belief.items():
            for nxt in STATES:
                prior[nxt] += pp*(stay if nxt == prev else ch)
        like = {s: (r if s == obs_state else (1-r)/(n-1)) for s in STATES}
        post = {s: prior[s]*like[s] for s in STATES}
        z = sum(post.values())
        self.belief = {s: post[s]/z for s in STATES}
        return dict(self.belief)

    def predict(self):
        return max(self.belief, key=self.belief.get)

METHODS = ["latest","recency","recency_rawconf","statemem_raw",
           "statemem_noconf","statemem_calibrated","statemem_adaptive"]

def transition_lags(truths, preds):
    idxs = [i for i in range(1,len(truths)) if truths[i] != truths[i-1]]
    lags, misses = [], 0
    for k,start in enumerate(idxs):
        target = truths[start]
        end = idxs[k+1] if k+1 < len(idxs) else len(truths)
        for j in range(start,end):
            if preds[j] == target:
                lags.append(j-start); break
        else:
            misses += 1
    return (sum(lags)/len(lags) if lags else float("nan"), len(lags), misses)

def evaluate_dataset(dataset, decay, calibrator, params):
    correct = {m:0 for m in METHODS}
    bad_correct = {m:0 for m in METHODS}
    total = bad_total = 0
    brier = {m:0.0 for m in ["statemem_raw","statemem_noconf",
                              "statemem_calibrated","statemem_adaptive"]}
    lag_sum = {m:0.0 for m in METHODS}
    lag_n = {m:0 for m in METHODS}
    lag_miss = {m:0 for m in METHODS}

    for ep in dataset:
        rec = RecencyTracker(decay, False)
        recc = RecencyTracker(decay, True)
        raw = StateMem(params["raw_stay"], "raw")
        no = StateMem(params["noconf_stay"], "noconf",
                      fixed_reliability=params["noconf_reliability"])
        cal = StateMem(params["calibrated_stay"], "calibrated", calibrator=calibrator)
        ada = StateMem(params["adaptive_stay"], "adaptive", calibrator=calibrator,
                       adaptive_strength=params["adaptive_strength"])
        truths = []
        preds = {m:[] for m in METHODS}

        for o in ep:
            br = raw.update(o.observed_state, o.confidence)
            bn = no.update(o.observed_state, o.confidence)
            bc = cal.update(o.observed_state, o.confidence)
            ba = ada.update(o.observed_state, o.confidence)
            pm = {
                "latest": o.observed_state,
                "recency": rec.update(o.observed_state, o.confidence),
                "recency_rawconf": recc.update(o.observed_state, o.confidence),
                "statemem_raw": raw.predict(),
                "statemem_noconf": no.predict(),
                "statemem_calibrated": cal.predict(),
                "statemem_adaptive": ada.predict(),
            }
            truths.append(o.true_state)
            for m,pred in pm.items():
                preds[m].append(pred)
                correct[m] += pred == o.true_state
                if o.corrupted: bad_correct[m] += pred == o.true_state
            for name,belief in [("statemem_raw",br),("statemem_noconf",bn),
                                ("statemem_calibrated",bc),("statemem_adaptive",ba)]:
                for s in STATES:
                    y = 1.0 if s == o.true_state else 0.0
                    brier[name] += (belief[s]-y)**2
            total += 1
            if o.corrupted: bad_total += 1

        for m in METHODS:
            lag,n,miss = transition_lags(truths,preds[m])
            if n:
                lag_sum[m] += lag*n
                lag_n[m] += n
            lag_miss[m] += miss

    out = {"n":total}
    for m in METHODS:
        out[m+"_acc"] = correct[m]/total
        out[m+"_noise_acc"] = bad_correct[m]/bad_total if bad_total else float("nan")
        out[m+"_transition_lag"] = lag_sum[m]/lag_n[m] if lag_n[m] else float("nan")
    for m,v in brier.items():
        out[m+"_brier"] = v/total
    return out

def mean_metric(rows,key):
    vals=[r[key] for r in rows if not (isinstance(r[key],float) and math.isnan(r[key]))]
    return sum(vals)/len(vals)

def build_conditions(episodes,steps,modes,switches,corruptions,seed):
    out={}; idx=0
    for mode in modes:
        for sw in switches:
            for cr in corruptions:
                out[(mode,sw,cr)] = generate_dataset(episodes,steps,cr,sw,mode,seed+idx*1009)
                idx+=1
    return out

def eval_for_tuning(conditions,cals,decay,params,key):
    rows=[]
    for (mode,_,_),ds in conditions.items():
        rows.append(evaluate_dataset(ds,decay,cals[mode],params))
    return mean_metric(rows,key)

def tune(val,cals,decay,stays,no_rels,strengths):
    base={"raw_stay":0.75,"noconf_stay":0.75,"noconf_reliability":0.75,
          "calibrated_stay":0.75,"adaptive_stay":0.75,"adaptive_strength":0.30}

    best=(-1,None)
    for s in stays:
        p=dict(base); p["raw_stay"]=s
        sc=eval_for_tuning(val,cals,decay,p,"statemem_raw_acc")
        if sc>best[0]: best=(sc,s)
    raw_stay=best[1]

    best=(-1,None)
    for s in stays:
        for r in no_rels:
            p=dict(base); p["noconf_stay"]=s; p["noconf_reliability"]=r
            sc=eval_for_tuning(val,cals,decay,p,"statemem_noconf_acc")
            if sc>best[0]: best=(sc,(s,r))
    no_stay,no_rel=best[1]

    best=(-1,None)
    for s in stays:
        p=dict(base); p["calibrated_stay"]=s
        sc=eval_for_tuning(val,cals,decay,p,"statemem_calibrated_acc")
        if sc>best[0]: best=(sc,s)
    cal_stay=best[1]

    best=(-1,None)
    for s in stays:
        for a in strengths:
            p=dict(base); p["adaptive_stay"]=s; p["adaptive_strength"]=a
            sc=eval_for_tuning(val,cals,decay,p,"statemem_adaptive_acc")
            if sc>best[0]: best=(sc,(s,a))
    ad_stay,ad_strength=best[1]

    return {"raw_stay":raw_stay,"noconf_stay":no_stay,
            "noconf_reliability":no_rel,"calibrated_stay":cal_stay,
            "adaptive_stay":ad_stay,"adaptive_strength":ad_strength}

def save_csv(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--episodes-val",type=int,default=80)
    ap.add_argument("--episodes-test",type=int,default=160)
    ap.add_argument("--steps",type=int,default=200)
    ap.add_argument("--corruptions",default="0,0.1,0.2,0.3,0.4")
    ap.add_argument("--switch-probs",default="0.01,0.05,0.10,0.20")
    ap.add_argument("--confidence-modes",default="reliable,overlapping,misleading")
    ap.add_argument("--stay-probs",default="0.55,0.65,0.70,0.75,0.80,0.85,0.90")
    ap.add_argument("--noconf-reliabilities",default="0.60,0.70,0.75,0.80,0.90")
    ap.add_argument("--adaptive-strengths",default="0.15,0.25,0.35,0.45")
    ap.add_argument("--calibration-bins",type=int,default=10)
    ap.add_argument("--decay",type=float,default=0.25)
    ap.add_argument("--seed",type=int,default=2026)
    ap.add_argument("--out-dir",default="statemem_reliability_results")
    a=ap.parse_args()

    fl=lambda x:[float(v) for v in x.split(",") if v.strip()]
    corruptions, switches, stays = fl(a.corruptions),fl(a.switch_probs),fl(a.stay_probs)
    no_rels, strengths = fl(a.noconf_reliabilities),fl(a.adaptive_strengths)
    modes=[x.strip() for x in a.confidence_modes.split(",") if x.strip()]
    out=Path(a.out_dir); out.mkdir(parents=True,exist_ok=True)

    print("Generating VALIDATION trajectories...")
    val=build_conditions(a.episodes_val,a.steps,modes,switches,corruptions,a.seed)
    print("Fitting validation-only confidence calibrators...")
    cals=fit_calibrators_by_mode(val,a.calibration_bins)

    for mode,cal in cals.items():
        print(f"\nCalibration: {mode}")
        for center,rel,n in cal.table():
            if n:
                print(f"  raw~{center:.2f} -> reliability={rel:.3f} (n={n})")

    print("\nTuning variants on VALIDATION...")
    params=tune(val,cals,a.decay,stays,no_rels,strengths)
    print(json.dumps(params,indent=2))

    print("\nGenerating independent TEST trajectories...")
    test=build_conditions(a.episodes_test,a.steps,modes,switches,corruptions,a.seed+1_000_000)
    rows=[]
    for (mode,sw,cr),ds in test.items():
        r=evaluate_dataset(ds,a.decay,cals[mode],params)
        rows.append({"confidence_mode":mode,"switch_prob":sw,"corruption_rate":cr,**r})

    print("\n"+"="*98)
    print("INDEPENDENT TEST RESULTS")
    print("="*98)
    print(f"{'Method':<24} | {'Accuracy':>9} | {'Noise Acc':>9} | {'Transition Lag':>14} | {'Brier':>9}")
    print("-"*98)
    pairs=[("Latest","latest"),("Recency","recency"),("Rec×RawConf","recency_rawconf"),
           ("StateMem-Raw","statemem_raw"),("StateMem-NoConf","statemem_noconf"),
           ("StateMem-Calibrated","statemem_calibrated"),("StateMem-Adaptive","statemem_adaptive")]
    for label,key in pairs:
        acc=mean_metric(rows,key+"_acc"); noise=mean_metric(rows,key+"_noise_acc")
        lag=mean_metric(rows,key+"_transition_lag")
        bk=key+"_brier"
        bs=f"{mean_metric(rows,bk):.4f}" if bk in rows[0] else "-"
        print(f"{label:<24} | {100*acc:8.2f}% | {100*noise:8.2f}% | {lag:14.3f} | {bs:>9}")
    print("="*98)

    print("\nMISLEADING-CONFIDENCE CONDITIONS")
    print(f"{'Switch':>7} | {'Noise':>6} | {'Raw':>7} | {'NoConf':>7} | {'Calibrated':>10} | {'Adaptive':>8} | {'Rec×Conf':>9}")
    print("-"*90)
    for r in sorted([x for x in rows if x["confidence_mode"]=="misleading"],
                    key=lambda x:(x["switch_prob"],x["corruption_rate"])):
        print(f"{r['switch_prob']:7.2f} | {100*r['corruption_rate']:5.0f}% | "
              f"{100*r['statemem_raw_acc']:6.2f}% | {100*r['statemem_noconf_acc']:6.2f}% | "
              f"{100*r['statemem_calibrated_acc']:9.2f}% | {100*r['statemem_adaptive_acc']:7.2f}% | "
              f"{100*r['recency_rawconf_acc']:8.2f}%")

    save_csv(out/"test_results.csv",rows)
    with (out/"experiment_config.json").open("w") as f:
        json.dump({"selected_parameters":params,"seed":a.seed,"modes":modes,
                   "switch_probs":switches,"corruptions":corruptions},f,indent=2)
    print(f"\nSaved results to {out}/")

if __name__=="__main__":
    main()

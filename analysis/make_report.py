"""Reproduce every number and figure of REPORT.md from data/features_20-1000Hz_anon.npz.
    python analysis/make_report.py
Writes figures/*.png and results.json.  Needs numpy, scipy, matplotlib."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import hvac_listen as h

FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)
rng = np.random.default_rng(12345)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": .25})

# ---------------------------------------------------------------- load
D = np.load(os.path.join(ROOT, "data", "features_20-1000Hz_anon.npz"))
X = D["spec_db"].astype(np.float32)
rms = D["rms_dbfs"].astype(np.float32)
freq = D["freq_hz"]
T, NB = X.shape
# continuous local-minute axis (minute-of-day + 1440*day_index)
mloc = D["minute_of_day"].astype(np.int64) + 1440 * D["day_index"].astype(np.int64)
m0 = mloc.min()
full = mloc.max() - m0 + 1
Xg = np.full((full, NB), np.nan, np.float32)
Xg[mloc - m0] = X
rg = np.full(full, np.nan, np.float32)
rg[mloc - m0] = rms
X, rms = Xg, rg
T = full
tm = m0 + np.arange(T)                    # local minute index (day*1440 + minute of day)
valid = ~np.isnan(X[:, 0])
hours = (tm - tm[0]) / 60.0
R = {"minutes": int(valid.sum()), "hours": round(T / 60, 2), "coverage": float(valid.mean()),
     "first_local_minute_of_day": int(tm[0] % 1440), "n_bins": NB, "bin_hz": float(np.diff(freq).mean())}


def clock(x_hours):
    mm = int(round(tm[0] + x_hours * 60))
    return "%02d:%02d" % ((mm % 1440) // 60, mm % 60)


def shade(ax, windows, color="tab:orange", label=None):
    for i, (a, b) in enumerate(windows):
        ax.axvspan(a, b, color=color, alpha=.15, label=label if i == 0 else None)


# schedule windows (in hours-from-start coordinates) for the pump
PUMP = [(3, 6), (15, 18)]
pump_w = []
for d in range(int(tm.min() // 1440), int(tm.max() // 1440) + 1):
    for a, b in PUMP:
        s, e = d * 1440 + a * 60, d * 1440 + b * 60
        if e > tm[0] and s < tm[-1]:
            pump_w.append(((s - tm[0]) / 60, (e - tm[0]) / 60))

# ---------------------------------------------------------------- features
med = np.median(X[valid], axis=0)
FT = h.maxpool3(h.prominence(X))
FL = X - med
feats = {"tonal": FT, "level": FL}
sdref = {k: np.nanstd(v, axis=0) for k, v in feats.items()}

# ---------------------------------------------------------------- FIG 1 overview
fig, ax = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, gridspec_kw={"height_ratios": [1, 2]})
ax[0].plot(hours, rms, lw=.8, color="k")
shade(ax[0], pump_w, label="pump schedule")
ax[0].set_ylabel("RMS (dBFS)"); ax[0].legend(loc="upper right")
ax[0].set_title("Fig 1. Broadband level and spectral tonal prominence, 20-1000 Hz (1-min resolution)")
im = ax[1].imshow(np.ma.masked_invalid(FT[:, :].T), aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=6,
                  extent=[hours[0], hours[-1], freq[0], freq[-1]])
shade(ax[1], pump_w, "cyan")
ax[1].set_ylabel("Frequency (Hz)"); ax[1].set_xlabel("local clock")
ticks = np.arange(0, hours[-1], 2)
ax[1].set_xticks(ticks); ax[1].set_xticklabels([clock(t) for t in ticks], rotation=45)
fig.colorbar(im, ax=ax, pad=.01, label="dB above local spectral floor")
fig.savefig(os.path.join(FIG, "fig1_overview.png"), bbox_inches="tight"); plt.close(fig)
R["rms_dbfs"] = {"mean": float(np.nanmean(rms)), "sd": float(np.nanstd(rms)), "min": float(np.nanmin(rms)),
                 "max": float(np.nanmax(rms))}
R["noise_floor_db"] = {"%dHz" % f: float(np.nanmedian(X[:, np.argmin(abs(freq - f))])) for f in (100, 500, 1000)}


# ---------------------------------------------------------------- pump events
def pump_events(base):
    ev = []
    for d in np.unique(tm // 1440):
        for h0, h1 in PUMP:
            s, e = d * 1440 + h0 * 60, d * 1440 + h1 * 60
            on0 = (tm >= s + 6) & (tm < e - 6) & base
            off0 = (((tm >= s - 130) & (tm < s - 10)) | ((tm >= e + 10) & (tm < e + 130))) & base
            on, off = on0 & valid, off0 & valid
            if on0.sum() > 30 and on.sum() >= .6 * on0.sum() and off0.sum() > 0 and off.sum() >= .5 * off0.sum() and off.sum() > 40:
                ev.append(dict(label="day%d %02d-%02d" % (d - tm[0] // 1440, h0, h1), on=on, off=off))
    return ev


EV = pump_events(np.ones(T, bool))
R["pump_events"] = [dict(label=e["label"], on_min=int(e["on"].sum()), off_min=int(e["off"].sum())) for e in EV]
pairs = [(e["on"], e["off"]) for e in EV]


def topk_sig(F, prs, sd, K=10):
    """Fixed-K line selection (one bin per spectral line, positive t only). No adaptive threshold, so the real
    data and permuted labels go through exactly the same procedure."""
    t, _ = h.tvals(F, prs, sd)
    t = np.nan_to_num(t, nan=-9)
    pt = np.pad(t, 3, constant_values=-9)
    loc = t >= np.lib.stride_tricks.sliding_window_view(pt, 7).max(axis=1)
    cand = h.nms(t, np.where(loc & (t > 0))[0], K)
    offall = np.any([f for _, f in prs], axis=0)
    mu, sdv = F[offall][:, cand].mean(0), F[offall][:, cand].std(0) + 1e-3
    w = np.maximum(t[cand], 1e-3); w = w / w.sum()
    return dict(bins=cand, w=w, mu=mu, sd=sdv)


def fit(kind, prs):
    sig, t = h.build_sig(feats[kind], prs, sdref[kind])
    return sig, t


def dp(sc, prs):
    return h.dprime(h.roll(sc, 15), prs)


res_pump = {}
for kind in feats:
    sig, t = fit(kind, pairs)
    if sig is None:
        continue
    sc = h.score(feats[kind], sig)
    d, m0s, m1s = dp(sc, pairs)
    res_pump[kind] = dict(sig=sig, t=t, sc=sc, d=d, m0=m0s, m1=m1s)
kind_p = max(res_pump, key=lambda k: res_pump[k]["d"])
P = res_pump[kind_p]
R["pump"] = {"kind": kind_p, "dprime_in_sample": float(P["d"]),
             "lines_hz": [round(float(freq[b]), 1) for b in P["sig"]["bins"][np.argsort(P["sig"]["w"])[::-1]]],
             "t_moderated": [round(float(P["t"][b]), 2) for b in P["sig"]["bins"][np.argsort(P["sig"]["w"])[::-1]]]}

# held-out: learn on event i, test on event j
ho = []
for i in range(len(EV)):
    for j in range(len(EV)):
        if i == j:
            continue
        sig = topk_sig(feats[kind_p], [pairs[i]], sdref[kind_p])
        sc = h.roll(h.score(feats[kind_p], sig), 15)
        d, _, _ = h.dprime(sc, [pairs[j]])
        on, off = sc[pairs[j][0]], sc[pairs[j][1]]
        on, off = on[~np.isnan(on)], off[~np.isnan(off)]
        auc = stats.mannwhitneyu(on, off).statistic / (len(on) * len(off))
        ho.append((EV[i]["label"], EV[j]["label"], float(d), float(auc)))
R["pump"]["holdout"] = [dict(train=a, test=b, dprime=c, auc=e) for a, b, c, e in ho]

# block bootstrap CI of the ON-OFF difference (dB) at signature lines
BL = 15


def blocks(mask):
    idx = np.flatnonzero(mask)
    return [b for b in np.array_split(idx, max(1, len(idx) // BL))]


def boot(F, prs, bins, B=2000):
    pre = []
    for on, off in prs:
        pre.append(([F[b][:, bins].mean(0) for b in blocks(on)], [F[b][:, bins].mean(0) for b in blocks(off)]))
    out = np.empty((B, len(bins)))
    for i in range(B):
        ds = []
        for bo, bf in pre:
            a = np.mean([bo[k] for k in rng.integers(0, len(bo), len(bo))], axis=0)
            b = np.mean([bf[k] for k in rng.integers(0, len(bf), len(bf))], axis=0)
            ds.append(a - b)
        out[i] = np.mean(ds, axis=0)
    return out


top = P["sig"]["bins"][np.argsort(P["sig"]["w"])[::-1]][:8]
bs = boot(X, pairs, top)                         # effect in raw dB (spectral level)
obs = np.mean([X[o][:, top].mean(0) - X[f][:, top].mean(0) for o, f in pairs], axis=0)
R["pump"]["line_effects_db"] = [dict(hz=round(float(freq[b]), 1), diff_db=round(float(o), 2),
                                     ci95=[round(float(np.percentile(bs[:, i], 2.5)), 2),
                                           round(float(np.percentile(bs[:, i], 97.5)), 2)])
                                for i, (b, o) in enumerate(zip(top, obs))]

# permutation null of the WHOLE pipeline (select lines -> score -> d'), block-label permutation inside events
NP = 300
null = []
for _ in range(NP):
    pp = []
    for on, off in pairs:
        idx = np.flatnonzero(on | off)
        bl = np.array_split(idx, max(1, len(idx) // BL))
        lab = np.array([on[b].mean() > .5 for b in bl])
        lab = rng.permutation(lab)
        non, noff = np.zeros(T, bool), np.zeros(T, bool)
        for b, l in zip(bl, lab):
            (non if l else noff)[b] = True
        pp.append((non, noff))
    sig = topk_sig(feats[kind_p], pp, sdref[kind_p])
    null.append(dp(h.score(feats[kind_p], sig), pp)[0])
null = np.array(null)
d_obs = dp(h.score(feats[kind_p], topk_sig(feats[kind_p], pairs, sdref[kind_p])), pairs)[0]
R["pump"]["dprime_topk10_in_sample"] = float(d_obs)
R["pump"]["perm_null"] = dict(n=NP, mean=float(null.mean()), p95=float(np.percentile(null, 95)),
                              p99=float(np.percentile(null, 99)), max=float(null.max()),
                              p_value=float((1 + (null >= d_obs).sum()) / (1 + NP)))

# FIG 2 pump spectra
fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
cols = ["tab:blue", "tab:red", "tab:green"]
for e, c in zip(EV, cols):
    ax[0].plot(freq, np.nanmean(X[e["on"]], 0), color=c, lw=.8, label=e["label"] + " ON")
    ax[0].plot(freq, np.nanmean(X[e["off"]], 0), color=c, lw=.8, ls=":", label=e["label"] + " OFF")
ax[0].set_ylabel("mean spectrum (dB)"); ax[0].legend(ncol=2, fontsize=7)
ax[0].set_title("Fig 2. Pump: ON vs OFF spectra, moderated t per frequency bin, selected lines")
for e, c in zip(EV, cols):
    ax[1].plot(freq, np.nanmean(X[e["on"]], 0) - np.nanmean(X[e["off"]], 0), color=c, lw=.8, label=e["label"])
ax[1].axhline(0, color="k", lw=.5); ax[1].set_ylabel("ON - OFF (dB)"); ax[1].legend(fontsize=7)
ax[2].plot(freq, P["t"], color="k", lw=.7)
ax[2].plot(freq[P["sig"]["bins"]], P["t"][P["sig"]["bins"]], "rv", label="selected lines (top %d)" % len(P["sig"]["bins"]))
ax[2].set_ylabel("moderated t"); ax[2].set_xlabel("Hz"); ax[2].legend()
fig.savefig(os.path.join(FIG, "fig2_pump_spectrum.png"), bbox_inches="tight"); plt.close(fig)

# FIG 3 score timeline + histograms
sm = h.roll(P["sc"], 15)
thr_hi = P["m0"] + .6 * (P["m1"] - P["m0"]); thr_lo = P["m0"] + .4 * (P["m1"] - P["m0"])
st = h.hysteresis(sm, thr_hi, thr_lo)
expn = np.zeros(T, bool)
for a, b in PUMP:
    expn |= ((tm % 1440) >= a * 60) & ((tm % 1440) < b * 60)
R["pump"]["agreement_with_schedule"] = float((st == expn)[valid].mean())
R["pump"]["detected_segments"] = [(int(s_), int(e_)) for s_, e_ in h.runs(st, 20, 10)]
fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.4, 1]})
ax[0].plot(hours, h.roll(P["sc"], 1), color="lightgray", lw=.5, label="1-min score")
ax[0].plot(hours, sm, color="k", lw=1, label="15-min mean")
ax[0].axhline(thr_hi, color="r", ls="--", lw=.8, label="ON threshold"); ax[0].axhline(thr_lo, color="b", ls="--", lw=.8, label="OFF threshold")
shade(ax[0], pump_w, label="schedule")
ax[0].fill_between(hours, ax[0].get_ylim()[0], ax[0].get_ylim()[0] + .15, where=st, color="g", step="mid", label="detected ON")
ax[0].set_xticks(ticks); ax[0].set_xticklabels([clock(t) for t in ticks], rotation=45)
ax[0].legend(fontsize=7, ncol=3); ax[0].set_title("Fig 3. Pump matched-filter score")
on_s = np.concatenate([sm[o] for o, _ in pairs]); off_s = np.concatenate([sm[f] for _, f in pairs])
bins_ = np.linspace(np.nanmin(sm), np.nanmax(sm), 40)
ax[1].hist(off_s[~np.isnan(off_s)], bins_, alpha=.6, label="OFF", color="tab:blue"); ax[1].hist(on_s[~np.isnan(on_s)], bins_, alpha=.6, label="ON", color="tab:red")
ax[1].set_title("d'=%.2f" % P["d"]); ax[1].legend()
fig.savefig(os.path.join(FIG, "fig3_pump_score.png"), bbox_inches="tight"); plt.close(fig)

# FIG 4 validation
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].hist(null, 25, color="gray"); ax[0].axvline(d_obs, color="r", label="observed d'=%.2f" % d_obs)
ax[0].axvline(np.percentile(null, 99), color="k", ls="--", label="null 99th pct"); ax[0].legend()
ax[0].set_title("Fig 4a. Block-permutation null (%d perms, whole pipeline)" % NP); ax[0].set_xlabel("in-sample d'")
labs = ["%s->%s" % (a.split()[0] + a.split()[1][:2], b.split()[0] + b.split()[1][:2]) for a, b, _, _ in ho]
ax[1].bar(range(len(ho)), [x[2] for x in ho], color="tab:green"); ax[1].set_xticks(range(len(ho))); ax[1].set_xticklabels(labs, fontsize=7)
ax[1].axhline(1, color="k", ls=":", lw=.8); ax[1].axhline(2, color="k", ls="--", lw=.8)
ax[1].set_title("Fig 4b. Held-out d' (learn on one event, test on the other)")
fig.savefig(os.path.join(FIG, "fig4_pump_validation.png"), bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- FAN
ON_LEN = 20


def fan_pairs(phase, ml=None):
    rel = tm - phase; cyc, w = rel // 60, rel % 60
    prs = []
    for c in np.unique(cyc):
        s = cyc == c
        on = s & (w >= 1) & (w < ON_LEN - 1) & valid
        off = s & (w >= ON_LEN + 1) & (w < 59) & valid
        if on.sum() >= .7 * (ON_LEN - 2) and off.sum() >= .7 * (60 - ON_LEN - 2):
            prs.append((on, off))
    return prs


def curve(F, sd, perm=None):
    c = np.full(60, np.nan)
    for ph in range(60):
        pr = fan_pairs(ph)
        if len(pr) >= 3:
            t, _ = h.tvals(F, pr, sd)
            c[ph] = np.sort(np.nan_to_num(t, nan=-9))[-4:].mean()
    return c


curves = {k: curve(feats[k], sdref[k]) for k in feats}
R["fan"] = {"cycles_at_best": {}, "curves": {k: [None if np.isnan(x) else round(float(x), 3) for x in v] for k, v in curves.items()}}
for k, c in curves.items():
    mad = 1.4826 * np.nanmedian(abs(c - np.nanmedian(c))) + 1e-6
    R["fan"][k] = dict(best_phase=int(np.nanargmax(c)), peak=float(np.nanmax(c)), median=float(np.nanmedian(c)),
                       robust_confidence=float((np.nanmax(c) - np.nanmedian(c)) / mad))
# (a block-rotation surrogate was removed: its wrap-around seam creates coherent hourly structure, so it is not a valid null.
#  The hourly hypothesis is judged by BLS at P = 60 min, fig9, and by the detection limit, fig8.)

# phase-free test: Fourier amplitude at the 60-min period per bin vs neighbouring periods
Xf = np.where(np.isnan(FT), 0, FT)                      # gaps -> 0 (only edges)
Xf = Xf - np.convolve(np.ones(1), [1])[0] * 0
hp = Xf - np.apply_along_axis(lambda v: np.convolve(v, np.ones(181) / 181, "same"), 0, Xf)   # remove > 3 h trend
N = T
spec = np.abs(np.fft.rfft(hp * np.hanning(N)[:, None], axis=0))
fr = np.fft.rfftfreq(N, d=1.0)                            # cycles / minute
k60 = np.argmin(abs(fr - 1 / 60))
nbh = [i for i in range(len(fr)) if 1 / 100 <= fr[i] <= 1 / 35 and abs(i - k60) > 2]
zline = (spec[k60] - np.median(spec[nbh], 0)) / (1.4826 * np.median(abs(spec[nbh] - np.median(spec[nbh], 0)), 0) + 1e-9)
R["fan"]["period60_z"] = dict(max=float(zline.max()), max_hz=float(freq[zline.argmax()]), n_gt_3=int((zline > 3).sum()),
                              n_gt_4=int((zline > 4).sum()), expected_gt_3_under_null=float(NB * (1 - stats.norm.cdf(3))))
ph_min = (np.angle(np.fft.rfft(hp * np.hanning(N)[:, None], axis=0)[k60]) / (-2 * np.pi) * 60) % 60   # minute of hour at max of the 60-min component (relative to data start)
ph_min = (ph_min + tm[0]) % 60
R["fan"]["period60_bins_z_gt3"] = [dict(hz=round(float(freq[i]), 1), z=round(float(zline[i]), 2), peak_minute=round(float(ph_min[i]), 1))
                                   for i in np.flatnonzero(zline > 3)]
fig, ax = plt.subplots(2, 2, figsize=(11, 6.5))
for k, a_ in zip(feats, ax[0]):
    a_.plot(range(60), curves[k], "k.-")
    a_.axhline(np.nanmedian(curves[k]), color="gray", ls="--", label="median over phases")
    a_.set_title("Fig 5. Fan phase scan - %s features" % k); a_.set_xlabel("assumed start minute of the hour"); a_.set_ylabel("mean of 4 strongest line t-values"); a_.legend(fontsize=7)
ax[1, 0].plot(freq, zline, lw=.6, color="k"); ax[1, 0].axhline(3, color="orange", ls="--"); ax[1, 0].axhline(4, color="r", ls="--")
ax[1, 0].set_xlabel("Hz"); ax[1, 0].set_ylabel("z of 60-min Fourier line"); ax[1, 0].set_title("Fig 6a. Phase-free test: 1/60 min^-1 component per frequency bin")
zz = np.sort(zline); ax[1, 1].hist(zline, 40, density=True, color="gray", alpha=.7)
xx = np.linspace(-4, 6, 200); ax[1, 1].plot(xx, stats.norm.pdf(xx), "r"); ax[1, 1].set_title("Fig 6b. z distribution vs N(0,1)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_fan.png")); plt.close(fig)

# folded profile of band-averaged prominence by minute of hour (flat if no fan)
bandp = np.nanmean(FT, axis=1)
mo = tm % 60
prof = np.array([np.nanmean(bandp[mo == k]) for k in range(60)])
sem = np.array([np.nanstd(bandp[mo == k]) / np.sqrt((mo == k).sum()) for k in range(60)])
fig, ax = plt.subplots(figsize=(8, 3))
ax.errorbar(range(60), prof, sem, fmt="k.-", lw=.8); ax.set_xlabel("minute of the hour"); ax.set_ylabel("mean tonal prominence 20-1000 Hz (dB)")
ax.set_title("Fig 7. Hour-folded broadband tonal prominence (a 20-min-per-hour fan would give a 20-min step)")
fig.savefig(os.path.join(FIG, "fig7_folded.png"), bbox_inches="tight"); plt.close(fig)
R["fan"]["folded_range_db"] = float(np.nanmax(prof) - np.nanmin(prof))
R["fan"]["folded_sem_db"] = float(np.nanmedian(sem))

# power / minimum detectable effect for the fan
prs = fan_pairs(R["fan"]["tonal"]["best_phase"])
Dm = np.array([FT[o].mean(0) - FT[f].mean(0) for o, f in prs])
sd_cyc = np.median(Dm.std(0, ddof=1))
n_now = len(prs)
alpha = 0.05 / NB                                       # Bonferroni over frequency bins
def mde(n):
    return (stats.t.ppf(1 - alpha / 2, n - 1) + stats.t.ppf(.8, n - 1)) * sd_cyc / np.sqrt(n)
R["fan"]["power"] = dict(cycles_now=n_now, sd_cycle_diff_db=float(sd_cyc), mde_now_db=float(mde(n_now)),
                         mde_72_cycles_db=float(mde(72)), mde_168_cycles_db=float(mde(168)))
ns = np.arange(4, 200)
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(ns, [mde(n) for n in ns], "k"); ax.axvline(n_now, color="r", ls="--", label="now (%d cycles)" % n_now)
ax.set_xlabel("hourly cycles recorded"); ax.set_ylabel("min. detectable ON-OFF line (dB)")
ax.set_title("Fig 8. Fan detection limit (80%% power, Bonferroni over %d bins)" % NB); ax.legend()
fig.savefig(os.path.join(FIG, "fig8_power.png"), bbox_inches="tight"); plt.close(fig)

exec(open(os.path.join(ROOT, "analysis", "fan_period.py"), encoding="utf-8").read())

json.dump(R, open(os.path.join(ROOT, "results.json"), "w"), indent=1)
print("done")

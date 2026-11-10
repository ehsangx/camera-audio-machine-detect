"""Period-free analysis of the periodic ON/OFF source (appended to make_report.py by `exec`)."""
# ------------------------------------------------------------------ BLS on the band level and on RMS
lvl = np.nanmean(X, axis=1)
bl_lvl = h.bls_scan(lvl, tm, valid, nperm=300, seed=7)
bl_rms = h.bls_scan(rms, tm, valid, nperm=100, seed=8)
for name, b in (("band_level", bl_lvl), ("rms", bl_rms)):
    R.setdefault("periodic", {})[name] = dict(period=b["period"], on_len=b["on_len"], phase=b["phase"], snr=b["snr"],
                                              null_p99=b["null_p99"], p=b["p"])
P_, L_, PH_ = bl_lvl["period"], bl_lvl["on_len"], bl_lvl["phase"]

fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
ax[0].plot(bl_lvl["periods"], bl_lvl["curve"], "k", label="band level (20-1000 Hz)")
ax[0].plot(bl_rms["periods"], bl_rms["curve"], color="tab:blue", lw=.8, label="RMS")
ax[0].axhline(bl_lvl["null_p99"], color="r", ls="--", label="block-permutation null 99%")
ax[0].axvline(60, color="gray", ls=":", label="60 min (assumed)")
ax[0].set_xlabel("period (min)"); ax[0].set_ylabel("best box-fit SNR"); ax[0].legend(fontsize=7)
ax[0].set_title("Fig 9a. Box-least-squares periodogram")
ax[1].plot(hours, rms, lw=.6, color="k")
st_on = rms > np.nanmean([np.nanpercentile(rms, 10), np.nanpercentile(rms, 90)])
ax[1].set_title("Fig 9b. RMS (dBFS) with the fitted 81-min box grid")
k0 = np.arange(-3, int(T / P_) + 4)
for k in k0:
    s = (PH_ - tm[0] % P_) % P_ + k * P_
    ax[1].axvspan(s / 60, (s + L_) / 60, color="orange", alpha=.2)
ax[1].set_xlim(hours[0], hours[-1]); ax[1].set_xticks(ticks); ax[1].set_xticklabels([clock(t) for t in ticks], rotation=45, fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig9_period_scan.png")); plt.close(fig)

# ------------------------------------------------------------------ edge timing regression from RMS
cut = 0.5 * (np.nanpercentile(rms, 10) + np.nanpercentile(rms, 90))
for _ in range(10):                                         # 2-means
    hi_, lo_ = np.nanmean(rms[rms > cut]), np.nanmean(rms[rms <= cut])
    cut = .5 * (hi_ + lo_)
state = np.nan_to_num(rms, nan=-99) > cut
segs = [(a, b) for a, b in h.runs(state, 10, 2) if 16 <= b - a + 1 <= 23]
starts = np.array([tm[a] for a, b in segs], float)
durs = np.array([b - a + 1 for a, b in segs], float)
kidx = np.round((starts - starts[0]) / P_)
# use only the clean train of pulses (regular grid) for the regression
res = stats.linregress(kidx, starts)
resid = starts - (res.intercept + res.slope * kidx)
good = np.abs(resid) < 3
res2 = stats.linregress(kidx[good], starts[good])
resid2 = starts[good] - (res2.intercept + res2.slope * kidx[good])
nxt_k = np.arange(kidx.max() + 1, kidx.max() + 8)
nxt = res2.intercept + res2.slope * nxt_k
R["periodic"]["edges"] = dict(
    threshold_dbfs=float(cut), rms_on_mean=float(hi_), rms_off_mean=float(lo_), step_db=float(hi_ - lo_),
    n_pulses=int(len(segs)), n_regular=int(good.sum()), period_est=float(res2.slope), period_se=float(res2.stderr),
    period_ci95=[float(res2.slope - 1.96 * res2.stderr), float(res2.slope + 1.96 * res2.stderr)],
    resid_sd_min=float(resid2.std(ddof=2)), on_dur_mean=float(durs[good].mean()), on_dur_sd=float(durs[good].std(ddof=1)),
    off_dur_mean=float(res2.slope - durs[good].mean()),
    onset_clock=[("%02d:%02d" % ((int(x) % 1440) // 60, int(x) % 60)) for x in starts[good]],
    next_onsets_clock=[("%02d:%02d" % ((int(round(x)) % 1440) // 60, int(round(x)) % 60)) for x in nxt])
R["periodic"]["rms_duty_cycle"] = float(durs[good].mean() / res2.slope)

# ------------------------------------------------------------------ fan spectrum + cycle-level bootstrap + odd/even holdout
prs81 = h.fan_pairs(tm, valid, PH_, L_, P_)
Dcyc = np.array([FL[o].mean(0) - FL[f].mean(0) for o, f in prs81])         # ON-OFF (dB), per cycle
mean_d, se_d = Dcyc.mean(0), Dcyc.std(0, ddof=1) / np.sqrt(len(Dcyc))
tcyc = mean_d / se_d
R["periodic"]["spectrum"] = dict(n_cycles=len(prs81), mean_diff_db_20_100=float(mean_d[freq < 100].mean()),
                                 mean_diff_db_100_300=float(mean_d[(freq >= 100) & (freq < 300)].mean()),
                                 mean_diff_db_300_600=float(mean_d[(freq >= 300) & (freq < 600)].mean()),
                                 mean_diff_db_600_1000=float(mean_d[freq >= 600].mean()),
                                 frac_bins_p_lt_0_001=float((np.abs(tcyc) > stats.t.ppf(1 - .0005, len(prs81) - 1)).mean()),
                                 max_t=float(tcyc.max()), max_t_hz=float(freq[tcyc.argmax()]))
ho_scores = []
sig_all = None
for par in (0, 1):
    tr = [p for i, p in enumerate(prs81) if i % 2 == par]
    te = [p for i, p in enumerate(prs81) if i % 2 != par]
    sg = topk_sig(FL, tr, sdref["level"], K=10)
    scr = h.roll(h.score(FL, sg), 5)
    ho_scores.append(float(h.dprime(scr, te)[0]))
R["periodic"]["holdout_dprime_odd_even"] = ho_scores
sg = topk_sig(FL, prs81, sdref["level"], K=10)
R["periodic"]["signature_lines_hz"] = [round(float(freq[b]), 1) for b in sg["bins"]]
R["periodic"]["signature_dprime_in_sample"] = float(h.dprime(h.roll(h.score(FL, sg), 5), prs81)[0])

fig, ax = plt.subplots(3, 1, figsize=(11, 8.5), sharex=False)
ax[0].plot(freq, mean_d, "k", lw=.8)
ax[0].fill_between(freq, mean_d - 1.96 * se_d, mean_d + 1.96 * se_d, color="gray", alpha=.4)
ax[0].axhline(0, color="r", lw=.5); ax[0].set_ylabel("ON - OFF (dB)"); ax[0].set_xlabel("Hz")
ax[0].set_title("Fig 10a. Spectrum of the 81-min source: mean ON-OFF difference over %d cycles (95%% CI)" % len(prs81))
ax[1].plot(freq, tcyc, "k", lw=.6); ax[1].axhline(stats.t.ppf(1 - .05 / 2 / NB, len(prs81) - 1), color="r", ls="--", label="Bonferroni 5%")
ax[1].set_ylabel("paired t over cycles"); ax[1].set_xlabel("Hz"); ax[1].legend()
ax[1].set_title("Fig 10b. Per-bin paired t-statistic")
# folded rms
ph_rel = (tm - PH_) % P_
mean_f = np.array([np.nanmean(rms[ph_rel == k]) for k in range(P_)]); sd_f = np.array([np.nanstd(rms[ph_rel == k]) for k in range(P_)])
ax[2].plot(range(P_), mean_f, "k.-"); ax[2].fill_between(range(P_), mean_f - sd_f, mean_f + sd_f, color="gray", alpha=.3)
ax[2].axvspan(0, L_, color="orange", alpha=.2, label="fitted ON window (%d min)" % L_)
ax[2].set_xlabel("minute within the %d-min cycle (0 = fitted onset)" % P_); ax[2].set_ylabel("RMS (dBFS)"); ax[2].legend()
ax[2].set_title("Fig 10c. Folded RMS (mean +- 1 sd across cycles)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig10_periodic_source.png")); plt.close(fig)

# edges figure
fig, ax = plt.subplots(1, 2, figsize=(11, 3.4))
ax[0].plot(kidx[good], resid2, "ko-"); ax[0].axhline(0, color="r", lw=.5)
ax[0].set_xlabel("cycle index"); ax[0].set_ylabel("onset residual (min)"); ax[0].set_title("Fig 11a. Onset-time residuals of the linear fit")
ax[1].hist(durs[good], bins=np.arange(16.5, 23.5, 1), color="gray"); ax[1].set_xlabel("ON duration (min)"); ax[1].set_title("Fig 11b. ON durations")
fig.savefig(os.path.join(FIG, "fig11_edges.png"), bbox_inches="tight"); plt.close(fig)

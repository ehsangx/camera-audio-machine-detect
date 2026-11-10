#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hvac_listen.py - listen to a camera's RTSP audio and find faint machine sounds
(a fan on the floor below: 20 min ON / 40 min OFF every hour, and a pool pump:
ON 03-06 and 15-18) using their KNOWN SCHEDULES as the teacher.

Why this is more than "look at the volume":
  * Faint, steady machines are narrow-band tones (motor hum 50/100/150 Hz, blade
    pass, bearing lines) sitting far below the room noise. Volume (RMS) never
    shows them, but a 1 Hz-resolution spectrum integrated over a minute does.
  * Every minute is stored as a transient-robust spectrum (25th percentile of
    ~117 FFT frames), so talking / door slams / dogs are rejected.
  * Features: 'tonal' (line prominence above the local spectral floor, +-1 Hz
    tolerant) and 'level' (per-bin level vs. the record median).
  * Learning is a matched filter: every frequency bin is tested for "higher when
    the machine should be ON than in its neighbouring OFF period" with a paired
    t-test across cycles (fan: 20-min ON vs the 40-min OFF of the SAME hour;
    pump: each 3 h window vs the 2 h before/after it). Slow drift in ambient
    noise cancels out. The fan phase (which minute of the hour it starts) is
    found by scanning all 60 phases.
  * The fan is masked out when learning the pump (and vice versa is cancelled by
    design) so the two do not contaminate each other.
  * Result = a signature (frequencies + weights + threshold), a cross-validated
    separability score (d'), the detected ON/OFF events, and a live monitor.

Usage (run it on the host that can reach the camera):
  export RTSP_USER=<user> RTSP_PASS='<password>' RTSP_HOST=<camera-ip> RTSP_PATH=/<stream-path>
  python3 hvac_listen.py probe                       # check the audio stream
  nohup python3 hvac_listen.py record  --out data &   # collect >= 24 h (48 h better)
  python3 hvac_listen.py analyze --out data          # learn signatures + report
  python3 hvac_listen.py monitor --out data          # live detection (record+detect)
Needs: python3, numpy, ffmpeg (ffprobe for 'probe').
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from urllib.parse import quote

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SR = 8000                    # camera audio is usually 8 kHz G.711 anyway
NFFT = 8192                  # ~0.98 Hz resolution
HOP = NFFT // 2
FMIN, FMAX = 20, 3500
K0, K1 = int(np.ceil(FMIN * NFFT / SR)), int(FMAX * NFFT / SR)
NB = K1 - K0
WIN = np.hanning(NFFT).astype(np.float32)
MIN_VALID = 0.6              # min fraction of a minute that must have audio
REC = np.dtype([("ts", "<f8"), ("rms", "<f4"), ("valid", "<f4"), ("spec", "<f4", (NB,))])


def hz(k):
    return (K0 + np.asarray(k)) * SR / NFFT


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #
def rtsp_url(a):
    # the password may contain '@' -> must be percent-encoded
    return "rtsp://%s:%s@%s:%s%s" % (quote(a.user, safe=""), quote(a.password, safe=""),
                                     a.host, a.port, a.path)


def masked(url):
    pre, post = url.rsplit("@", 1)
    return pre.split("//")[0] + "//***:***@" + post


def minute_feature(x):
    """int16 samples of ~1 minute -> (spec_dB[NB], rms_dBFS) or None"""
    x = x.astype(np.float32) / 32768.0
    x -= x.mean()
    n = (len(x) - NFFT) // HOP + 1
    if n < 16:
        return None
    idx = np.arange(NFFT)[None, :] + HOP * np.arange(n)[:, None]
    P = np.abs(np.fft.rfft(x[idx] * WIN, axis=1)) ** 2 / float((WIN ** 2).sum())
    p = np.percentile(P[:, K0:K1], 25, axis=0)          # transient-robust floor
    spec = (10 * np.log10(p + 1e-14)).astype(np.float32)
    rms = 20 * np.log10(float(np.sqrt(np.mean(x ** 2))) + 1e-9)
    return spec, rms


class Recorder:
    def __init__(self, url, out, on_minute=None):
        self.url, self.out, self.on_minute = url, out, on_minute
        self.stop = False
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "meta.json"), "w") as f:
            json.dump(dict(sr=SR, nfft=NFFT, k0=K0, nb=NB), f)
        self.fn = os.path.join(out, "feat.bin")

    def _flush(self, minute, buf):
        if not buf:
            return
        x = np.concatenate(buf)
        valid = len(x) / (SR * 60.0)
        if valid < MIN_VALID:
            print("[%s] minute skipped (only %.0f%% audio)" % (self._t(minute), valid * 100), flush=True)
            return
        r = minute_feature(x)
        if r is None:
            return
        rec = np.zeros(1, REC)
        rec["ts"], rec["rms"], rec["valid"], rec["spec"] = minute * 60.0, r[1], min(valid, 1.0), r[0]
        with open(self.fn, "ab") as f:
            f.write(rec.tobytes())
        if self.on_minute:
            self.on_minute(minute * 60.0, r[0])
        else:
            print("[%s] rms=%.1f dBFS ok" % (self._t(minute), r[1]), flush=True)

    @staticmethod
    def _t(minute):
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(minute * 60))

    def run(self):
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp",
               "-i", self.url, "-vn", "-ac", "1", "-ar", str(SR),
               "-f", "s16le", "-"]
        print("capturing from", masked(self.url), flush=True)
        fails = 0
        while not self.stop:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            last = [time.time()]
            got = [0]

            def watchdog():
                while proc.poll() is None and not self.stop:
                    if time.time() - last[0] > 25:
                        print("stream stalled -> restarting", flush=True)
                        proc.kill()
                        return
                    time.sleep(2)
            threading.Thread(target=watchdog, daemon=True).start()
            cur, buf = int(time.time() // 60), []
            while not self.stop:
                d = proc.stdout.read(SR)                  # 0.5 s of int16
                if not d:
                    break
                last[0] = time.time()
                got[0] += len(d)
                m = int(time.time() // 60)
                if m != cur:
                    self._flush(cur, buf)
                    cur, buf = m, []
                buf.append(np.frombuffer(d[: len(d) // 2 * 2], np.int16))
            proc.kill()
            if self.stop:
                break
            fails = fails + 1 if got[0] == 0 else 0
            if fails >= 5:
                sys.exit("ffmpeg gets no audio from the stream (no audio track enabled on the camera? "
                         "wrong credentials/URL path?)")
            print("stream ended, reconnecting in 5 s", flush=True)
            time.sleep(5)


# --------------------------------------------------------------------------- #
# Features / statistics
# --------------------------------------------------------------------------- #
def prominence(X, w=41):
    """dB above the running median across frequency (tonal lines stand out)."""
    out = np.empty_like(X)
    pad = w // 2
    for i in range(0, len(X), 32):
        b = X[i:i + 32]
        P = np.pad(b, ((0, 0), (pad, pad)), mode="edge")
        out[i:i + 32] = b - np.median(sliding_window_view(P, w, axis=1), axis=2)
    return out


def maxpool3(F):                     # tolerate +-1 bin (~1 Hz) of motor speed drift
    P = np.pad(F, ((0, 0), (1, 1)), mode="edge")
    return np.maximum(np.maximum(P[:, :-2], P[:, 1:-1]), P[:, 2:])


def roll(s, k):
    v = ~np.isnan(s)
    x = np.where(v, s, 0.0)
    ker = np.ones(k)
    num, den = np.convolve(x, ker, "same"), np.convolve(v.astype(float), ker, "same")
    out = np.where(den >= k * 0.5, num / np.maximum(den, 1), np.nan)
    out[~v] = np.nan
    return out


def tvals(F, pairs, sd_ref):
    """paired ON-OFF contrast t-value per bin across (on,off) mask pairs."""
    D = np.array([F[o].mean(0) - F[f].mean(0) for o, f in pairs])
    n = len(D)
    if n >= 3:                      # moderated t (SAM): s0 stops tiny-variance bins from faking huge t
        sdk = D.std(0, ddof=1)
        return D.mean(0) / ((sdk + np.median(sdk)) / np.sqrt(n) + 1e-3), n
    non = sum(o.sum() for o, _ in pairs)          # provisional: assume ~15 min autocorrelation
    noff = sum(f.sum() for _, f in pairs)
    return D.mean(0) / (sd_ref * np.sqrt(15.0 / max(non, 1) + 15.0 / max(noff, 1)) + 1e-3), n


def build_sig(F, pairs, sd_ref, K=25):
    t, n = tvals(F, pairs, sd_ref)
    t = np.nan_to_num(t, nan=-9)
    tmin = max(1.0, np.median(t) + 5 * 1.4826 * np.median(np.abs(t - np.median(t))))   # 5 robust sigmas above the noise bins
    pt = np.pad(t, 3, constant_values=-9)
    loc = t >= sliding_window_view(pt, 7).max(axis=1)          # one bin per spectral line
    cand = np.where(loc & (t > tmin))[0]
    cand = cand[np.argsort(t[cand])[::-1][:K]]
    if len(cand) < 2:
        return None, t
    offall = np.any([f for _, f in pairs], axis=0)
    mu, sd = F[offall][:, cand].mean(0), F[offall][:, cand].std(0) + 1e-3
    w = t[cand] / t[cand].sum()
    return dict(bins=cand, w=w, mu=mu, sd=sd), t


def score(F, sig):
    return ((F[:, sig["bins"]] - sig["mu"]) / sig["sd"]) @ sig["w"]


def dprime(sm, pairs):
    on = np.concatenate([sm[o] for o, _ in pairs])
    off = np.concatenate([sm[f] for _, f in pairs])
    on, off = on[~np.isnan(on)], off[~np.isnan(off)]
    if len(on) < 5 or len(off) < 5:
        return float("nan"), 0, 0
    d = (on.mean() - off.mean()) / np.sqrt(0.5 * (on.var() + off.var()) + 1e-9)
    return d, off.mean(), on.mean()


def cv_dprime(F, pairs, sd_ref, smooth):
    """learn on half of the cycles/events, test on the other half (and back)."""
    if len(pairs) < 4:
        return float("nan")
    res = []
    for par in (0, 1):
        tr = [p for i, p in enumerate(pairs) if i % 2 == par]
        te = [p for i, p in enumerate(pairs) if i % 2 != par]
        sig, _ = build_sig(F, tr, sd_ref)
        if sig is None:
            res.append(0.0)
            continue
        res.append(dprime(roll(score(F, sig), smooth), te)[0])
    return float(np.nanmean(res))


def runs(state, min_len, max_gap):
    idx = np.flatnonzero(state)
    if len(idx) == 0:
        return []
    segs, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > max_gap + 1:
            segs.append([s, p])
            s = i
        p = i
    segs.append([s, p])
    return [(a, b) for a, b in segs if b - a + 1 >= min_len]


def hysteresis(sm, hi, lo):
    st, cur = np.zeros(len(sm), bool), False
    for i, v in enumerate(sm):
        if np.isnan(v):
            st[i] = cur
            continue
        if not cur and v > hi:
            cur = True
        elif cur and v < lo:
            cur = False
        st[i] = cur
    return st


# --------------------------------------------------------------------------- #
# Analyze
# --------------------------------------------------------------------------- #
def load(out):
    meta = json.load(open(os.path.join(out, "meta.json")))
    if (meta["sr"], meta["nfft"], meta["nb"]) != (SR, NFFT, NB):
        sys.exit("data/meta mismatch")
    r = np.fromfile(os.path.join(out, "feat.bin"), dtype=REC)
    if len(r) < 30:
        sys.exit("not enough data yet (%d minutes)" % len(r))
    m = (r["ts"] // 60).astype(np.int64)
    m0 = m.min()
    X = np.full((m.max() - m0 + 1, NB), np.nan, np.float32)
    X[m - m0] = r["spec"]
    return m0, X


def bls_scan(y, mloc, valid, periods=range(30, 241), on_lens=range(12, 31), nperm=60, seed=1):
    """Box-least-squares search for a periodic ON/OFF pattern in a 1-D level series (dB).
    Needs NO assumption about the period. Returns dict(period, phase, on_len, snr, p, curve) or None.
    Significance: block-permutation null (30-min blocks) of the max SNR over all (period, phase, on_len)."""
    ok = valid & ~np.isnan(y)
    if ok.sum() < 240:
        return None
    med_ = np.median(y[ok])
    sd_ = 1.4826 * np.median(np.abs(y[ok] - med_)) + 1e-6
    z0 = np.clip((y - med_) / sd_, -4, 4)

    def best(z):
        zz = np.where(ok, z, 0.0)
        tot, totn = zz.sum(), ok.sum()
        top, curve = (-1e9, None), []
        for P in periods:
            if P * 4 > len(y):
                curve.append(np.nan)
                continue
            idx = mloc % P
            sm = np.bincount(idx, weights=zz, minlength=P)
            ct = np.bincount(idx, weights=ok.astype(float), minlength=P)
            cs = np.concatenate([[0], np.cumsum(np.tile(sm, 2))])
            cc = np.concatenate([[0], np.cumsum(np.tile(ct, 2))])
            pbest = -1e9
            for L in on_lens:
                if L >= P - 5:
                    continue
                ons, onn = cs[L:L + P] - cs[:P], cc[L:L + P] - cc[:P]
                offs, offn = tot - ons, totn - onn
                good = (onn >= 4 * L * 0.7) & (offn > 10)
                if not good.any():
                    continue
                snr = (ons / np.maximum(onn, 1) - offs / np.maximum(offn, 1)) / np.sqrt(1 / np.maximum(onn, 1) + 1 / np.maximum(offn, 1))
                snr = np.where(good, snr, -1e9)
                k = int(np.argmax(snr))
                pbest = max(pbest, snr[k])
                if snr[k] > top[0]:
                    top = (snr[k], (P, k, L))
            curve.append(pbest if pbest > -1e8 else np.nan)
        return top, np.array(curve)

    (snr, (P, ph, L)), curve = best(z0)
    rng = np.random.default_rng(seed)
    blk = 30
    nb = len(y) // blk
    null = []
    for _ in range(nperm):
        order = rng.permutation(nb)
        zp = z0.copy()
        zp[:nb * blk] = z0[:nb * blk].reshape(nb, blk)[order].ravel()
        null.append(best(zp)[0][0])
    null = np.array(null)
    return dict(period=int(P), phase=int(ph), on_len=int(L), snr=float(snr), null_p99=float(np.percentile(null, 99)),
                p=float((1 + (null >= snr).sum()) / (1 + nperm)), curve=curve, periods=list(periods))


def fan_pairs(mloc, valid, phase, on_len, period=60):
    rel = mloc - phase
    cyc, w = rel // period, rel % period
    pairs = []
    for c in np.unique(cyc):
        s = cyc == c
        on = s & (w >= 1) & (w < on_len - 1) & valid          # skip edge minutes (timing jitter)
        off = s & (w >= on_len + 1) & (w < period - 1) & valid
        if on.sum() >= 0.7 * (on_len - 2) and off.sum() >= 0.7 * (period - on_len - 2):
            pairs.append((on, off))
    return pairs


def pump_pairs(mloc, valid, base, windows, flank=120, gap=10, margin=6):
    """valid&base: usable minutes (base = not fan-ON). One (on,off) pair per window per day."""
    pairs = []
    for d in np.unique(mloc // 1440):
        for h0, h1 in windows:
            s, e = d * 1440 + h0 * 60, d * 1440 + h1 * 60
            on0 = (mloc >= s + margin) & (mloc < e - margin) & base
            off0 = (((mloc >= s - flank - gap) & (mloc < s - gap)) |
                    ((mloc >= e + gap) & (mloc < e + gap + flank))) & base
            on, off = on0 & valid, off0 & valid
            if on.sum() >= 0.6 * on0.sum() and on0.sum() > 30 and off.sum() >= 0.5 * max(off0.sum(), 1) and off.sum() > 40:
                pairs.append((on, off))
    return pairs


def verdict(d):
    if np.isnan(d):
        return "n/a"
    return "STRONG" if d >= 2 else "moderate" if d >= 1 else "WEAK / not reliable"


def timeline(name, state, valid, mloc, fmt_day):
    print("\n%s timeline (each char = 15 min; #=ON  .=OFF  ?=no data)" % name)
    print("            " + "".join("%-4d" % h for h in range(0, 24, 1)).replace("   ", "   "))
    days = mloc // 1440
    for d in np.unique(days):
        line = ""
        for slot in range(96):
            sel = (days == d) & ((mloc % 1440) // 15 == slot)
            nv = (sel & valid).sum()
            line += "?" if nv < 8 else ("#" if (state & sel).sum() >= nv / 2 else ".")
        print("  %s  %s" % (fmt_day(d), "".join(line[i:i + 4] for i in range(0, 96, 4)).replace("", "")))


def cmd_analyze(a):
    m0, X = load(a.out)
    T = len(X)
    off = a.utc_offset if a.utc_offset is not None else time.localtime(m0 * 60).tm_gmtoff // 60
    mloc = m0 + off + np.arange(T)
    valid = ~np.isnan(X[:, 0])

    def fmt(i):
        return datetime.fromtimestamp((m0 + i) * 60 + off * 60, timezone.utc).strftime("%Y-%m-%d %H:%M")

    def fmt_day(d):
        return datetime.fromtimestamp(int(d) * 86400, timezone.utc).strftime("%Y-%m-%d")

    print("data: %d minutes (%.1f h), coverage %.0f%%, local UTC offset %+d min" %
          (T, T / 60, 100 * valid.mean(), off))
    if valid.sum() < 180:
        print("WARNING: <3 h of data: result will be unreliable. Fan needs >=3 cycles, pump >=1 day.")
    print("computing features ...", flush=True)
    med = np.median(X[valid], axis=0)
    nlim = int(a.fmax_sig * NFFT / SR) - K0          # machine hum lives low: search only below fmax_sig
    feats = {"tonal": maxpool3(prominence(X))[:, :nlim], "level": (X - med)[:, :nlim]}
    print("signature search band: %d-%d Hz" % (FMIN, a.fmax_sig))
    sdref = {k: np.nanstd(v, axis=0) for k, v in feats.items()}
    events_out = []
    windows = [tuple(int(x) for x in w.split("-")) for w in a.pump_windows.split(",")]

    # ---------------- FAN ----------------
    fan_P, fan_L = max(a.fan_period, 60 if a.fan_period <= 0 else a.fan_period), a.fan_on
    lvl = np.nanmean(X, axis=1)
    bl = bls_scan(lvl, mloc, valid) if (a.fan_period <= 0 or a.fan_auto) else None
    if bl:
        print("\n=== FAN: period-free box-least-squares scan (30-240 min) on the band level ===")
        print("  best: period %d min, ON %d min, start :%02d (mod period)  SNR=%.1f  null99%%=%.1f  p=%.3f" %
              (bl["period"], bl["on_len"], bl["phase"], bl["snr"], bl["null_p99"], bl["p"]))
        if bl["p"] < 0.05:
            fan_P, fan_L = bl["period"], bl["on_len"]
            print("  -> significant periodic ON/OFF source. Using period=%d min, ON=%d min (a 60-min period is %s)" %
                  (fan_P, fan_L, "consistent" if fan_P == 60 else "NOT what the data show"))
        else:
            print("  -> no significant periodic pattern in the broadband level; using the assumed schedule")
    print("\n=== FAN: %d min ON every %d min, scanning %d phases ===" % (fan_L, fan_P, fan_P))
    best = None
    for kind, F in feats.items():
        curve = np.full(fan_P, np.nan)
        for ph in range(fan_P):
            pr = fan_pairs(mloc, valid, ph, fan_L, fan_P)
            if len(pr) >= 3:
                t, _ = tvals(F, pr, sdref[kind])
                curve[ph] = np.sort(np.nan_to_num(t, nan=-9))[-4:].mean()   # strongest lines
        if np.all(np.isnan(curve)):
            continue
        mad = 1.4826 * np.nanmedian(np.abs(curve - np.nanmedian(curve))) + 1e-6
        conf = (np.nanmax(curve) - np.nanmedian(curve)) / mad
        print("  [%s] best start = :%02d  peak=%.1f  phase-curve confidence=%.1f"
              % (kind, np.nanargmax(curve), np.nanmax(curve), conf))
        if best is None or conf > best[0]:
            best = (conf, kind, int(np.nanargmax(curve)))
    fan_mask = np.zeros(T, bool)
    if best is None:
        print("  not enough full cycles yet.")
    else:
        conf, kind, ph = best
        F, pr = feats[kind], fan_pairs(mloc, valid, ph, fan_L, fan_P)
        sig, _ = build_sig(F, pr, sdref[kind])
        for _ in range(3 if sig is not None else 0):            # refine the phase on the learned signature
            sc0 = score(F, sig)
            prof = np.array([np.nanmean(sc0[(mloc % fan_P) == k]) for k in range(fan_P)])
            gain = [np.mean(np.take(prof, np.arange(p_, p_ + fan_L), mode="wrap")) -
                    np.mean(np.take(prof, np.arange(p_ + fan_L, p_ + fan_P), mode="wrap")) for p_ in range(fan_P)]
            ph2 = int(np.argmax(gain))
            pr2 = fan_pairs(mloc, valid, ph2, fan_L, fan_P)
            sig2, _ = build_sig(F, pr2, sdref[kind])
            if sig2 is None:
                break
            ph, pr, sig = ph2, pr2, sig2
        if sig is None:
            print("  no frequency lines stand out -> no narrow-band signature (the source may be broadband: see the level scan above).")
        else:
            sc = roll(score(F, sig), 5)
            d, m0s, m1s = dprime(sc, pr)
            cv = cv_dprime(F, pr, sdref[kind], 5)
            print("  chosen: %s features, start :%02d mod %d, ON %d min, %d cycles" % (kind, ph, fan_P, fan_L, len(pr)))
            print("  d' = %.2f   cross-validated d' = %.2f   -> %s" % (d, cv, verdict(cv if not np.isnan(cv) else d)))
            fr = np.unique(np.round(hz(sig["bins"][np.argsort(sig["w"])[::-1]][:8]), 1))
            print("  signature lines (Hz):", ", ".join("%.1f" % f for f in fr))
            hi, lo = m0s + 0.6 * (m1s - m0s), m0s + 0.4 * (m1s - m0s)
            st = hysteresis(sc, hi, lo)
            segs = runs(st, 4, 2)
            for s_, e_ in segs:
                events_out.append(("fan", fmt(s_), fmt(e_ + 1), e_ - s_ + 1, float(np.nanmean(sc[s_:e_ + 1]))))
            print("  detected %d ON periods, median length %s min" %
                  (len(segs), int(np.median([e - s + 1 for s, e in segs])) if segs else "-"))
            timeline("FAN", st, valid, mloc, fmt_day)
            json.dump(dict(name="fan", kind=kind, bins=sig["bins"].tolist(), freqs=hz(sig["bins"]).tolist(),
                           w=sig["w"].tolist(), mu=sig["mu"].tolist(), sd=sig["sd"].tolist(),
                           ref=med[sig["bins"]].tolist(), thr=float(hi), thr_lo=float(lo), smooth=5,
                           dprime=float(d), cv_dprime=None if np.isnan(cv) else float(cv),
                           phase=ph, on_len=fan_L, period=fan_P),
                      open(os.path.join(a.out, "sig_fan.json"), "w"), indent=1)
            if conf > 4 and (np.isnan(cv) or cv > 0.8):
                fan_mask = ((mloc - ph) % fan_P) < fan_L + 2         # mask when learning the pump

    # ---------------- PUMP ----------------
    print("\n=== PUMP: ON during %s ===" % a.pump_windows)
    base = ~fan_mask
    pr = pump_pairs(mloc, valid, base, windows)
    if fan_mask.any():
        print("  (fan-ON minutes masked out)")
    if not pr:
        print("  no complete pump window recorded yet (need the data around 03-06 and/or 15-18 incl. 2 h before/after).")
    else:
        print("  %d window events usable" % len(pr))
        bestp = None
        for kind, F in feats.items():
            sig, t = build_sig(F, pr, sdref[kind])
            if sig is None:
                continue
            sc = roll(score(F, sig), 15)
            d = dprime(sc, pr)[0]
            cv = cv_dprime(F, pr, sdref[kind], 15)
            print("  [%s] d'=%.2f  cross-validated d'=%.2f" % (kind, d, cv))
            key = cv if not np.isnan(cv) else d * 0.5          # prefer held-out evidence
            if bestp is None or key > bestp[0]:
                bestp = (key, kind, sig, sc, d, cv)
        if bestp is None:
            print("  no frequency lines stand out -> pump not detectable from this audio.")
        else:
            _, kind, sig, sc, d, cv = bestp
            _, m0s, m1s = dprime(sc, pr)
            print("  chosen: %s features -> %s%s" % (kind, verdict(cv if not np.isnan(cv) else d),
                  "" if not np.isnan(cv) else "  (PROVISIONAL: <4 events, no held-out test; record >= 2 days)"))
            fr = np.unique(np.round(hz(sig["bins"][np.argsort(sig["w"])[::-1]][:8]), 1))
            print("  signature lines (Hz):", ", ".join("%.1f" % f for f in fr))
            hi, lo = m0s + 0.6 * (m1s - m0s), m0s + 0.4 * (m1s - m0s)
            st = hysteresis(sc, hi, lo)
            segs = runs(st, 20, 10)
            for s_, e_ in segs:
                events_out.append(("pump", fmt(s_), fmt(e_ + 1), e_ - s_ + 1, float(np.nanmean(sc[s_:e_ + 1]))))
            exp = np.zeros(T, bool)
            for h0, h1 in windows:
                exp |= ((mloc % 1440) >= h0 * 60) & ((mloc % 1440) < h1 * 60)
            ok = valid
            print("  agreement with schedule: %.1f%% of minutes" % (100 * (st == exp)[ok].mean()))
            unexp = [(s_, e_) for s_, e_ in segs if exp[s_:e_ + 1].mean() < 0.3]
            if unexp:
                print("  ON outside schedule:", ", ".join("%s-%s" % (fmt(s_), fmt(e_ + 1)) for s_, e_ in unexp))
            timeline("PUMP", st, valid, mloc, fmt_day)
            json.dump(dict(name="pump", kind=kind, bins=sig["bins"].tolist(), freqs=hz(sig["bins"]).tolist(),
                           w=sig["w"].tolist(), mu=sig["mu"].tolist(), sd=sig["sd"].tolist(),
                           ref=med[sig["bins"]].tolist(), thr=float(hi), thr_lo=float(lo), smooth=15,
                           dprime=float(d), cv_dprime=None if np.isnan(cv) else float(cv)),
                      open(os.path.join(a.out, "sig_pump.json"), "w"), indent=1)

    fn = os.path.join(a.out, "events.csv")
    with open(fn, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "start", "end", "minutes", "mean_score"])
        w.writerows(events_out)
    print("\nevents written to", fn)
    print("note: d'>=2 strong, 1-2 moderate, <1 weak. Re-run after more days of data to improve it.")


# --------------------------------------------------------------------------- #
# Live monitor (record + detect with learned signatures)
# --------------------------------------------------------------------------- #
class Live:
    def __init__(self, out):
        self.out, self.sig = out, {}
        for n in ("fan", "pump"):
            p = os.path.join(out, "sig_%s.json" % n)
            if os.path.exists(p):
                s = json.load(open(p))
                for k in ("bins", "w", "mu", "sd", "ref"):
                    s[k] = np.array(s[k])
                s.update(hist=deque(maxlen=s["smooth"]), on=False)
                self.sig[n] = s
        if not self.sig:
            sys.exit("no sig_*.json found: run 'analyze' first")
        print("loaded signatures:", ", ".join("%s (d'=%.1f)" % (n, s["dprime"]) for n, s in self.sig.items()), flush=True)

    def feed(self, ts, spec):
        ton = None
        for n, s in self.sig.items():
            if s["kind"] == "tonal":
                if ton is None:
                    ton = maxpool3(prominence(spec[None, :]))[0]
                f = ton[s["bins"]]
            else:
                f = spec[s["bins"]] - s["ref"]
            sc = float(((f - s["mu"]) / s["sd"]) @ s["w"])
            s["hist"].append(sc)
            if len(s["hist"]) < s["hist"].maxlen:
                continue
            sm = float(np.mean(s["hist"]))
            new = s["on"]
            if not s["on"] and sm > s["thr"]:
                new = True
            elif s["on"] and sm < s["thr_lo"]:
                new = False
            t = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))
            if new != s["on"]:
                s["on"] = new
                print("[%s] %s -> %s (score %.2f)" % (t, n.upper(), "ON" if new else "OFF", sm), flush=True)
                with open(os.path.join(self.out, "events_live.csv"), "a") as f:
                    f.write("%s,%s,%s,%.3f\n" % (t, n, "ON" if new else "OFF", sm))


# --------------------------------------------------------------------------- #
def cmd_probe(a):
    url = rtsp_url(a)
    print("ffprobe", masked(url))
    r = subprocess.run(["ffprobe", "-hide_banner", "-rtsp_transport", "tcp", "-show_streams",
                        "-select_streams", "a", "-of", "default=nw=1", url],
                       capture_output=True, text=True, timeout=40)
    keep = [l for l in r.stdout.splitlines() if l.split("=")[0] in
            ("codec_name", "sample_rate", "channels", "bit_rate")]
    print("\n".join(keep) or "NO AUDIO STREAM FOUND\n" + r.stderr[-400:])
    if keep:
        print("capturing 10 s ...")
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-rtsp_transport", "tcp", "-i", url, "-t", "10",
                              "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-"],
                             capture_output=True, timeout=40).stdout
        x = np.frombuffer(raw, np.int16).astype(float) / 32768
        if len(x):
            print("rms %.1f dBFS, peak %.1f dBFS, DC %.5f" %
                  (20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9), 20 * np.log10(np.abs(x).max() + 1e-9), x.mean()))
            print("(rms below about -70 dBFS => very quiet; analysis still works but needs more days)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["probe", "record", "analyze", "monitor"])
    ap.add_argument("--out", default="data")
    ap.add_argument("--user", default=os.environ.get("RTSP_USER", ""))
    ap.add_argument("--password", default=os.environ.get("RTSP_PASS", ""))
    ap.add_argument("--host", default=os.environ.get("RTSP_HOST", ""))
    ap.add_argument("--port", default=os.environ.get("RTSP_PORT", "554"))
    ap.add_argument("--path", default=os.environ.get("RTSP_PATH", ""))
    ap.add_argument("--fan-on", type=int, default=20, help="assumed minutes ON per cycle")
    ap.add_argument("--fan-period", type=int, default=0, help="assumed cycle length in minutes; 0 = find it automatically")
    ap.add_argument("--fan-auto", action="store_true", help="also run the period-free scan even when --fan-period is given")
    ap.add_argument("--pump-windows", default="3-6,15-18", help="hours, local time")
    ap.add_argument("--fmax-sig", type=int, default=1000, help="upper frequency (Hz) for the signature search")
    ap.add_argument("--utc-offset", type=int, default=None, help="minutes east of UTC, e.g. 60 for UTC+01:00")
    a = ap.parse_args()
    if a.cmd == "analyze":
        return cmd_analyze(a)
    if not a.password:
        sys.exit("set RTSP_USER, RTSP_PASS and RTSP_HOST (and RTSP_PATH)")
    if a.cmd == "probe":
        return cmd_probe(a)
    live = Live(a.out) if a.cmd == "monitor" else None
    rec = Recorder(rtsp_url(a), a.out, live.feed if live else None)
    signal.signal(signal.SIGINT, lambda *_: setattr(rec, "stop", True))
    signal.signal(signal.SIGTERM, lambda *_: setattr(rec, "stop", True))
    rec.run()


if __name__ == "__main__":
    main()

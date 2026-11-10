"""Turn a raw feat.bin (from hvac_listen.py record) into the anonymised dataset.
Keeps: per-minute 25th-percentile spectrum (dB) 20-1000 Hz, RMS, local minute-of-day.
Drops: absolute dates, host/IP/camera identifiers and any audio (spectra cannot be inverted to speech)."""
import sys, time, numpy as np
sys.path.insert(0, __file__.rsplit("analysis", 1)[0])
import hvac_listen as h
raw, out, utc_offset_min = sys.argv[1], sys.argv[2], int(sys.argv[3])
r = np.fromfile(raw, dtype=h.REC)
m = (r["ts"] // 60).astype(np.int64)
nlim = int(1000 * h.NFFT / h.SR) - h.K0
loc = m + utc_offset_min
mod = (loc % 1440).astype(np.int16)             # minute of local day
day = (loc // 1440 - (loc // 1440).min()).astype(np.int8)   # 0,1,... (no calendar date)
np.savez_compressed(out, spec_db=r["spec"][:, :nlim].astype(np.float16), rms_dbfs=r["rms"].astype(np.float16),
                    minute_of_day=mod, day_index=day, freq_hz=h.hz(np.arange(nlim)).astype(np.float32),
                    sample_rate=h.SR, nfft=h.NFFT)
print(len(r), "minutes ->", out)

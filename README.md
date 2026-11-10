# hvac-audio-detect

Detect faint, schedule-driven machines (a ventilation fan, a pool pump) from the audio of an ordinary IP camera's RTSP
stream, using per-minute spectra, schedule-supervised matched filtering and a period-free box-least-squares search.

* **[REPORT.md](REPORT.md)** - full technical report: method, maths, statistics, figures
* [README.fa.md](README.fa.md) - Persian summary
* `hvac_listen.py` - capture (`record`), learn + report (`analyze`), live detection (`monitor`), stream check (`probe`)
* `analysis/make_report.py` - reproduces every figure/number from the anonymised data
* `data/features_20-1000Hz_anon.npz` - 20.9 h of per-minute spectra (no audio, no dates, no addresses)

![overview](figures/fig1_overview.png)

Key results (20.9 h of data): the pool pump is detected (block-permutation p = 0.003, held-out d' 1.6-1.8);
a perfectly regular 81-minute ON/OFF source (ON 18.9 min) is found without assuming its period (BLS p = 0.003),
while the believed hourly 20/40 pattern is not present. See the report for caveats.

```bash
pip install -r requirements.txt
python analysis/make_report.py
```

Only point this at cameras you own or are authorised to access.

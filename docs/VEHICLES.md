# Vehicle detection from camera sound + video (design and status)

**Status: design + data collection only.** What exists today: the recorder logs a 0.5 s fine-grained audio feature stream
(`fine.bin`, see below). Everything else on this page is a plan with the reasoning and the hardware limits behind it.
Nothing here claims accuracy that has not been measured.

## 1. Goal

Detect passing / approaching / stationary vehicles (cars, motorcycles, trucks, tractors) from the camera's audio, use the
video as an extra sensor where the scene allows, and ideally estimate vehicle type and colour (make/model if feasible).
Sounds may be near or far, a vehicle may stand with the engine running or just drive by.

## 2. Why this is a different problem from the fan/pump

| | fan / pump | vehicles |
|---|---|---|
| duration | minutes to hours, stationary | 3-60 s, non-stationary (Doppler, level ramp) |
| labels | known schedule (weak, free) | none; the **video is the teacher** |
| per-minute 25th-percentile spectra | ideal (rejects transients) | useless: it deletes exactly the events we want |
| needed time resolution | 1 min | <= 0.5 s |

## 3. Hardware and scene limits (measured on the actual device)

| Item | Value | Consequence |
|---|---|---|
| Compute | Raspberry Pi 3 Model B, 4 x Cortex-A53 @ 1.2 GHz, 906 MB RAM (~700 MB free) | no big models; use gating |
| Power | `vcgencmd get_throttled` = `0x50005` (under-voltage seen, throttling active) | fix the power supply *before* adding CPU load |
| Storage | 29 GB SD card, 23 GB free | fine log is 75 MB/day, but SD wear matters: batch writes |
| Audio | G.711 mu-law, 8 kHz, mono, 64 kbit/s, mean level -55 dBFS | usable band < 4 kHz; engine harmonics and tyre noise partly visible; faint/far vehicles may sit below quantisation noise |
| Video | H.264 Main, 1280x720, 50 fps stream timing, IR camera | software decoding of all frames would eat a core: decode **key frames only** |
| Scene | balcony view over a railing; the lane is partially hidden behind a wall; vehicles will often be small, occluded or out of frame | video can confirm only a part of the events; audio must work alone |
| Night | IR mode is monochrome | colour estimation is unreliable at night |
| Single RTSP session | the camera returned HTTP-500-style errors when probed repeatedly | all processing must share **one** connection (one ffmpeg, several outputs) |

## 4. Architecture

```
camera --RTSP--> ffmpeg (one session)
                  |-- audio s16le 8 kHz --> fine features every 0.5 s  --> fine.bin          (implemented)
                  |-- video key frames (-skip_frame nokey) --> motion gate --> detector --> crops + colour (planned)
fine.bin --> event segmentation --> audio features --> classifier (trained with video labels)   (planned)
crops    --> make/model/colour review off-device (PC or a larger model)                        (planned)
```

### 4.1 Implemented: fine-grained audio log

`hvac_listen.py record` now also appends one 436-byte record every 0.5 s to `fine.bin`:
timestamp, RMS (dBFS), 6 coarse bands 0.8-3.9 kHz (dB), and 400 spectrum bins 19.5-800 Hz at 1.95 Hz resolution,
quantised to 0.5 dB (`q = clip((dB+120)*2, 0, 255)`). Cost: about 75 MB/day. This is enough for engine harmonics.

### 4.2 Planned: audio event analysis

1. **Segmentation:** event when the 0.5 s level exceeds a 10-min running median by a margin (and is not explained by the
   periodic 81-min source, which is subtracted by its schedule), with hysteresis and gap merging.
2. **Features per event:** duration; level ramp up/down asymmetry (approach vs pass vs depart); peak level (a distance
   proxy only, uncalibrated); spectral centroid trajectory (Doppler shift of engine harmonics when passing);
   **engine fundamental** f0 from a harmonic-comb search over 20-400 Hz (firing frequency = rpm/60 x cylinders/2);
   harmonic-to-noise ratio; tyre-noise band (0.8-3.9 kHz) to low-band ratio; idle stability (std of f0 over time).
3. **Expected separations** (hypotheses, to be tested): idling vs passing (stationary f0 vs time-varying level);
   motorcycle (high f0, wide rpm swings) vs car (mid f0) vs tractor/diesel truck (low f0, very strong low harmonics,
   steady rpm). Accuracy is **unknown** until labels exist.

### 4.3 Planned: video as the teacher

* One ffmpeg, key frames only (cheap), saved to RAM disk; frame-difference motion gate in the road region of interest.
* A small COCO detector (car, truck, bus, motorcycle; YOLOv8n/NanoDet class, ONNX) runs **only on gated frames**,
  at most 0.5 fps. Budget on a Pi 3: roughly 1-2 s per 320-px inference, ~150 MB RAM (onnxruntime).
* A detection overlapping an audio event in time becomes a **label** (class, colour, direction). With a few hundred
  matched events a small audio classifier (gradient boosting on the event features) can be trained and validated on held-out
  days. Cross-modal self-supervision is the only route that needs no manual labelling.
* Colour: dominant hue/saturation/value of the detector box's central body region, mapped to ~10 colour names;
  reported as unknown at night (IR) or when the box is smaller than ~40 px.

### 4.4 Make and model

Not realistic on this device and with this view:

* A Pi 3 cannot run a fine-grained vehicle model; the 720p, partly occluded, often small view limits any model.
* Public fine-grained datasets (e.g. Stanford Cars) do not cover many regional models.

Pragmatic route: keep **crops only for the best frame of each event** (stored locally, never published) and classify them
**off-device** in batches with a stronger vision model (CLIP-style zero-shot, a fine-grained CNN on a PC, or a
multimodal LLM such as Fable/Claude reviewing the crops). This uses the strong model only on a few kilobytes per event, never
on the raw stream, and it keeps the Pi inside its budget. Expect make/model to be *unknown* for many events.

## 5. Privacy

Frames and crops show a private property and people. They stay on the device / the owner's machine and are **never
committed**. The repository only ever receives derived, non-identifying numbers (event times, features, class labels, colour
names). The published data contains no images, no audio and no plates.

## 6. Roadmap

1. (done) fine-grained audio log running.
2. Collect >= 3 days of `fine.bin`; build the event segmenter and look at the events by eye.
3. Fix power supply; add key-frame capture + motion gate; save ROI crops locally.
4. Add the detector (only if CPU/RAM allow); build the labelled audio-event set.
5. Train/validate the audio classifier on held-out days; report accuracy with confidence intervals.
6. Off-device make/model/colour review of crops; evaluate against a manually checked sample.

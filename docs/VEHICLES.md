# Vehicle detection from camera sound + video (design and status)

**Status: data collection is running; no accuracy has been measured yet.** Implemented: (a) a 0.5 s fine-grained audio
feature log (`fine.bin`), (b) key-frame capture from the same RTSP session, (c) `vehicle_watch.py`, a CPU detector
(MobileNet-SSD via OpenCV DNN) that logs vehicles with a body-colour estimate. Everything else is a plan. Nothing here
claims accuracy that has not been measured.

## 0. Priorities (decided with the owner)

1. **Sound is the primary sensor.** Detection and classification of vehicles must work from audio alone (the camera sees
   only part of the road, is IR at night, and key frames arrive every ~4 s).
2. **Video is used only for validation / labelling** of audio events (did a vehicle really pass? which class? which colour?).
3. **Later: several microphones** to estimate the **direction** of a sound (see section 7).

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
| Audio | G.711 mu-law, 8 kHz, mono, 64 kbit/s, mean level -55 dBFS | usable band < 4 kHz; whether engine harmonics and tyre noise are resolvable is unmeasured; faint/far vehicles may sit below quantisation noise |
| Video | H.264 Main, 1280x720, 50 fps stream timing, IR camera | software decoding of all frames would eat a core: decode **key frames only** |
| Scene | balcony view over a railing; the lane is partially hidden behind a wall; vehicles will often be small, occluded or out of frame | video can confirm only a part of the events; audio must work alone |
| Night | IR mode is monochrome | colour estimation is unreliable at night |
| Single RTSP session | the camera returned HTTP-500-style errors when probed repeatedly | all processing must share **one** connection (one ffmpeg, several outputs) |

## 4. Architecture

```
camera --RTSP--> ffmpeg (one session)
                  |-- audio s16le 8 kHz --> fine features every 0.5 s  --> fine.bin          (implemented)
                  |-- video key frames (-skip_frame nokey, ~1 per 4 s) --> /dev/shm --> vehicle_watch.py:
                  |        motion number --> MobileNet-SSD (car/bus/motorbike/bicycle) --> detections.csv + local crops (implemented)
fine.bin --> event segmentation --> audio features --> classifier (trained with video labels)   (planned)
crops    --> make/model/colour review off-device (PC or a larger model)                        (planned)
```

### 4.1 Implemented: fine-grained audio log

`hvac_listen.py record` now also appends one 436-byte record every 0.5 s to `fine.bin`:
timestamp, RMS (dBFS), 6 coarse bands 0.8-3.9 kHz (dB), and 400 spectrum bins 19.5-800 Hz at 1.95 Hz resolution,
quantised to 0.5 dB (`q = clip((dB+120)*2, 0, 255)`). Cost: about 75 MB/day. Whether this is enough to resolve engine harmonics has **not been measured yet**.

### 4.1b Implemented: key frames and the vehicle watcher (measured on the Pi 3)

* One `ffmpeg` process, `-skip_frame nokey`: the camera sends a key frame about every 4 s; decoding them costs about 10 % of one
  core (measured: 3.95 s user CPU in 40 s). Frames are 960 px wide JPEGs (~133 kB) on a RAM disk and are deleted after processing.
* MobileNet-SSD (VOC classes: car, bus, motorbike, bicycle; no truck/tractor class) runs in **~0.71 s per frame** at 300x300 with
  2 threads. A frame-difference number gates the detector (plus every 8th frame as heartbeat).
* Output: `detections.csv` (time, class, confidence, box, moving flag, colour name, median brightness, IR flag, persons-in-frame
  count) and local crops. **People are counted but never cropped or stored.**
* Known limitations measured so far: on a static scene the detector produced a confident false positive (class "boat" on the
  railing/gate area), which is why only vehicle classes with a minimum box width are accepted and a motion flag is logged;
  one key frame per 4 s means a fast vehicle may appear in 0-2 frames (audio events are the primary clock).
* Colour: median hue of saturated pixels in the central body region mapped to 8 names, otherwise black/white/grey by brightness;
  reported as `ir` when the frame is monochrome (night).

### 4.2 Planned: audio event analysis

1. **Segmentation:** event when the 0.5 s level exceeds a running baseline by a margin, with hysteresis and gap merging. A plain
   10-min running median would create minutes-long false events at the edges of the 19-min, +2.7 dB step of the periodic source;
   that source must be modelled and removed (its schedule needs re-phasing: ~1.5 min/day drift plus 0.7 min jitter) or the
   baseline must be computed on the band that the source does not affect.
2. **Features per event:** duration; level ramp up/down asymmetry (approach vs pass vs depart); peak level (a distance
   proxy only, uncalibrated); **tracked engine-harmonic frequency** f0(t) (Doppler: delta f / f is only ~4 % at 50 km/h, i.e. +-2-4 Hz on a 100 Hz harmonic versus 1.95 Hz bins, so it must be tracked, and a spectral centroid is not a Doppler estimator because tyre noise dominates it);
   **engine fundamental** f0 from a harmonic-comb search over 20-400 Hz (firing frequency = rpm/60 x cylinders/2);
   harmonic-to-noise ratio; tyre-noise band (0.8-3.9 kHz) to low-band ratio; idle stability (std of f0 over time).
3. **Expected separations** (hypotheses, to be tested): idling vs passing (stationary f0 vs time-varying level);
   motorcycle (high f0, wide rpm swings) vs car (mid f0) vs tractor/diesel truck (low f0, very strong low harmonics,
   steady rpm). Accuracy is **unknown** until labels exist.

### 4.3 Planned: video as the teacher

* One ffmpeg, key frames only (cheap), saved to RAM disk; frame-difference motion gate in the road region of interest.
* A small detector (car, truck, bus, motorcycle; YOLOv8n/NanoDet class, ONNX) runs **only on gated frames**,
  at most 0.5 fps. Budget on a Pi 3: roughly 1-2 s per 320-px inference, ~150 MB RAM (onnxruntime).
* A detection overlapping an audio event in time becomes a **noisy label** (class, colour, direction). With a few hundred
  matched events a small audio classifier (gradient boosting on the event features) can be trained and validated on held-out
  days. Caveat: the detector's labels are noisy (it already produced a false 'boat', and has no truck/tractor class), so accuracy
  measured against the detector is **not** accuracy against the truth. A manually checked sample (a person or a stronger model
  looking at the crops) is required to calibrate the audio classifier, not only for make/model.
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

1. (done) fine-grained audio log running; key-frame capture and vehicle watcher running.
2. Collect >= 3 days of `fine.bin`; build the event segmenter and look at the events by eye.
3. Fix the power supply; tune the key-frame watcher (ROI, thresholds) and review crops locally.
4. Add the detector (only if CPU/RAM allow); build the labelled audio-event set.
5. Train/validate the audio classifier on held-out days; report accuracy with confidence intervals.
6. Off-device make/model/colour review of crops; evaluate against a manually checked sample.

## 7. Direction of arrival with several microphones (future)

The camera has one microphone, so direction cannot be estimated from it. A small array can:

* **Method:** time difference of arrival between microphone pairs, estimated with GCC-PHAT, then a bearing (2 mics: an angle
  with a front/back ambiguity; 3-4 mics: full azimuth). Wideband sounds such as tyre noise localise well; low engine tones
  localise poorly because their wavelength (3.4 m at 100 Hz) is large compared with a small array.
* **Hardware constraints (Raspberry Pi 3):** all channels must be sampled by **one clock** (a multi-channel USB audio interface
  or an I2S array such as a 4-mic board); separate USB sound cards drift against each other. Sample at >= 16 kHz. The Pi 3
  shares USB 2.0 with Ethernet, so keep the channel count and rate modest.
* **Geometry:** spacing of 10-20 cm gives usable angular resolution above ~500 Hz; larger spacing helps low frequencies but
  causes spatial aliasing at high frequencies. Mount mics away from the building's own noise sources and out of the wind.
* **Use:** approaching vs receding, left vs right of the building, and separating a vehicle from the periodic machines inside the
  building (which arrive from a fixed bearing). Bearing + time also lets the video check whether the object seen is the one heard.

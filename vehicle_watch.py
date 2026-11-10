#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vehicle_watch.py - consume the key frames that `hvac_listen.py record --video-dir DIR` writes, detect vehicles with a
small CPU detector (MobileNet-SSD through OpenCV DNN, ~0.7 s/frame on a Raspberry Pi 3), estimate the body colour and
log one CSV row per detection. Crops are kept LOCALLY only (never publish them: they show a private property).

Design for a weak device:
  * key frames only (the camera sends one every ~4 s) -> ~10 % of one core for decoding
  * a cheap frame-difference "motion" number is computed for every frame; the detector runs on frames with motion,
    plus a periodic heartbeat frame
  * people are only counted, never cropped or stored
Usage:
  python3 vehicle_watch.py --frames /dev/shm/vframes --models ~/vehicles/models --out ~/vehicles
Models: MobileNet-SSD (Caffe) deploy.prototxt + mobilenet_iter_73000.caffemodel from github.com/chuanqi305/MobileNet-SSD
"""
import argparse
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
VEHICLES = {"car", "bus", "motorbike", "bicycle"}
HUES = [(8, "red"), (20, "orange"), (35, "yellow"), (85, "green"), (100, "cyan"), (130, "blue"), (160, "purple"), (172, "pink")]


def colour_name(crop):
    """Body colour from the central region of a box: hue histogram for saturated pixels, otherwise black/grey/white."""
    h, w = crop.shape[:2]
    core = crop[int(h * .25):int(h * .8), int(w * .2):int(w * .8)]
    if core.size == 0:
        return "unknown", 0
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    sat, val, hue = hsv[..., 1], hsv[..., 2], hsv[..., 0]
    mv = int(np.median(val))
    mask = (sat > 70) & (val > 60)
    if mask.mean() < .25:
        if mv < 60:
            return "black", mv
        if mv > 190 and np.median(sat) < 40:
            return "white", mv
        return "grey/silver", mv
    hh = int(np.median(hue[mask]))
    for edge, name in HUES:
        if hh < edge:
            return name, mv
    return "red", mv


def is_ir(frame):
    """Night/IR mode delivers (almost) monochrome frames: colour is meaningless then."""
    hsv = cv2.cvtColor(cv2.resize(frame, (96, 54)), cv2.COLOR_BGR2HSV)
    return float(hsv[..., 1].mean()) < 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="/dev/shm/vframes")
    ap.add_argument("--models", default=os.path.expanduser("~/vehicles/models"))
    ap.add_argument("--out", default=os.path.expanduser("~/vehicles"))
    ap.add_argument("--conf", type=float, default=0.40)
    ap.add_argument("--min-width", type=int, default=30, help="ignore boxes narrower than this (px of the 960-px frame)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-crops", type=int, default=4000)
    a = ap.parse_args()

    cv2.setNumThreads(a.threads)
    net = cv2.dnn.readNetFromCaffe(os.path.join(a.models, "deploy.prototxt"),
                                   os.path.join(a.models, "mobilenet_iter_73000.caffemodel"))
    os.makedirs(a.frames, exist_ok=True)
    cdir = os.path.join(a.out, "crops")
    os.makedirs(cdir, exist_ok=True)
    csv_path = os.path.join(a.out, "detections.csv")
    new = not os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["ts", "class", "conf", "x", "y", "w", "h", "moving", "colour", "median_v", "ir", "persons_in_frame", "motion", "crop"])
    prev, n, nrun, last_beat = None, 0, 0, time.time()
    print("watching", a.frames, flush=True)
    while True:
        files = sorted(glob.glob(os.path.join(a.frames, "*.jpg")))
        files = [p for p in files if time.time() - os.path.getmtime(p) > 1.0]     # wait until ffmpeg closed the file
        if not files:
            time.sleep(1.0)
        for p in files:
            ts = os.path.basename(p)[:-4]
            img = cv2.imread(p)
            try:
                os.remove(p)
            except OSError:
                pass
            if img is None:
                continue
            n += 1
            g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
            motion_mask = None
            motion = 0.0
            if prev is not None:
                motion_mask = cv2.absdiff(g, prev) > 25
                motion = float(motion_mask.mean())
            prev = g
            if not (motion > 0.002 or n % 8 == 0):
                continue
            nrun += 1
            H, W = img.shape[:2]
            net.setInput(cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 0.007843, (300, 300), 127.5))
            out = net.forward()[0, 0]
            persons = int(sum(1 for d in out if d[2] > a.conf and CLASSES[int(d[1])] == "person"))
            ir = is_ir(img)
            for d in out:
                cls = CLASSES[int(d[1])]
                if d[2] < a.conf or cls not in VEHICLES:
                    continue
                x0, y0, x1, y1 = (d[3:7] * np.array([W, H, W, H])).astype(int)
                x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
                if x1 - x0 < a.min_width or y1 - y0 < 15:
                    continue
                moving = False
                if motion_mask is not None:
                    sub = motion_mask[int(y0 * 90 / H):int(y1 * 90 / H) + 1, int(x0 * 160 / W):int(x1 * 160 / W) + 1]
                    moving = bool(sub.size and sub.mean() > 0.04)
                name, mv = colour_name(img[y0:y1, x0:x1])
                pad = int(.1 * max(x1 - x0, y1 - y0))
                crop = img[max(y0 - pad, 0):min(y1 + pad, H), max(x0 - pad, 0):min(x1 + pad, W)]
                cp = os.path.join(cdir, "%s_%s_%02d.jpg" % (ts, cls, int(d[2] * 100)))
                cv2.imwrite(cp, crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
                w.writerow([ts, cls, round(float(d[2]), 2), x0, y0, x1 - x0, y1 - y0, int(moving), "ir" if ir else name,
                            mv, int(ir), persons, round(motion, 4), os.path.basename(cp)])
                f.flush()
        # housekeeping: stale frames (watcher was down) and crop cap
        for p in glob.glob(os.path.join(a.frames, "*.jpg")):
            if time.time() - os.path.getmtime(p) > 600:
                os.remove(p)
        crops = sorted(glob.glob(os.path.join(cdir, "*.jpg")))
        for p in crops[:max(0, len(crops) - a.max_crops)]:
            os.remove(p)
        if time.time() - last_beat > 300:
            print("%s frames=%d detector_runs=%d" % (time.strftime("%F %T"), n, nrun), flush=True)
            last_beat = time.time()


if __name__ == "__main__":
    main()

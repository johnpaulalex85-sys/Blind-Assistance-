"""
Check wall detection on your own photos.

    python test_wall_images.py path/to/folder_of_images [--expect wall|not_wall]

Prints, for every image, whether a wall was detected and by which method,
then a summary. Run it once with a folder of WALL photos (--expect wall) and
once with a folder of NON-wall photos (--expect not_wall) to see the real
hit rate and false-alarm rate on your camera and your places.
"""
import argparse, glob, os, sys, time
import cv2
from services.wall_service import WallDetector

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--expect", choices=["wall", "not_wall"], default=None)
args = ap.parse_args()

det = WallDetector()
if det.segmenter is not None and det.segmenter.ready:
    print("Pretrained wall segmentation: ON")
else:
    print("Pretrained wall segmentation: OFF (model not available) - texture cues only")

files = sorted(f for ext in ("jpg", "jpeg", "png", "bmp", "webp") for f in glob.glob(os.path.join(args.folder, f"*.{ext}")))
if not files:
    sys.exit("No images found.")

ok = 0
for f in files:
    img = cv2.imread(f)
    if img is None:
        continue
    img = cv2.resize(img, (640, 480))
    det.reset()
    res = []
    for _ in range(6):                      # let debouncing + background segmentation settle
        res = det.detect(img)
        time.sleep(0.3)
    got = bool(res)
    good = (args.expect is None) or (got == (args.expect == "wall"))
    ok += good
    print(f"{'WALL ' if got else 'none '} {'' if good else '<-- WRONG  '}{os.path.basename(f)}" + (f"  conf={res[0]['confidence']}" if got else ""))
if args.expect:
    print(f"\n{ok}/{len(files)} correct ({100*ok/len(files):.0f}%)")

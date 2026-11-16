#!/usr/bin/env python3
"""Run svo_pipeline on one stereo recording.

Usage:
    python svo_run.py --left left.mp4 --right right.mp4 --calib calib.json --out out/clip01
    python svo_run.py --stacked stereo.mp4 --calib calib.json --out out/clip01
    python svo_run.py ... --preset all
    python svo_run.py ... --set kf_add_inliers=130 guard_max_coast=5
    python svo_run.py --list
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PRESETS = {
    'default': {},
    'no_guard': {'guard_max_step_t_mm': 1000.0, 'guard_max_step_r_deg': 1000.0},
    'loose_guard': {'guard_max_step_t_mm': 6.6, 'guard_max_step_r_deg': 2.6},
    'early_keyframes': {'kf_add_inliers': 200},
    'recall': {'recall': True, 'recall_reloc': True},
}
NO_IMAGES = {'debug_every': 0, 'debug_max_reject_images': 0}
WEIGHTS = {'hsm_weights': 'hsm_selftrain_best.pth', 'superpoint_weights': 'superpoint_v1.pth'}


def value(s):
    if s.lower() in ('none', 'null'):
        return None
    if s.lower() in ('true', 'false'):
        return s.lower() == 'true'
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return s


def parse_set(groups):
    out = {}
    for item in [x for g in (groups or []) for x in g]:
        if '=' not in item:
            sys.exit(f'--set: expected key=value, got {item}')
        k, v = item.split('=', 1)
        out[k.strip()] = value(v.strip())
    return out


def size(s):
    try:
        w, h = (int(x) for x in s.lower().split('x'))
    except ValueError:
        sys.exit(f'bad size {s}, use WxH')
    return w, h


def video_size(path):
    import cv2
    if not Path(path).exists():
        sys.exit(f'not found: {path}')
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        sys.exit(f'cannot open {path}')
    wh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return wh


def need(module, hint):
    try:
        ok = importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        ok = False
    if not ok:
        sys.exit(f'cannot import {module}: {hint}')


def weights(a, code):
    out = {}
    for key, name in WEIGHTS.items():
        p = Path(getattr(a, key)).expanduser() if getattr(a, key) else code / name
        if not p.exists():
            sys.exit(f'{name} not found ({p})')
        out[key] = str(p)
    return out


def config(a, code):
    if bool(a.stacked) == bool(a.left or a.right) or (not a.stacked and not (a.left and a.right)):
        sys.exit('use --left and --right, or --stacked')
    if not a.calib or not a.out:
        sys.exit('--calib and --out are required')
    if not Path(a.calib).is_file():
        sys.exit(f'calibration file not found: {a.calib}')

    p = {'stereo_json': a.calib}
    if a.stacked:
        p.update(source='stacked', stacked_video=a.stacked)
        w, h = video_size(a.stacked)
        h //= 2
    else:
        p.update(source='pair', left_video=a.left, right_video=a.right)
        w, h = video_size(a.left)
        if video_size(a.right) != (w, h):
            sys.exit('left and right videos have different sizes')
    if a.crop:
        y0, y1, x0, x1 = a.crop
        if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
            sys.exit(f'crop {a.crop} is outside the {w}x{h} frame')
        p['crop'] = a.crop
        w, h = x1 - x0, y1 - y0
    if a.resize:
        w, h = size(a.resize)
        p['allow_resize'] = True
    pw, ph = size(a.proc_size) if a.proc_size else (640, int(round(640 * h / w / 8)) * 8)
    p.update(input_w=w, input_h=h, proc_w=pw, proc_h=ph, output_dir=a.out, video_start_frame=a.start,
             max_frames=a.max_frames, verbose=a.verbose, **weights(a, code))
    if a.no_images:
        p.update(NO_IMAGES)
    if a.stream:
        p['stream_poses'] = True
    p.update(parse_set(a.set))
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--left')
    ap.add_argument('--right')
    ap.add_argument('--stacked', help='single video, left eye on top')
    ap.add_argument('--calib', help='calibration json')
    ap.add_argument('--out', help='output folder')
    ap.add_argument('--crop', type=int, nargs=4, metavar=('Y0', 'Y1', 'X0', 'X1'), help='crop each eye first')
    ap.add_argument('--resize', help='resize each eye to WxH (the calibrated size)')
    ap.add_argument('--proc-size', help='processing size WxH, default 640 wide')
    ap.add_argument('--start', type=int, default=0, help='first frame')
    ap.add_argument('--max-frames', type=int)
    ap.add_argument('--code', default=str(HERE), help='folder with superpoint.py and the weights')
    ap.add_argument('--hsm-weights', dest='hsm_weights')
    ap.add_argument('--superpoint-weights', dest='superpoint_weights')
    ap.add_argument('--no-images', action='store_true', help='do not save debug images')
    ap.add_argument('--stream', action='store_true', help='write the pose csv files frame by frame')
    ap.add_argument('--verbose', type=int, default=1, choices=(0, 1, 2))
    ap.add_argument('--preset', nargs='+', default=['default'],
                    help=f"one or more of {', '.join(PRESETS)}, or all; several presets go to <out>/<preset>")
    ap.add_argument('--set', action='append', nargs='+', metavar='KEY=VALUE', help='override a pipeline setting')
    ap.add_argument('--list', action='store_true', help='list all settings and presets and exit')
    a = ap.parse_args(argv)

    code = Path(a.code).expanduser().resolve()
    sys.path[:0] = [str(HERE), str(code)]
    need('svo_pipeline', 'svo_pipeline.py must be next to svo_run.py or in --code')
    import svo_pipeline as svo

    if a.list:
        for k, v in svo.DEFAULTS.items():
            print(f'{k:30s} {v!r}')
        print()
        for k, v in PRESETS.items():
            print(f'preset {k:16s} {v}')
        return 0

    names = list(PRESETS) if a.preset == ['all'] else a.preset
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        sys.exit(f"unknown preset {', '.join(unknown)}; choose from {', '.join(PRESETS)} or all")
    base = config(a, code)
    cfgs = {}
    for n in names:
        p = {**base, **PRESETS[n], **parse_set(a.set)}
        if len(names) > 1:
            p['output_dir'] = str(Path(a.out) / n)
        try:
            cfgs[n] = svo.make_config(**p)
            svo.validate(cfgs[n])
        except (KeyError, ValueError) as e:
            sys.exit(str(e))
    need('superpoint', f'put superpoint.py in {code}')
    need('sksurgerytorch', 'pip install scikit-surgerytorch')

    models = svo.load_models(cfgs[names[0]])
    summary = []
    for n, cfg in cfgs.items():
        changed = {k: v for k, v in svo.config_diff(cfg).items()
                   if k not in ('source', 'input_w', 'input_h', 'proc_w', 'proc_h', 'video_start_frame', 'max_frames')}
        print(f"\n{n}: {cfg['input_w']}x{cfg['input_h']} -> {cfg['proc_w']}x{cfg['proc_h']}"
              + (f', changed: {changed}' if changed else ''))
        t0 = time.time()
        svo.run(cfg, models=models)
        meta = Path(cfg['output_dir']) / 'run_meta.json'
        m = json.loads(meta.read_text()) if meta.exists() else {}
        d = m.get('decisions', {})
        summary.append((n, m.get('n_frames', 0), sum(v for k, v in d.items() if k.startswith('reject_')),
                        d.get('reseed', 0), m.get('n_keyframes', 0), time.time() - t0, m.get('completed', False)))

    print(f"\n{'preset':16s} {'frames':>7s} {'held':>6s} {'reseeds':>8s} {'keyframes':>10s} {'time':>7s}")
    for n, f, h, r, k, t, _ in summary:
        print(f'{n:16s} {f:7d} {h:6d} {r:8d} {k:10d} {t:6.0f}s')
    return 0 if all(s[-1] for s in summary) else 1


if __name__ == '__main__':
    sys.exit(main())

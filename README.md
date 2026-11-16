# SVO pipeline

## Setup

Put these in one folder:

- `svo_pipeline.py`, `svo_run.py`
- `superpoint.py`, `superpoint_v1.pth`, `hsm_selftrain_best.pth`

Needs numpy, opencv-python, torch, pandas, matplotlib and scikit-surgerytorch. A CUDA GPU is needed to run in real time.

If `superpoint.py` and the weights are in another folder, pass it with `--code`.

## Running

```
python svo_run.py --left left.mp4 --right right.mp4 --calib calib.json --out out/clip01
```

For a single video with the left eye on top, use `--stacked video.mp4` instead of `--left`/`--right`. Side-by-side videos need to be split into two files first.

Useful options:

- `--start N`, `--max-frames N`: process part of the video
- `--no-images`: don't save debug images (they contain video frames)
- `--stream`: write the pose files frame by frame during the run instead of at the end
- `--crop Y0 Y1 X0 X1`, `--resize WxH`: crop and/or resize each eye before processing, if the calibration was made on cropped or resized frames
- `--set key=value`: change a setting, e.g. `--set kf_add_inliers=130`; keyframe recall is off by default and turned on with `--set recall=true` (plus `recall_reloc=true` for relocalisation after a re-seed)
- `--list`: show all settings and their defaults

The frame size is taken from the video (after cropping/resizing) and processing runs at 640 px width (1920×1080 → 640×360).

## Presets

`--preset all` runs the clip once with each preset, into `<out>/<preset>`, and prints a short comparison at the end. Single presets: `--preset no_guard`, or several, `--preset default no_guard`.

- `default`: the standard settings
- `no_guard`: no limit on the camera motion per frame
- `loose_guard`: twice the default motion limit (6.6 mm / 2.6° per frame)
- `early_keyframes`: new keyframes are made sooner (below 200 instead of 160 inliers)
- `recall`: keyframe recall and relocalisation on

## Calibration

JSON with the OpenCV stereo calibration, baseline in metres:

```json
{
  "cameraMatrixL": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "distCoeffsL": [[k1, k2, p1, p2, k3]],
  "cameraMatrixR": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "distCoeffsR": [[k1, k2, p1, p2, k3]],
  "R": [[...], [...], [...]],
  "T": [tx, ty, tz]
}
```

`R`, `T` as returned by `cv2.stereoCalibrate`. The calibration has to be made at the same resolution as the video. If the baseline is in mm, add `--set calib_unit=mm`.

## Output

- `camera_pose_recovered.csv`: pose per frame, smoothed. Use this one for the overlay.
- `camera_pose_raw.csv`: same without smoothing.
- `frame_debug.csv`: per-frame log (inliers, accept/reject and why, timing).
- `debug_frames/`: keypoint images of rejected frames and of every 25th frame.
- `run_meta.json`: settings, `R_l` and `K_rect`.

Pose columns: `rvec_model_to_camera_*` and `tvec_model_to_camera_*` are the world-to-camera pose (OpenCV convention), `cam_in_model_*` is the camera position. `prior_ok = 0` means the frame was rejected and the previous pose is repeated.
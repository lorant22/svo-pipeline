
# Poses are of the rectified left camera, world = that camera at the first frame, unit = calibration baseline unit.

import csv
import json
import math
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

DEFAULTS = {
    'source': 'pair',             # 'pair' (left_video, right_video) or 'stacked' (one video, left eye on top)
    'left_video': None, 'right_video': None, 'stacked_video': None,
    'stereo_json': None,
    'calib_unit': 'm',
    'output_dir': None,
    'hsm_weights': 'hsm_selftrain_best.pth',
    'superpoint_weights': 'superpoint_v1.pth',
    'video_start_frame': 0,
    'max_frames': None,
    'input_h': None, 'input_w': None, 'proc_h': None, 'proc_w': None,
    'crop': None,                 # [y0, y1, x0, x1] applied to each eye before the size check
    'allow_resize': False,
    'check_principal_point': True,
    'stream_poses': False,
    'max_kpts': 1800,
    'sp_max_keypoints': 1024,
    'sp_threshold': 0.005,
    'sp_nms_radius': 4,
    'match_max_desc_dist': 0.7,
    'min_inliers': 20,
    'reproj_error': 3.0,
    'iterations': 300,
    'pnp_confidence': 0.999,
    'pnp_lo_iters': 3,
    'depth_gate_min_mm': 10.0, 'depth_gate_max_mm': 350.0,
    'disparity_median_ksize': 3,
    'hsm_max_disp': 255, 'hsm_level': 1, 'hsm_scale_factor': 1,
    'map_max_keyframes': 24,
    'kf_add_inliers': 160,
    'temp_limit': 100,
    'guard_min_inliers': 20,
    'guard_max_step_t_mm': 3.3,
    'guard_max_step_r_deg': 1.3,
    'guard_max_coast': 10,
    'guard_reseed_after': 30,
    'info_accel_t_mm': 0.15,
    'info_accel_r_deg': 0.15,
    'info_gate_chi2': 100.0,
    'info_cov_scale': 3.0,
    'recall': False,
    'recall_every': 3,
    'recall_view_deg': 25.0,
    'recall_view_frac': 0.3,
    'recall_min_inliers': 80,
    'recall_min_ratio': 0.4,
    'recall_max_std_mm': 0.25,
    'recall_min_t_mm': 0.3, 'recall_min_r_deg': 0.2,
    'recall_max_t_mm': 20.0, 'recall_max_r_deg': 10.0,
    'recall_confirm_t_mm': 0.5, 'recall_confirm_r_deg': 0.3,
    'recall_cooldown': 15,
    'recall_archive_max': 400,
    'recall_reloc': False,        # needs recall
    'recall_reloc_frames': 600,
    'recall_reloc_recent': 3,
    'recall_reloc_max_t_mm': 60.0,
    'recall_reloc_max_r_deg': 30.0,
    'debug_every': 25,
    'debug_max_reject_images': 200,
    'verbose': 1,
}


def make_config(**overrides):
    cfg = dict(DEFAULTS)
    for k, v in overrides.items():
        if k not in DEFAULTS:
            raise KeyError(f"unknown setting '{k}'")
        cfg[k] = v
    return cfg


def config_diff(cfg):
    paths = ('left_video', 'right_video', 'stacked_video', 'stereo_json', 'output_dir', 'hsm_weights',
             'superpoint_weights')
    return {k: v for k, v in cfg.items() if k not in paths and DEFAULTS.get(k) != v}


def validate(cfg):
    for k in ('input_h', 'input_w', 'proc_h', 'proc_w', 'output_dir', 'stereo_json'):
        if cfg.get(k) is None:
            raise ValueError(f"'{k}' is not set")
    if cfg['source'] not in ('pair', 'stacked'):
        raise ValueError("source must be 'pair' or 'stacked'")
    if cfg['source'] == 'pair' and not (cfg['left_video'] and cfg['right_video']):
        raise ValueError("source='pair' needs left_video and right_video")
    if cfg['source'] == 'stacked' and not cfg['stacked_video']:
        raise ValueError("source='stacked' needs stacked_video")
    if cfg['calib_unit'] not in ('m', 'mm'):
        raise ValueError("calib_unit must be 'm' or 'mm'")
    if cfg['crop'] is not None and len(cfg['crop']) != 4:
        raise ValueError('crop must be [y0, y1, x0, x1]')
    if cfg['recall_reloc'] and not cfg['recall']:
        raise ValueError('recall_reloc needs recall=True')


def load_calibration(path):
    d = json.loads(Path(path).read_text(encoding='utf-8'))
    return (np.array(d['cameraMatrixL'], np.float64), np.array(d['cameraMatrixR'], np.float64),
            np.array(d['distCoeffsL'][0], np.float64), np.array(d['distCoeffsR'][0], np.float64),
            np.array(d['R'], np.float64), np.array(d['T'], np.float64).reshape(3, 1))


def build_geometry(cfg):
    Kl, Kr, dl, dr, R, T = load_calibration(cfg['stereo_json'])
    h, w, hp, wp = cfg['input_h'], cfg['input_w'], cfg['proc_h'], cfg['proc_w']
    mm = 1000.0 if cfg['calib_unit'] == 'm' else 1.0

    baseline_mm = float(np.linalg.norm(T)) * mm
    if not 1.0 <= baseline_mm <= 50.0:
        raise ValueError(f"baseline = {baseline_mm:.3f} mm with calib_unit='{cfg['calib_unit']}': "
                         f"the calibration file is probably in the other unit")
    cx, cy = Kl[0, 2], Kl[1, 2]
    if cfg['check_principal_point'] and (abs(cx - w / 2) > 0.15 * w or abs(cy - h / 2) > 0.15 * h):
        raise ValueError(f'principal point ({cx:.0f}, {cy:.0f}) is far from the centre of the {w}x{h} input: '
                         f'the calibration belongs to another image size')

    R_l, R_r, P_l, P_r, Q, _, _ = cv2.stereoRectify(Kl, dl, Kr, dr, (w, h), R, T,
                                                    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
                                                    newImageSize=(wp, hp))
    maps_l = cv2.initUndistortRectifyMap(Kl, dl, R_l, P_l, (wp, hp), cv2.CV_16SC2)
    maps_r = cv2.initUndistortRectifyMap(Kr, dr, R_r, P_r, (wp, hp), cv2.CV_16SC2)
    return {'Kl': Kl, 'R_l': R_l, 'K_rect': P_l[:3, :3].copy(), 'Q': Q,
            'maps_l': maps_l, 'maps_r': maps_r, 'mm': mm, 'baseline_mm': baseline_mm,
            'f_rect': float(P_l[0, 0]),
            'gate': (cfg['depth_gate_min_mm'] / mm, cfg['depth_gate_max_mm'] / mm),
            'proc_wh': (wp, hp)}


def rectify(img, maps):
    return cv2.remap(img, maps[0], maps[1], cv2.INTER_LINEAR)


class StereoReader:
    def __init__(self, cfg):
        self.stacked = cfg['source'] == 'stacked'
        paths = [cfg['stacked_video']] if self.stacked else [cfg['left_video'], cfg['right_video']]
        self.caps = []
        for p in paths:
            if not Path(p).exists():
                raise FileNotFoundError(p)
            cap = cv2.VideoCapture(str(p))
            if not cap.isOpened():
                raise RuntimeError(f'cannot open {p}')
            self.caps.append(cap)
        start = int(cfg['video_start_frame'])
        if start > 0:
            for cap in self.caps:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) != start:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    for _ in range(start):
                        cap.grab()

    def read(self):
        if self.stacked:
            ok, f = self.caps[0].read()
            if not ok:
                return False, None, None
            h = f.shape[0] // 2
            return True, f[:h], f[h:2 * h]
        okl, l = self.caps[0].read()
        okr, r = self.caps[1].read()
        return (okl and okr), l, r

    def release(self):
        for cap in self.caps:
            cap.release()


def load_models(cfg):
    import numpy.lib
    numpy.lib.pad = np.pad                     # scikit-surgerytorch still calls numpy.lib.pad
    from sksurgerytorch.models import high_res_stereo
    from superpoint import SuperPointFrontend
    hsm = high_res_stereo.HSMNet(max_disp=cfg['hsm_max_disp'], entropy_threshold=-1, level=cfg['hsm_level'],
                                 scale_factor=cfg['hsm_scale_factor'], weights=cfg['hsm_weights'])
    sp = SuperPointFrontend(weights_path=cfg['superpoint_weights'], device=None,
                            nms_radius=int(cfg['sp_nms_radius']), keypoint_threshold=float(cfg['sp_threshold']),
                            max_keypoints=int(cfg['sp_max_keypoints']), border=4)
    return {'hsm': hsm, 'superpoint': sp}


def compute_depth(L, R, hsm, geo, cfg):
    import torch
    with torch.no_grad():
        out = hsm.predict(L, R)
    disp_raw = np.asarray(out[0] if isinstance(out, tuple) else out, dtype=np.float32)
    if disp_raw.ndim == 3 and disp_raw.shape[0] == 1:
        disp_raw = disp_raw[0]
    elif disp_raw.ndim == 3 and disp_raw.shape[2] == 1:
        disp_raw = disp_raw[..., 0]
    if disp_raw.shape != L.shape[:2]:
        raise RuntimeError(f'HSM disparity {disp_raw.shape} != processing size {L.shape[:2]}')
    keep = disp_raw > 0
    disp = disp_raw.copy()
    if cfg['disparity_median_ksize'] and cfg['disparity_median_ksize'] > 1:
        disp = cv2.medianBlur(disp_raw, int(cfg['disparity_median_ksize']))
        disp[~keep] = 0.0
    keep &= disp > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        pts = cv2.reprojectImageTo3D(disp, geo['Q'])
    keep &= np.isfinite(pts).all(-1)
    gmin, gmax = geo['gate']
    with np.errstate(invalid='ignore'):
        keep &= (pts[..., 2] > gmin) & (pts[..., 2] < gmax)
    pts[~keep] = np.nan
    return pts, keep


def rt_to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, np.float64).reshape(3, 1))[0]
    T[:3, 3] = np.asarray(tvec, np.float64).reshape(3)
    return T


def T_to_rt(T):
    return cv2.Rodrigues(np.asarray(T[:3, :3], np.float64))[0], np.asarray(T[:3, 3], np.float64).reshape(3, 1).copy()


def centre(T_mc):
    return -T_mc[:3, :3].T @ T_mc[:3, 3]


def rot_deg(R):
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(np.asarray(R, np.float64))[0])))


def step_between(T_a, T_b, mm):
    return (float(np.linalg.norm(centre(T_b) - centre(T_a)) * mm),
            rot_deg(T_b[:3, :3] @ T_a[:3, :3].T))


def quat_wxyz(R):
    R = np.asarray(R, np.float64)
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    q = np.array(q) / np.linalg.norm(q)
    return -q if q[0] < 0 else q


def make_pose_row(frame, T_mc, prior_ok, inliers):
    R, t = T_mc[:3, :3], T_mc[:3, 3]
    c = -R.T @ t
    q = quat_wxyz(R.T)
    rv = cv2.Rodrigues(R)[0].ravel()
    row = {'frame': int(frame), 'backend': 'superpoint', 'prior_ok': int(bool(prior_ok)), 'inliers': int(inliers),
           'rvec_model_to_camera_x': rv[0], 'rvec_model_to_camera_y': rv[1], 'rvec_model_to_camera_z': rv[2],
           'tvec_model_to_camera_x': t[0], 'tvec_model_to_camera_y': t[1], 'tvec_model_to_camera_z': t[2],
           'cam_in_model_x': c[0], 'cam_in_model_y': c[1], 'cam_in_model_z': c[2],
           'quat_w': q[0], 'quat_x': q[1], 'quat_y': q[2], 'quat_z': q[3]}
    for i in range(3):
        for j in range(3):
            row[f'R_mc_{i}{j}'] = R[i, j]
    return row


def pose_cols(T_mc, mm):
    c = centre(T_mc) * mm / 1000.0
    return dict(zip(['x_m', 'y_m', 'z_m', 'qw', 'qx', 'qy', 'qz'], [*c, *quat_wxyz(T_mc[:3, :3].T)]))


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], np.float64)


def se3_exp(xi):
    rho, phi = np.asarray(xi[:3], np.float64), np.asarray(xi[3:], np.float64)
    th, K, I = np.linalg.norm(phi), _skew(phi), np.eye(3)
    if th < 1e-8:
        R, V = I + K + 0.5 * K @ K, I + 0.5 * K + K @ K / 6.0
    else:
        A, B = math.sin(th) / th, (1 - math.cos(th)) / th ** 2
        R, V = cv2.Rodrigues(phi)[0], I + B * K + (1 - A) / th ** 2 * K @ K
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, V @ rho
    return T


def se3_log(T):
    phi = cv2.Rodrigues(np.asarray(T[:3, :3], np.float64))[0].ravel()
    th, K, I = np.linalg.norm(phi), _skew(phi), np.eye(3)
    if th < 1e-8:
        V_inv = I - 0.5 * K + K @ K / 12.0
    else:
        V_inv = I - 0.5 * K + (1 / th ** 2 - (1 + math.cos(th)) / (2 * math.sin(th)) / th) * K @ K
    return np.hstack((V_inv @ T[:3, 3], phi))


class InfoFilter:
    def __init__(self, accel_t, accel_r, gate, cov_scale=1.0):
        self.Q = np.diag([accel_r ** 2] * 3 + [accel_t ** 2] * 3)
        self.gate = gate
        self.k2 = cov_scale ** 2
        self.prev = None

    @staticmethod
    def _vec(T):
        rv, tv = T_to_rt(T)
        return np.r_[rv.ravel(), tv.ravel()]

    def update(self, T_meas, cov, reset=False):
        z = self._vec(T_meas)
        if reset or cov is None or self.prev is None or len(self.prev) < 2:
            self.prev = [z] if (reset or self.prev is None) else self.prev[-1:] + [z]
            return T_meas.copy()
        p0, p1 = self.prev
        pred = p1 + (p1 - p0)
        cov = cov * self.k2
        S = cov + self.Q
        d = pred - z
        if float(d @ np.linalg.solve(S, d)) > self.gate:
            self.prev = [z]
            return T_meas.copy()
        G = cov @ np.linalg.inv(S)
        x = z + G @ d
        self.prev = [p1, x]
        return rt_to_T(x[:3], x[3:])


class KeyframePool:
    def __init__(self, max_keyframes, archive_max=0):
        self.max_kf = int(max_keyframes)
        self.next_id = 0
        self.version = 0
        self._dev_cache = None
        self.kf_pose = {}
        self.kf_frame = {}
        self.archive = {}
        self.archive_max = int(archive_max)
        self.pts = np.zeros((0, 3), np.float32)
        self.des = None
        self.owner = np.zeros(0, np.int64)
        self.kf_ids = deque()

    def des_tensor(self, device):
        import torch
        c = self._dev_cache
        if c is None or c[0] != self.version or c[1] != str(device):
            t = torch.from_numpy(np.ascontiguousarray(self.des, dtype=np.float32)).to(device)
            self._dev_cache = (self.version, str(device), t, (t * t).sum(1))
        return self._dev_cache[2], self._dev_cache[3]

    def n_points(self):
        return len(self.pts)

    def n_keyframes(self):
        return len(self.kf_ids)

    def _keep(self, keep):
        self.pts, self.des, self.owner = self.pts[keep], self.des[keep], self.owner[keep]
        self.version += 1

    def add(self, pts, des, T_mc, frame):
        kid = self.next_id
        self.next_id += 1
        self.kf_pose[kid] = np.asarray(T_mc, np.float64).copy()
        self.kf_frame[kid] = frame
        self.pts = np.vstack([self.pts, pts.astype(np.float32)])
        self.des = des.astype(np.float32) if self.des is None else np.vstack([self.des, des.astype(np.float32)])
        self.owner = np.concatenate([self.owner, np.full(len(pts), kid)])
        self.kf_ids.append(kid)
        self.version += 1
        self._evict_oldest()
        return len(pts)

    def _evict_oldest(self):
        while len(self.kf_ids) > self.max_kf:
            old = self.kf_ids.popleft()
            m = self.owner == old
            self._archive(old, m)
            self._keep(~m)

    def _archive(self, kid, mask):
        if self.archive_max <= 0 or kid not in self.kf_pose or not mask.any():
            return
        T = self.kf_pose[kid]
        zc = self.pts[mask].astype(np.float64) @ T[:3, :3].T + T[:3, 3]
        self.archive[kid] = {'pts': self.pts[mask].copy(), 'des': self.des[mask].copy(), 'T': T,
                             'frame': self.kf_frame.get(kid), 'depth': float(np.median(zc[:, 2]))}
        while len(self.archive) > self.archive_max:
            del self.archive[min(self.archive)]

    def recall_candidates(self, T_mc, cfg, max_n=1):
        if not self.archive:
            return []
        Rc, tc = T_mc[:3, :3], T_mc[:3, 3]
        zc = Rc[2]
        cos_max = math.cos(math.radians(cfg['recall_view_deg']))
        fx = cfg['recall_view_frac']
        out = []
        for kid, a in self.archive.items():
            R, t = a['T'][:3, :3], a['T'][:3, 3]
            ca = float(R[2] @ zc)
            if ca < cos_max:
                continue
            P = -R.T @ t + R[2] * a['depth']
            pc = Rc @ P + tc
            if pc[2] <= 0.6 * a['depth'] or pc[2] >= 1.6 * a['depth']:
                continue
            if abs(pc[0] / pc[2]) < fx and abs(pc[1] / pc[2]) < 0.8 * fx:
                out.append((-ca, kid))
        return [kid for _, kid in sorted(out)[:max_n]]

    def archive_all(self):
        for kid in list(self.kf_ids):
            self._archive(kid, self.owner == kid)
        return self.archive

    def _move(self, kid, W):
        m = self.owner == kid
        if m.any():
            self.pts[m] = (self.pts[m].astype(np.float64) @ W[:3, :3].T + W[:3, 3]).astype(np.float32)
        if kid in self.archive:
            a = self.archive[kid]
            a['pts'] = (a['pts'].astype(np.float64) @ W[:3, :3].T + W[:3, 3]).astype(np.float32)
            a['T'] = self.kf_pose[kid]

    def apply_offset(self, W, first_kid):
        for kid in list(self.kf_pose):
            if kid < first_kid:
                continue
            self.kf_pose[kid] = self.kf_pose[kid] @ np.linalg.inv(W)
            self._move(kid, W)
        self.version += 1

    def apply_correction(self, W, from_kid):
        # the correction grows linearly over the keyframes made since from_kid
        span = float(self.next_id - from_kid)
        xi = se3_log(W)
        for kid in list(self.kf_pose):
            if kid <= from_kid:
                continue
            Wk = se3_exp(xi * (kid - from_kid) / span)
            self.kf_pose[kid] = self.kf_pose[kid] @ np.linalg.inv(Wk)
            self._move(kid, Wk)
        self.version += 1

    def reinsert(self, kid):
        a = self.archive.pop(kid)
        n = len(a['pts'])
        self.pts = np.vstack([self.pts, a['pts']])
        self.des = a['des'].copy() if self.des is None else np.vstack([self.des, a['des']])
        self.owner = np.concatenate([self.owner, np.full(n, kid)])
        self.kf_ids.append(kid)
        self.version += 1
        self._evict_oldest()


def extract_features(img, superpoint, max_kpts):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    kp, des, _ = superpoint.extract(gray, max_keypoints=max_kpts)
    if kp is None or des is None or len(kp) == 0:
        return np.zeros((0, 2), np.float32), None
    return np.float32(kp), np.asarray(des, np.float32)


def match_bf(des, pool_des):
    matches = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True).match(des, pool_des)
    q = np.fromiter((x.queryIdx for x in matches), int, count=len(matches))
    m = np.fromiter((x.trainIdx for x in matches), int, count=len(matches))
    d = np.fromiter((x.distance for x in matches), float, count=len(matches))
    return q, m, d


MNN_SHORTLIST = 8


def _exact_best(A, B, cand):
    import torch
    order = torch.sort(cand, dim=1).values
    d = torch.linalg.vector_norm(A[:, None, :] - B[order], dim=2)
    k = d.argmin(1)
    r = torch.arange(len(A), device=A.device)
    return order[r, k], d[r, k]


def match_pool(des, pool):
    # matmul distances only shortlist candidates; the direct distance decides, as in cv2.BFMatcher
    import torch
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    P, p2 = pool.des_tensor(device)
    with torch.no_grad():
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            Q = torch.from_numpy(np.ascontiguousarray(des, dtype=np.float32)).to(device)
            d2 = (Q * Q).sum(1, keepdim=True) + p2[None, :] - 2.0 * (Q @ P.T)
            k = min(MNN_SHORTLIST, d2.shape[1])
            j, dist = _exact_best(Q, P, torch.topk(d2, k, dim=1, largest=False).indices)
            J = torch.unique(j)
            kq = min(MNN_SHORTLIST, d2.shape[0])
            cq = torch.topk(d2[:, J], kq, dim=0, largest=False).indices.T
            best_q, _ = _exact_best(P[J], Q, cq)
            back = torch.full((P.shape[0],), -1, dtype=torch.long, device=device)
            back[J] = best_q
            q = torch.nonzero(back[j] == torch.arange(len(Q), device=device)).flatten()
            m, dist = j[q], dist[q]
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        return q.cpu().numpy().astype(int), m.cpu().numpy().astype(int), dist.cpu().numpy().astype(float)


def local_optimisation(obj, img, K, rvec, tvec, inl, thr, n_iter):
    inl = np.asarray(inl).ravel()
    for _ in range(int(n_iter)):
        if len(inl) < 6:
            break
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
        except cv2.error:
            break
        proj = cv2.projectPoints(obj, rvec, tvec, K, None)[0].reshape(-1, 2)
        new = np.flatnonzero(np.linalg.norm(proj - img.reshape(-1, 2), axis=1) < thr)
        if len(new) < 6 or np.array_equal(new, inl):
            break
        inl = new
    if len(inl) >= 6:
        try:
            rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
        except cv2.error:
            pass
    return rvec, tvec, inl


def solve_against(kp, des, pts, pts_des, geo, cfg):
    if des is None or len(kp) < cfg['min_inliers'] or len(pts) < cfg['min_inliers']:
        return None
    q, m, dist = match_bf(des, pts_des)
    if cfg['match_max_desc_dist'] > 0:
        keep = dist <= cfg['match_max_desc_dist']
        q, m = q[keep], m[keep]
    if len(q) < cfg['min_inliers']:
        return None
    K = geo['K_rect']
    obj = pts[m].astype(np.float64).reshape(-1, 1, 3)
    img = kp[q].astype(np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=cfg['iterations'],
                                             reprojectionError=cfg['reproj_error'], confidence=cfg['pnp_confidence'],
                                             flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < cfg['min_inliers']:
        return None
    rvec, tvec, inl = local_optimisation(obj, img, K, rvec, tvec, inl.ravel(), cfg['reproj_error'], cfg['pnp_lo_iters'])
    if len(inl) < cfg['min_inliers']:
        return None
    proj, J = cv2.projectPoints(obj[inl], rvec, tvec, K, None)
    r = (proj.reshape(-1, 2) - img.reshape(-1, 2)[inl]).ravel()
    sigma = max(1.4826 * float(np.median(np.abs(r))), 0.1)
    try:
        cov = np.linalg.inv(J[:, :6].T @ J[:, :6] + 1e-12 * np.eye(6)) * sigma ** 2
        std_t = float(np.linalg.norm(np.sqrt(np.abs(np.diag(cov)[3:]))) * geo['mm'])
    except np.linalg.LinAlgError:
        std_t = float('inf')
    return {'T': rt_to_T(rvec, tvec), 'n_inliers': int(len(inl)), 'ratio': float(len(inl)) / len(q), 'std_t_mm': std_t}


def lift_keypoints(kp, des, pts, keep, T_mc):
    h, w = keep.shape
    xs = np.clip(kp[:, 0].astype(int), 0, w - 1)
    ys = np.clip(kp[:, 1].astype(int), 0, h - 1)
    ok = keep[ys, xs]
    pc = pts[ys[ok], xs[ok]].astype(np.float64)
    T_cm = np.linalg.inv(T_mc)
    world = pc @ T_cm[:3, :3].T + T_cm[:3, 3]
    return world.astype(np.float32), (des[ok] if des is not None else None)


def localize(kp, des, pool, geo, cfg):
    nan = float('nan')
    res = {'ok': False, 'reason': 'ok', 'T': None, 'n_matches': 0, 'n_inliers': 0, 'ratio': 0.0,
           'reproj_med_px': nan, 'q': None, 'inl': None, 'proj': None, 'cov': None, 'std_t_mm': nan,
           'std_r_deg': nan, 't_match': 0.0, 't_pnp': 0.0}
    K = geo['K_rect']
    mn = cfg['min_inliers']
    if des is None or len(kp) < mn:
        res['reason'] = 'no_features'
        return res
    if pool.n_points() < mn:
        res['reason'] = 'map_empty'
        return res

    t0 = time.perf_counter()
    q, m, dist = match_pool(des, pool)
    if cfg['match_max_desc_dist'] > 0 and len(q):
        keep = dist <= cfg['match_max_desc_dist']
        q, m = q[keep], m[keep]
    res['t_match'] = time.perf_counter() - t0
    res['n_matches'], res['q'] = len(q), q
    if len(q) < mn:
        res['reason'] = 'few_matches'
        return res

    t0 = time.perf_counter()
    obj = pool.pts[m].astype(np.float64).reshape(-1, 1, 3)
    img = kp[q].astype(np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=cfg['iterations'],
                                             reprojectionError=cfg['reproj_error'], confidence=cfg['pnp_confidence'],
                                             flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok or inl is None or len(inl) < 4:
        res['reason'] = 'pnp_fail'
        res['t_pnp'] = time.perf_counter() - t0
        return res
    rvec, tvec, inl = local_optimisation(obj, img, K, rvec, tvec, inl.ravel(), cfg['reproj_error'], cfg['pnp_lo_iters'])
    res['t_pnp'] = time.perf_counter() - t0

    proj = cv2.projectPoints(obj, rvec, tvec, K, None)[0].reshape(-1, 2)
    mask = np.zeros(len(q), bool)
    mask[inl] = True
    err = np.linalg.norm(proj - img.reshape(-1, 2), axis=1)
    J = cv2.projectPoints(obj[inl], rvec, tvec, K, None)[1][:, :6]
    r = (proj[inl] - img.reshape(-1, 2)[inl]).ravel()
    sigma = max(1.4826 * float(np.median(np.abs(r))), 0.1)
    try:
        cov = np.linalg.inv(J.T @ J + 1e-12 * np.eye(6)) * sigma ** 2
    except np.linalg.LinAlgError:
        cov = None
    if cov is not None:
        sd = np.sqrt(np.abs(np.diag(cov)))
        res['std_t_mm'] = float(np.linalg.norm(sd[3:]) * geo['mm'])
        res['std_r_deg'] = float(np.degrees(np.linalg.norm(sd[:3])))
    res.update({'T': rt_to_T(rvec, tvec), 'cov': cov, 'inl': mask, 'proj': proj, 'n_inliers': int(len(inl)),
                'ratio': float(len(inl)) / len(q), 'reproj_med_px': float(np.median(err[inl]))})
    if len(inl) < mn:
        res['reason'] = 'few_inliers'
        return res
    res['ok'] = True
    return res


def guard_decide(loc, T_est, n_coasted, prev_cand, cfg, mm):
    # step limits scale with the frames since the last accepted pose; two agreeing rejected
    # candidates re-lock; after guard_max_coast rejected frames the next candidate is accepted
    info = {'step_t_mm': float('nan'), 'step_r_deg': float('nan'), 'limit_t_mm': float('nan'),
            'limit_r_deg': float('nan'), 'passes_floor': False}
    if not loc['ok']:
        return False, 'no_pnp', info
    k = n_coasted + 1
    lim_t, lim_r = cfg['guard_max_step_t_mm'] * k, cfg['guard_max_step_r_deg'] * k
    st, sr = step_between(T_est, loc['T'], mm)
    info.update(step_t_mm=st, step_r_deg=sr, limit_t_mm=lim_t, limit_r_deg=lim_r)
    if loc['n_inliers'] < cfg['guard_min_inliers']:
        return False, 'min_inliers', info
    info['passes_floor'] = True
    if st > lim_t:
        reason = 'step_t'
    elif sr > lim_r:
        reason = 'step_r'
    else:
        return True, 'ok', info
    if prev_cand is not None:
        pt, pr = step_between(prev_cand, loc['T'], mm)
        if pt <= cfg['guard_max_step_t_mm'] and pr <= cfg['guard_max_step_r_deg']:
            return True, 'relock', info
    if n_coasted >= cfg['guard_max_coast']:
        return True, 'forced', info
    return False, reason, info


def _label(img, lines, y0=18):
    for i, t in enumerate(lines):
        y = y0 + 18 * i
        cv2.putText(img, t, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, t, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    return img


def save_keypoint_image(path, img, kp, loc, header, residual_scale=5.0):
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    matched = np.zeros(len(kp), bool)
    if loc['q'] is not None:
        matched[loc['q']] = True
    for x, y in kp[~matched]:
        cv2.circle(out, (int(x), int(y)), 1, (150, 150, 150), -1)
    if loc['q'] is not None and loc['proj'] is not None:
        for i, qi in enumerate(loc['q']):
            p = kp[qi]
            col = (60, 220, 60) if loc['inl'][i] else (40, 40, 230)
            r = (loc['proj'][i] - p) * residual_scale
            n = np.linalg.norm(r)
            if n > 20:
                r = r / n * 20
            e = p + r
            if np.isfinite(e).all():
                cv2.line(out, (int(p[0]), int(p[1])), (int(e[0]), int(e[1])), col, 1, cv2.LINE_AA)
            cv2.circle(out, (int(p[0]), int(p[1])), 2, col, -1)
    elif loc['q'] is not None:
        for qi in loc['q']:
            cv2.circle(out, (int(kp[qi][0]), int(kp[qi][1])), 2, (0, 200, 255), -1)
    _label(out, header + [f'green inlier, red outlier, grey unmatched; residual x{residual_scale:g} (max 20 px)'])
    cv2.imwrite(str(path), out, [cv2.IMWRITE_JPEG_QUALITY, 88])


FRAME_FIELDS = (['frame', 'decision', 'reason', 'loc_reason', 'n_kp', 'n_matches', 'n_inliers', 'inlier_ratio',
                 'reproj_med_px', 'std_t_mm', 'std_r_deg', 'step_t_mm', 'step_r_deg', 'limit_t_mm', 'limit_r_deg',
                 'n_coasted', 'kf_added', 'map_kfs', 'map_pts', 'recall_kf', 'recall_inliers', 'recall_applied',
                 'recall_reloc']
                + [f'est_{k}' for k in ('x_m', 'y_m', 'z_m', 'qw', 'qx', 'qy', 'qz')]
                + ['t_read_ms', 't_rect_ms', 't_feat_ms', 't_match_ms', 't_pnp_ms', 't_depth_ms', 't_total_ms'])


class _CsvLog:
    def __init__(self, path, fields):
        self.f = open(path, 'w', newline='')
        self.w = csv.DictWriter(self.f, fieldnames=fields, extrasaction='ignore')
        self.w.writeheader()
        self.n = 0

    def write(self, row):
        self.w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in row.items()})
        self.n += 1
        if self.n % 20 == 0:
            self.f.flush()

    def close(self):
        self.f.close()


class _PoseStream:
    def __init__(self, out):
        fields = list(make_pose_row(0, np.eye(4), True, 0))
        self.files = [open(out / name, 'w', newline='') for name in ('camera_pose_recovered.csv', 'camera_pose_raw.csv')]
        self.writers = [csv.DictWriter(f, fieldnames=fields) for f in self.files]
        for w, f in zip(self.writers, self.files):
            w.writeheader()
            f.flush()

    def write(self, row, row_raw):
        for w, f, r in zip(self.writers, self.files, (row, row_raw)):
            w.writerow({k: (float(v) if isinstance(v, np.floating) else v) for k, v in r.items()})
            f.flush()

    def close(self):
        for f in self.files:
            f.close()


def _times(t_read, t_rect, t_feat, t_match, t_pnp, t_depth, t_total):
    return {k: 1000.0 * v for k, v in zip(['t_read_ms', 't_rect_ms', 't_feat_ms', 't_match_ms', 't_pnp_ms',
                                            't_depth_ms', 't_total_ms'],
                                           [t_read, t_rect, t_feat, t_match, t_pnp, t_depth, t_total])}


def run(cfg, models=None):
    validate(cfg)
    t_wall = time.time()
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'debug_frames').mkdir(exist_ok=True)
    for stale in ('camera_pose_recovered.csv', 'camera_pose_raw.csv', 'run_meta.json', 'error.txt'):
        if (out / stale).exists():
            (out / stale).unlink()

    geo = build_geometry(cfg)
    mm = geo['mm']
    if models is None:
        models = load_models(cfg)
    hsm, sp = models['hsm'], models['superpoint']
    reader = StereoReader(cfg)
    verbose = cfg['verbose']
    crop = cfg['crop']
    print(f"[svo] input {cfg['input_w']}x{cfg['input_h']} -> proc {cfg['proc_w']}x{cfg['proc_h']} | "
          f"baseline {geo['baseline_mm']:.3f} mm | f_rect {geo['f_rect']:.1f} px | out {out}")

    flog = _CsvLog(out / 'frame_debug.csv', FRAME_FIELDS)
    stream = _PoseStream(out) if cfg['stream_poses'] else None
    archive_max = int(cfg['recall_archive_max']) if cfg['recall'] else 0
    pool = KeyframePool(cfg['map_max_keyframes'], archive_max)
    recall_state = {'pending': None, 'cooldown': 0, 'reloc_kid': None, 'reloc_until': -1, 'reloc_recent': []}
    info_filter = InfoFilter(cfg['info_accel_t_mm'] / mm, np.radians(cfg['info_accel_r_deg']), cfg['info_gate_chi2'],
                             cfg['info_cov_scale'])

    T_est = np.eye(4)
    output_T = np.eye(4)
    n_coasted = 0
    prev_cand = None
    frames_since_kf = 0
    n_kf = n_reject_img = 0
    rows, rows_raw, decisions, timing = [], [], Counter(), {}
    seed = {}
    completed = False
    error = None
    frame = int(cfg['video_start_frame']) - 1
    processed = 0

    def emit(row, row_raw):
        rows.append(row)
        rows_raw.append(row_raw)
        if stream is not None:
            stream.write(row, row_raw)

    def add_keyframe(kind, frame, Lr, Rr, kp, des, T_mc, target=None):
        nonlocal n_kf
        target = pool if target is None else target
        t0 = time.perf_counter()
        pts, keep = compute_depth(Lr, Rr(), hsm, geo, cfg)
        t_depth = time.perf_counter() - t0
        world, des_ok = lift_keypoints(kp, des, pts, keep, T_mc)
        if len(world) < cfg['min_inliers'] and kind != 'kf':
            return None, keep, pts, t_depth
        n_new = target.add(world, des_ok, T_mc, frame)
        n_kf += 1
        return n_new, keep, pts, t_depth

    try:
        while True:
            if cfg['max_frames'] is not None and processed >= cfg['max_frames']:
                break
            t_start = time.perf_counter()
            ok, L, R = reader.read()
            if not ok:
                break
            frame += 1
            processed += 1
            if crop is not None:
                y0, y1, x0, x1 = (int(v) for v in crop)
                L, R = L[y0:y1, x0:x1], R[y0:y1, x0:x1]
            if L.shape[:2] != (cfg['input_h'], cfg['input_w']):
                if not cfg['allow_resize']:
                    raise ValueError(f"frame {frame} is {L.shape[1]}x{L.shape[0]} but the input size is "
                                     f"{cfg['input_w']}x{cfg['input_h']}")
                L = cv2.resize(L, (cfg['input_w'], cfg['input_h']))
                R = cv2.resize(R, (cfg['input_w'], cfg['input_h']))
            t_read = time.perf_counter() - t_start
            t0 = time.perf_counter()
            Lr = rectify(L, geo['maps_l'])
            cache = {}

            def Rr():
                if 'r' not in cache:
                    cache['r'] = rectify(R, geo['maps_r'])
                return cache['r']
            t_rect = time.perf_counter() - t0
            t0 = time.perf_counter()
            kp, des = extract_features(Lr, sp, cfg['max_kpts'])
            t_feat = time.perf_counter() - t0
            t_depth = 0.0
            fd = {'frame': frame, 'n_kp': len(kp), 'kf_added': 0}

            if pool.n_points() == 0:
                n_new, keep, pts, td = add_keyframe('seed', frame, Lr, Rr, kp, des, T_est)
                t_depth += td
                kept = 100.0 * float(keep.mean())
                med = float(np.median(pts[keep, 2]) * mm) if keep.any() else float('nan')
                if n_new is not None:
                    fd.update(decision='seed', reason='seed', kf_added=1)
                    emit(make_pose_row(frame, output_T, True, 0), make_pose_row(frame, T_est, True, 0))
                    seed.setdefault('frame', frame)
                    seed.setdefault('median_depth_mm', round(med, 2))
                    print(f'[{frame:05d}] SEED {n_new} pts | median depth {med:.1f} mm | {kept:.0f}% of pixels kept')
                else:
                    fd.update(decision='seed_fail', reason='seed')
                    print(f'[{frame:05d}] seed skipped: {len(kp)} kp, {kept:.0f}% depth kept')
                decisions[fd['decision']] += 1
                fd.update(map_kfs=pool.n_keyframes(), map_pts=pool.n_points(), n_coasted=n_coasted,
                          **{f'est_{k}': v for k, v in pose_cols(T_est, mm).items()})
                fd.update(_times(t_read, t_rect, t_feat, 0, 0, t_depth, time.perf_counter() - t_start))
                flog.write(fd)
                timing.setdefault('t_total_ms', []).append(fd['t_total_ms'])
                continue

            loc = localize(kp, des, pool, geo, cfg)
            accept, reason, ginfo = guard_decide(loc, T_est, n_coasted, prev_cand, cfg, mm)
            fd.update(loc_reason=loc['reason'], n_matches=loc['n_matches'], n_inliers=loc['n_inliers'],
                      inlier_ratio=loc['ratio'], reproj_med_px=loc['reproj_med_px'], std_t_mm=loc['std_t_mm'],
                      std_r_deg=loc['std_r_deg'], **{k: v for k, v in ginfo.items() if k != 'passes_floor'})

            if accept:
                T_est = loc['T']
                output_T = info_filter.update(T_est, loc['cov'], reset=n_coasted > 0)
                n_coasted = 0
                prev_cand = None
                decision = 'accept'
            else:
                n_coasted += 1
                prev_cand = loc['T'] if ginfo['passes_floor'] else None
                decision = 'reject'

            if not accept and n_coasted >= cfg['guard_reseed_after'] and loc['reason'] != 'no_features':
                old_pts = pool.n_points()
                new_pool = KeyframePool(cfg['map_max_keyframes'], archive_max)
                new_pool.next_id = pool.next_id
                n_new, _, _, td = add_keyframe('reseed', frame, Lr, Rr, kp, des, T_est, target=new_pool)
                t_depth += td
                if n_new is not None:
                    if cfg['recall'] and cfg['recall_reloc']:
                        recent = list(pool.kf_ids)[-int(cfg['recall_reloc_recent']):]
                        new_pool.archive = pool.archive_all()
                        new_pool.kf_pose.update({k: v for k, v in pool.kf_pose.items() if k in new_pool.archive})
                        new_pool.kf_frame.update({k: v for k, v in pool.kf_frame.items() if k in new_pool.archive})
                        recall_state.update(reloc_kid=new_pool.kf_ids[0],
                                            reloc_until=frame + int(cfg['recall_reloc_frames']),
                                            reloc_recent=recent[::-1], cooldown=0)
                    pool = new_pool
                    recall_state['pending'] = None
                    decision, reason = 'reseed', 'reseed'
                    fd['kf_added'] = 1
                    n_coasted, prev_cand, frames_since_kf = 0, None, 0
                    print(f'[{frame:05d}] RESEED at held pose: map {old_pts} -> {pool.n_points()} pts')

            if cfg['recall'] and accept and decision == 'accept':
                if recall_state['cooldown'] > 0:
                    recall_state['cooldown'] -= 1
                elif recall_state['pending'] is not None or processed % int(cfg['recall_every']) == 0:
                    T_est, output_T = _recall(kp, des, pool, geo, cfg, recall_state, T_est, output_T, info_filter,
                                              frame, fd, decisions, verbose, mm)

            frames_since_kf += 1
            if accept and (loc['n_inliers'] < cfg['kf_add_inliers'] or frames_since_kf >= cfg['temp_limit']):
                n_new, _, _, td = add_keyframe('kf', frame, Lr, Rr, kp, des, T_est)
                t_depth += td
                fd['kf_added'] = 1
                frames_since_kf = 0
                if verbose >= 2 or (verbose and reason != 'ok'):
                    print(f"      +KF kfs={pool.n_keyframes()} pts={pool.n_points()} inl={loc['n_inliers']}")

            decisions[reason if decision != 'reject' else 'reject_' + reason] += 1
            fd.update(decision=decision, reason=reason, n_coasted=n_coasted, map_kfs=pool.n_keyframes(),
                      map_pts=pool.n_points(), **{f'est_{k}': v for k, v in pose_cols(T_est, mm).items()})
            emit(make_pose_row(frame, output_T, accept, loc['n_inliers']),
                 make_pose_row(frame, T_est, accept, loc['n_inliers']))
            fd.update(_times(t_read, t_rect, t_feat, loc['t_match'], loc['t_pnp'], t_depth,
                             time.perf_counter() - t_start))
            flog.write(fd)
            timing.setdefault('t_total_ms', []).append(fd['t_total_ms'])

            is_reject = decision == 'reject'
            periodic = cfg['debug_every'] and (processed - 1) % cfg['debug_every'] == 0
            if (is_reject and n_reject_img < cfg['debug_max_reject_images']) or periodic or reason in ('relock', 'forced'):
                n_reject_img += int(is_reject)
                save_keypoint_image(
                    out / 'debug_frames' / f'f_{frame:06d}_{decision}_{reason}.jpg', Lr, kp, loc,
                    [f"frame {frame}  {decision} ({reason})  loc: {loc['reason']}",
                     f"matches {loc['n_matches']}  inliers {loc['n_inliers']}  ratio {loc['ratio']:.2f}  "
                     f"reproj {loc['reproj_med_px']:.2f}px",
                     f"step {ginfo['step_t_mm']:.3f} mm / {ginfo['step_r_deg']:.3f} deg  "
                     f"(limit {ginfo['limit_t_mm']:.2f} / {ginfo['limit_r_deg']:.2f})  coasted {n_coasted}"])
            if verbose >= 2 or (verbose and (decision != 'accept' or reason != 'ok')) or (verbose and processed % 50 == 0):
                print(f"[{frame:05d}] {decision:6s} {reason:11s} inl={loc['n_inliers']:4d}/{loc['n_matches']:<4d} "
                      f"ratio={loc['ratio']:.2f} dt={ginfo['step_t_mm']:.3f}mm dr={ginfo['step_r_deg']:.3f}deg "
                      f"coast={n_coasted} kfs={pool.n_keyframes()} pts={pool.n_points()} "
                      f"{fd['t_total_ms']:.0f}ms")
        completed = True
    except KeyboardInterrupt:
        print('\n[INFO] interrupted: saving what was tracked so far')
    except Exception as e:
        error = f'{type(e).__name__}: {e}'
        import traceback
        (out / 'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        reader.release()
        if stream is not None:
            stream.close()
        flog.close()
        _finish(out, cfg, geo, seed, rows, rows_raw, decisions, timing, completed, error, time.time() - t_wall, n_kf)
    return {'output_dir': str(out), 'frames': len(rows), 'decisions': dict(decisions), 'completed': completed}


def _recall(kp, des, pool, geo, cfg, state, T_est, output_T, info_filter, frame, fd, decisions, verbose, mm):
    # an archived keyframe that sees the same tissue again is used to remove the drift since it was made;
    # a correction needs two consecutive agreeing frames
    pend = state['pending']
    reloc = state['reloc_kid'] is not None and frame <= state['reloc_until']
    if not reloc:
        state['reloc_kid'] = None
    cands = [pend[0]] if pend is not None else pool.recall_candidates(T_est, cfg)
    if reloc and pend is None:
        cands = cands + [k for k in state['reloc_recent'] if k not in cands]
    max_t = cfg['recall_reloc_max_t_mm'] if reloc else cfg['recall_max_t_mm']
    max_r = cfg['recall_reloc_max_r_deg'] if reloc else cfg['recall_max_r_deg']
    try_all = reloc and pend is None
    state['pending'] = None
    for kid in cands:
        if kid not in pool.archive:
            continue
        a = pool.archive[kid]
        sol = solve_against(kp, des, a['pts'], a['des'], geo, cfg)
        fd['recall_kf'] = kid
        if sol is None:
            if try_all:
                continue
            break
        fd['recall_inliers'] = sol['n_inliers']
        if (sol['n_inliers'] < cfg['recall_min_inliers'] or sol['ratio'] < cfg['recall_min_ratio']
                or sol['std_t_mm'] > cfg['recall_max_std_mm']):
            if try_all:
                continue
            break
        ct, cr = step_between(T_est, sol['T'], mm)
        if ct > max_t or cr > max_r:
            decisions['recall_implausible'] += 1
            if try_all:
                continue
            break
        if ct < cfg['recall_min_t_mm'] and cr < cfg['recall_min_r_deg']:
            break
        W = np.linalg.inv(sol['T']) @ T_est
        if pend is None:
            state['pending'] = (kid, W, frame)
            break
        dt, dr = step_between(np.linalg.inv(pend[1]), np.linalg.inv(W), mm)
        if dt > cfg['recall_confirm_t_mm'] or dr > cfg['recall_confirm_r_deg']:
            decisions['recall_unconfirmed'] += 1
            break
        if reloc and kid < state['reloc_kid']:
            pool.apply_offset(W, state['reloc_kid'])
            state['reloc_kid'] = None
            fd['recall_reloc'] = 1
            decisions['recall_reloc'] += 1
        else:
            pool.apply_correction(W, kid)
        pool.reinsert(kid)
        T_est = sol['T']
        output_T = info_filter.update(T_est, None, reset=True)
        fd['recall_applied'] = 1
        decisions['recall_applied'] += 1
        state['cooldown'] = int(cfg['recall_cooldown'])
        if verbose:
            print(f'[{frame:05d}] RECALL keyframe {kid} (frame {a["frame"]}): correction {ct:.2f} mm / '
                  f'{cr:.2f} deg, {sol["n_inliers"]} inliers')
        break
    return T_est, output_T


def _finish(out, cfg, geo, seed, rows, rows_raw, decisions, timing, completed, error, wall_s, n_kf):
    import pandas as pd
    df = pd.DataFrame(rows)
    if len(df):
        if not cfg['stream_poses']:
            df.to_csv(out / 'camera_pose_recovered.csv', index=False)
            pd.DataFrame(rows_raw).to_csv(out / 'camera_pose_raw.csv', index=False)
        try:
            save_plots(df, out)
        except Exception as e:
            print(f'[WARNING] plots failed: {e}')
    tstats = {k: {'median': round(float(np.median(v[10:] or v)), 1), 'p95': round(float(np.percentile(v[10:] or v, 95)), 1),
                  'p99': round(float(np.percentile(v[10:] or v, 99)), 1)}
              for k, v in timing.items() if v}
    meta = {'completed': completed, 'error': error, 'wall_s': round(wall_s, 1),
            'n_frames': len(df), 'n_keyframes': n_kf, 'decisions': dict(decisions),
            'pose_camera': 'rectified_left', 'pose_unit': cfg['calib_unit'],
            'R_l': geo['R_l'].tolist(), 'K_rect': geo['K_rect'].tolist(), 'proc_wh': list(geo['proc_wh']),
            'baseline_mm': round(geo['baseline_mm'], 4), 'seed': seed, 'timing_ms': tstats, 'config': cfg}
    (out / 'run_meta.json').write_text(json.dumps(meta, indent=2, default=str))
    print(f"[svo] {'done' if completed else 'STOPPED'}: {len(df)} frames, {n_kf} keyframes, "
          f"decisions {dict(decisions)}, {wall_s:.0f} s -> {out}")


def save_plots(df, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for c in 'xyz':
        ax[0].plot(df['frame'], df[f'cam_in_model_{c}'], label=c)
    ax[0].set_ylabel('camera position')
    ax[0].legend()
    for c in 'wxyz':
        ax[1].plot(df['frame'], df[f'quat_{c}'], label=f'q{c}')
    ax[1].set_ylabel('quaternion')
    ax[1].legend()
    ax[2].plot(df['frame'], df['inliers'], color='0.4', lw=0.8)
    bad = df['prior_ok'] == 0
    ax[2].plot(df['frame'][bad], df['inliers'][bad], 'r.', ms=3, label='rejected')
    ax[2].set_ylabel('PnP inliers')
    ax[2].set_xlabel('frame')
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(Path(out_dir) / 'pipeline_overview.png', dpi=150)
    plt.close(fig)

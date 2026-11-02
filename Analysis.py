import csv
import sys

import numpy as np

NOSE = 0
LEFT = {'shoulder': 11, 'elbow': 13, 'wrist': 15, 'hip': 23, 'knee': 25, 'ankle': 27, 'heel': 29, 'toe': 31}
RIGHT = {'shoulder': 12, 'elbow': 14, 'wrist': 16, 'hip': 24, 'knee': 26, 'ankle': 28, 'heel': 30, 'toe': 32}

MIN_VISIBILITY = 0.5

REP_DOWN_THRESHOLD = 140
REP_UP_THRESHOLD = 160


def joint_angle(a, b, c):
    a, b, c = (np.asarray(p, dtype=float)[:3] for p in (a, b, c))
    ba, bc = a - b, c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return np.nan
    cos = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def angle_from_vertical(bottom, top):
    v = np.asarray(top, dtype=float)[:3] - np.asarray(bottom, dtype=float)[:3]
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(-v[1] / norm, -1.0, 1.0))))


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float)[:3] - np.asarray(b, dtype=float)[:3]))


def pick_side(frames):
    keys = ['shoulder', 'hip', 'knee', 'ankle']
    scores = {'left': [], 'right': []}
    for f in frames:
        lm = f.get('landmarks')
        if lm is None:
            continue
        scores['left'].append(np.mean([lm[LEFT[k], 3] for k in keys]))
        scores['right'].append(np.mean([lm[RIGHT[k], 3] for k in keys]))
    if not scores['left']:
        return 'left'
    return 'left' if np.mean(scores['left']) >= np.mean(scores['right']) else 'right'


def squat_frame_metrics(frame, side='left'):
    lm, world = frame.get('landmarks'), frame.get('world_landmarks')
    if lm is None or world is None:
        return None

    idx = LEFT if side == 'left' else RIGHT
    needed = ['shoulder', 'hip', 'knee', 'ankle']
    if min(lm[idx[k], 3] for k in needed) < MIN_VISIBILITY:
        return None

    shoulder, hip, knee, ankle = (world[idx[k]] for k in needed)

    return {
        'knee_angle': joint_angle(hip, knee, ankle),
        'hip_angle': joint_angle(shoulder, hip, knee),
        'torso_lean': angle_from_vertical(hip, shoulder),
        'shin_lean': angle_from_vertical(ankle, knee),
        'hip_below_knee_m': float(hip[1] - knee[1]),
    }


def squat_metrics_series(frames, side=None):
    side = side or pick_side(frames)
    rows = []
    for i, f in enumerate(frames):
        m = squat_frame_metrics(f, side)
        row = {'frame': i, 'timestamp_ms': f.get('timestamp_ms', 0), 'valid': m is not None}
        row.update(m or {k: np.nan for k in ('knee_angle', 'hip_angle', 'torso_lean', 'shin_lean', 'hip_below_knee_m')})
        rows.append(row)
    return rows, side


def fill_and_smooth(values, window=5):
    v = np.asarray(values, dtype=float)
    valid = ~np.isnan(v)
    if valid.sum() < 2:
        return v
    idx = np.arange(len(v))
    v = np.interp(idx, idx[valid], v[valid])
    if window > 1 and len(v) >= window:
        kernel = np.ones(window) / window
        padded = np.pad(v, (window // 2, window - 1 - window // 2), mode='edge')
        v = np.convolve(padded, kernel, mode='valid')
    return v


def detect_reps(knee_angles, down=REP_DOWN_THRESHOLD, up=REP_UP_THRESHOLD):
    reps = []
    state = 'up'
    start = 0
    bottom = 0
    for i, a in enumerate(knee_angles):
        if np.isnan(a):
            continue
        if state == 'up':
            if a >= up:
                start = i
            elif a < down:
                state, bottom = 'down', i
        else:
            if a < knee_angles[bottom]:
                bottom = i
            if a >= up:
                reps.append((start, bottom, i))
                state, start = 'up', i
    return reps


def summarize_rep(rows, start, bottom, end, fps):
    span = rows[start:end + 1]
    lean = [r['torso_lean'] for r in span if r['valid']]
    b = rows[bottom]
    return {
        'start_frame': start,
        'bottom_frame': bottom,
        'end_frame': end,
        'duration_s': round((end - start) / fps, 2),
        'descent_s': round((bottom - start) / fps, 2),
        'ascent_s': round((end - bottom) / fps, 2),
        'min_knee_angle': b['knee_angle_smooth'],
        'hip_angle_at_bottom': b['hip_angle'],
        'torso_lean_at_bottom': b['torso_lean'],
        'max_torso_lean': max(lean) if lean else np.nan,
        'shin_lean_at_bottom': b['shin_lean'],
        'hip_below_knee_m': b['hip_below_knee_m'],
        'below_parallel': bool(b['hip_below_knee_m'] > 0) if not np.isnan(b['hip_below_knee_m']) else None,
    }


def ratios_from_measurements(measurements):
    if not measurements:
        return None
    m = measurements
    out = {}
    if m.get('femur') and m.get('torso'):
        out['femur_to_torso'] = m['femur'] / m['torso']
    if m.get('lower_leg') and m.get('femur'):
        out['lower_leg_to_femur'] = m['lower_leg'] / m['femur']
    if m.get('arm') and m.get('torso'):
        out['arm_to_torso'] = m['arm'] / m['torso']
    return out


def ratios_from_landmarks(frames, side='left'):
    idx = LEFT if side == 'left' else RIGHT
    torso, femur, lower_leg, arm = [], [], [], []
    for f in frames:
        w, lm = f.get('world_landmarks'), f.get('landmarks')
        if w is None or min(lm[idx[k], 3] for k in ('shoulder', 'hip', 'knee', 'ankle', 'elbow', 'wrist')) < MIN_VISIBILITY:
            continue
        torso.append(distance(w[idx['shoulder']], w[idx['hip']]))
        femur.append(distance(w[idx['hip']], w[idx['knee']]))
        lower_leg.append(distance(w[idx['knee']], w[idx['ankle']]))
        arm.append(distance(w[idx['shoulder']], w[idx['elbow']]) + distance(w[idx['elbow']], w[idx['wrist']]))
    if not torso:
        return None
    t, fe, ll, a = (float(np.median(x)) for x in (torso, femur, lower_leg, arm))
    return {'femur_to_torso': fe / t, 'lower_leg_to_femur': ll / fe, 'arm_to_torso': a / t}


def compare_ratios(entered, detected, tolerance=0.2):
    warnings = []
    if not entered or not detected:
        return warnings
    for key, val in entered.items():
        if key in detected and val:
            diff = abs(detected[key] - val) / val
            if diff > tolerance:
                warnings.append(f"{key}: entered {val:.2f}, video suggests {detected[key]:.2f}. Check the measurement or camera angle.")
    return warnings


def analyze_squat(frames, fps=30.0, measurements=None):
    rows, side = squat_metrics_series(frames)
    smooth = fill_and_smooth([r['knee_angle'] for r in rows])
    for r, s in zip(rows, smooth):
        r['knee_angle_smooth'] = float(s)

    reps = [summarize_rep(rows, *r, fps=fps) for r in detect_reps(smooth)]

    valid_count = sum(r['valid'] for r in rows)
    entered = ratios_from_measurements(measurements)
    detected = ratios_from_landmarks(frames, side)

    warnings = []
    if not frames:
        warnings.append("Video had no frames.")
    elif valid_count / len(frames) < 0.5:
        warnings.append("Key joints were hidden in most frames. Try a side view with your whole body in frame, and avoid baggy clothing.")
    if valid_count and not reps:
        warnings.append("No full reps found. Make sure you start and finish standing up.")
    warnings += compare_ratios(entered, detected)

    return {
        'exercise': 'squat',
        'side': side,
        'fps': fps,
        'frames_total': len(frames),
        'frames_valid': valid_count,
        'reps': reps,
        'body': {'entered_ratios': entered, 'detected_ratios': detected},
        'per_frame': rows,
        'warnings': warnings,
    }


def analyze_squat_image(pose, measurements=None):
    if pose is None:
        return {'exercise': 'squat', 'metrics': None, 'warnings': ["No person detected in the photo."]}
    frame = dict(pose)
    side = pick_side([frame])
    metrics = squat_frame_metrics(frame, side)
    warnings = [] if metrics else ["Key joints aren't visible enough. Use a side view with your whole body in frame."]
    return {
        'exercise': 'squat',
        'side': side,
        'metrics': metrics,
        'body': {'entered_ratios': ratios_from_measurements(measurements)},
        'warnings': warnings,
    }


def save_per_frame_csv(result, path):
    rows = result['per_frame']
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    from comp_vision import detect_video

    frames, fps = detect_video(sys.argv[1])
    result = analyze_squat(frames, fps)
    print(f"Side: {result['side']}  Valid frames: {result['frames_valid']}/{result['frames_total']}")
    for n, rep in enumerate(result['reps'], 1):
        print(f"Rep {n}: knee {rep['min_knee_angle']:.0f} deg, hip {rep['hip_angle_at_bottom']:.0f} deg, "
              f"max lean {rep['max_torso_lean']:.0f} deg, below parallel: {rep['below_parallel']}")
    for w in result['warnings']:
        print("Warning:", w)
    if len(sys.argv) > 2:
        save_per_frame_csv(result, sys.argv[2])
        print("Saved", sys.argv[2])

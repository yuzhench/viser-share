"""Which grid block is which finger: the 11-07 23-12-44 no-contact take (single-finger curls, nothing touched). Per finger, the
bend from the glove's own skeleton (sum of the joint angles along the finger); per grid block, the summed raw rise above rest.
Correlate the two over time, per glove."""
import json
from pathlib import Path
import numpy as np
T = Path('/root/autodl-tmp/tactile_baseline/data/hilotah/full/2026-11-07/23-12-44/gloves')
H, W = 24, 31
BLK = {'c0-4 r5-13': (slice(5, 14), slice(0, 5)), 'c5-10 r1-9': (slice(1, 10), slice(5, 11)), 'c13-18 r0-9': (slice(0, 10), slice(13, 19)),
       'c21-25 r1-9': (slice(1, 10), slice(21, 26)), 'c26-30 r9-17': (slice(9, 18), slice(26, 31)), 'palm r13-23 c5-25': (slice(13, 24), slice(5, 26)),
       'c0 r15-23': (slice(15, 24), slice(0, 1))}
names = ['wrist', 'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip', 'index_finger_mcp', 'index_finger_pip', 'index_finger_dip', 'index_finger_tip',
         'middle_finger_mcp', 'middle_finger_pip', 'middle_finger_dip', 'middle_finger_tip', 'ring_finger_mcp', 'ring_finger_pip', 'ring_finger_dip',
         'ring_finger_tip', 'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip']
FING = {'thumb': [0, 1, 2, 3, 4], 'index': [0, 5, 6, 7, 8], 'middle': [0, 9, 10, 11, 12], 'ring': [0, 13, 14, 15, 16], 'little': [0, 17, 18, 19, 20]}
def ang(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9); b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return np.degrees(np.arccos(np.clip((a * b).sum(-1), -1, 1)))
out = {}
for side in ('left', 'right'):
    sk = np.load(T / side / 'skeleton.npz'); tc = np.load(T / side / 'tactile.npz')
    dt = (tc['t_us'] - sk['t_us']) / 1e3; print(side, 'tactile minus skeleton time (ms): median %.2f, min %.2f, max %.2f' % (np.median(dt), dt.min(), dt.max()))
    P = sk['data'][:, :, :3]; t = (sk['t_us'] - sk['t_us'][0]) / 1e6
    print(side, 'skeleton sample 0 joint xyz (first 6):', np.round(P[0, :6], 3).tolist(), ' extra cols', np.round(sk['data'][0, :3, 3:], 3).tolist())
    flex = {}
    for f, ix in FING.items():
        Q = P[:, ix]; seg = np.diff(Q, axis=1)
        flex[f] = sum(ang(seg[:, i], seg[:, i + 1]) for i in range(seg.shape[1] - 1))
    R = tc['data'].astype(np.float64); valid = (R != -1).all(0); dead = valid & ((R > 0).mean(0) < 0.01)
    rest = np.percentile(np.where(valid, R, 0), 5, axis=0); X = np.where(valid & ~dead, np.clip(R - rest, 0, None), 0).reshape(-1, H, W)
    tt = (tc['t_us'] - sk['t_us'][0]) / 1e6
    act = {b: np.interp(t, tt, X[:, rs, cs].reshape(len(X), -1).sum(1)) for b, (rs, cs) in BLK.items()}
    # smooth both to 0.25 s
    k = np.ones(30) / 30
    sm = lambda x: np.convolve(x, k, mode='same')
    F = {f: sm(v) for f, v in flex.items()}; A = {b: sm(v) for b, v in act.items()}
    print(side, '%d samples over %.1f s' % (len(t), t[-1]))
    print('   finger bend range (deg, p5..p95):', {f: (round(float(np.percentile(v, 5))), round(float(np.percentile(v, 95)))) for f, v in F.items()})
    hdr = '%-20s' % 'block \\ finger' + ''.join('%9s' % f for f in F)
    print('  ', hdr)
    C = {}
    for b in A:
        row = [np.corrcoef(A[b], F[f])[0, 1] for f in F]; C[b] = row
        print('   %-20s' % b + ''.join('%9.2f' % r for r in row), '  best:', list(F)[int(np.argmax(row))])
    # when each finger bends most: windows where that finger's bend is in its top 10% and is the most-bent finger (z-scored)
    Z = {f: (v - np.median(v)) / (np.percentile(v, 95) - np.median(v) + 1e-6) for f, v in F.items()}
    zf = np.stack([Z[f] for f in F], 1); top = np.argmax(zf, 1)
    print('   seconds where each finger is the most bent (z>0.6):', {f: round(float(((top == i) & (zf[:, i] > 0.6)).sum() / 120), 1) for i, f in enumerate(F)})
    print('   mean block activity in those moments:')
    for b in A:
        print('     %-20s' % b + ''.join('%9.2f' % (A[b][(top == i) & (zf[:, i] > 0.6)].mean() if ((top == i) & (zf[:, i] > 0.6)).any() else np.nan) for i, f in enumerate(F)))
    out[side] = dict(t=t, F=F, A=A)
np.savez('/root/autodl-tmp/tactile_baseline/tmp/nc_blocks_ts.npz', **{'%s_%s' % (s, k): v for s in out for k, v in [('t', out[s]['t'])] + [('F_' + f, x) for f, x in out[s]['F'].items()] + [('A_' + b.split()[0], x) for b, x in out[s]['A'].items()]})
print('NC_BLOCKS_DONE')

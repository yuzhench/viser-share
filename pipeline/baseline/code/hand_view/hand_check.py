"""Check pictures for setup/hand_taxel_map.py: where the slots land, each taxel's patch on the hand (checkerboard), and one frame of
the cotton take drawn on the hands next to the grid."""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, '/root/autodl-tmp/tactile_baseline/setup')
from hand_mano_base import raster
TB = Path('/root/autodl-tmp/tactile_baseline'); CHK = TB / 'results/site_preview/tmp_check'
Z = np.load(TB / 'results/tactile_hand/hand_debug.npz'); P, F, N, G, COV, lab = Z['P'], Z['F'], Z['N'], Z['G'], Z['COV'], Z['lab']
H, W = 24, 31; SIG = 0.45
STOPS = [(0, (50, 55, 64)), (0.012, (96, 26, 30)), (0.06, (158, 30, 36)), (0.13, (226, 44, 40)), (0.21, (250, 138, 36)), (0.3, (255, 232, 92))]
def cmap(v):
    xs = [s[0] for s in STOPS]; out = np.stack([np.interp(v, xs, [s[1][c] for s in STOPS]) for c in range(3)], -1) / 255
    return out
R = raster(P, F, lab, res=0.3e-3); tid = R['tid']; m = tid >= 0; b = R['bary'][m].astype(float)
corner = F[tid[m]]
Gp = np.einsum('nk,nkd->nd', b, G[corner]); Cp = np.einsum('nk,nk->n', b, COV[corner]); Np = np.einsum('nk,nkd->nd', b, N[corner])
Np /= np.linalg.norm(Np, axis=1, keepdims=True)
L = np.array([0.35, 0.45, 0.82]); L /= np.linalg.norm(L); lam = np.clip(Np @ L, 0, 1)
SKIN = np.array([0.34, 0.36, 0.40])
def shade_img(rgb_pix, alpha):
    img = np.zeros(tid.shape + (3,)) + np.array([0.078, 0.09, 0.11])
    skin = SKIN[None] * (0.38 + 0.62 * lam[:, None])
    heat = rgb_pix * (0.72 + 0.28 * lam[:, None])
    img[m] = skin * (1 - alpha[:, None]) + heat * alpha[:, None]
    return img
def heat_pix(val, mask):           # val, mask: (744,) arrays (value 0..1, mask 0/1)
    c0 = np.floor(Gp + 0.5); ws = np.zeros(len(Gp)); vs = np.zeros(len(Gp))
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            c = c0 + np.array([dx, dy]); ok = (c[:, 0] >= 0) & (c[:, 0] <= W - 1) & (c[:, 1] >= 0) & (c[:, 1] <= H - 1)
            idx = (np.clip(c[:, 1], 0, H - 1) * W + np.clip(c[:, 0], 0, W - 1)).astype(int)
            d = Gp - c; w = np.exp(-(d * d).sum(1) / (2 * SIG * SIG)) * mask[idx] * ok
            ws += w; vs += w * val[idx]
    v = vs / np.maximum(ws, 1e-4)
    t = np.clip((ws - 0.15) / 0.4, 0, 1); a = t * t * (3 - 2 * t) * Cp
    return v, a
ext = [R['lo'][0] * 1e3, (R['lo'][0] + tid.shape[1] * R['res']) * 1e3, R['lo'][1] * 1e3, (R['lo'][1] + tid.shape[0] * R['res']) * 1e3]

# 1) where the slots land + checkerboard of taxel patches (all 744 grid cells treated as present)
cols = plt.cm.tab10(np.arange(10))
fig, ax = plt.subplots(1, 2, figsize=(16, 9.5))
base = shade_img(np.zeros((m.sum(), 3)), np.zeros(m.sum()))
ax[0].imshow(base, origin='lower', extent=ext)
sid, sxy, sreg = Z['slot_ids'], Z['slot_xyz'][:, :2], Z['slot_reg']
valid = np.zeros(744, bool)
for side in ('left', 'right'):
    d = np.load(TB / 'data/hilotah/full/2026-11-07/23-12-44/gloves' / side / 'tactile.npz')['data'][:50]; valid |= (d != -1).all(0)
names = [str(x) for x in Z['RN']]
for i, nm in enumerate(names):
    k = (sreg == i) & valid[sid]
    ax[0].scatter(sxy[k, 0] * 1e3, sxy[k, 1] * 1e3, s=7, color=cols[i], label=nm)
for r in (0, 1, 5, 9, 10, 13, 18, 23):
    for c in (0, 5, 13, 21, 26, 30):
        k = np.nonzero((sid == r * W + c) & valid[sid])[0]
        if len(k): ax[0].text(sxy[k[0], 0] * 1e3 + 1, sxy[k[0], 1] * 1e3 + 1, '%d,%d' % (r, c), fontsize=6, color='w')
ax[0].legend(fontsize=7, loc='lower left'); ax[0].set_title('grid slots placed on the right hand (palm view); labels: row,col', fontsize=10)
chk = ((np.floor(Gp[:, 0] + 0.5) + np.floor(Gp[:, 1] + 0.5)) % 2).astype(float)
cidx = (np.clip(np.floor(Gp[:, 1] + 0.5), 0, H - 1) * W + np.clip(np.floor(Gp[:, 0] + 0.5), 0, W - 1)).astype(int)
rgb = np.where(chk[:, None] > 0, np.array([0.95, 0.75, 0.3]), np.array([0.3, 0.55, 0.95])) * valid[cidx][:, None]
ax[1].imshow(shade_img(rgb, Cp * valid[cidx]), origin='lower', extent=ext); ax[1].set_title('each taxel cell on the surface (checkerboard; grey = no taxel)', fontsize=10)
for a in ax: a.set_aspect('equal')
plt.tight_layout(); plt.savefig(CHK / 'hand_map_slots.png', dpi=100); plt.close()
from PIL import Image
im = Image.open(CHK / 'hand_map_slots.png'); w, h = im.size; im.crop((0, 0, w // 2, h)).save(CHK / 'hand_map_slots_a.png'); im.crop((w // 2, 0, w, h)).save(CHK / 'hand_map_slots_b.png')

# 2) one frame of the cotton take, both gloves: grid vs hand
pg = TB / 'results/site_preview/pipeline/baseline_1105b/tactile'; meta = json.loads((pg / 'meta.json').read_text()); S = len(meta['slots'])
slots = np.array(meta['slots'])
best = None
for k, Pm in meta['parts'].items():
    a = np.frombuffer((pg / Pm['file']).read_bytes(), np.uint8).reshape(4, Pm['n'], S)
    sc = (a[0] > 0).sum(1) + (a[2] > 0).sum(1)
    j = int(np.argmax(sc))
    if best is None or sc[j] > best[0]: best = (sc[j], k, j, a)
_, kp, j, a = best; print('frame', kp, j, 'contact taxels', best[0])
fig, ax = plt.subplots(2, 2, figsize=(11, 11))
for h, side in enumerate(('left', 'right')):
    v = np.zeros(744); v[slots] = a[2 * h, j] / 255.0
    mask = np.zeros(744); mask[slots] = 1
    for sl in meta['dead'][side] + meta['absent'][side]: mask[sl] = 0
    ax[0, h].set_facecolor((0.078, 0.09, 0.11))
    for s_i, sl in enumerate(slots):
        r, c = divmod(int(sl), W)
        if mask[sl] == 0:
            if sl not in meta['absent'][side]: ax[0, h].add_patch(plt.Circle((c, r), 0.32, fill=False, color='#5b636e', lw=0.8))
            continue
        ax[0, h].add_patch(plt.Circle((c, r), 0.42, color=cmap(v[sl])))
    ax[0, h].set_xlim(-0.6, W - 0.4); ax[0, h].set_ylim(H - 0.4, -0.6); ax[0, h].set_aspect('equal'); ax[0, h].set_title('%s glove, filtered, grid' % side)
    val, alpha = heat_pix(v, mask)
    img = shade_img(cmap(val), alpha)
    if side == 'left': img = img[:, ::-1]
    ax[1, h].imshow(img, origin='lower'); ax[1, h].set_title('%s glove on the %shand (palm view)' % (side, 'mirrored ' if side == 'left' else ''))
    ax[1, h].axis('off')
plt.tight_layout(); plt.savefig(CHK / 'hand_map_frame.png', dpi=90); plt.close()
print('HAND_CHECK_DONE')

import os
import json
import random
import cv2
import numpy as np
from scipy.fft import fft
from PIL import Image, ImageDraw
from itertools import combinations
from collections import defaultdict

# === Fourier Descriptors ===
def compute_fd(path, M=32):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot load image for FD: {path}")
    if img.shape[2] < 4:
        raise ValueError(f"Image has no alpha channel: {path}")
    alpha = img[:, :, 3]
    mask = (alpha > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError(f"No contour found in mask: {path}")
    cnt = max(contours, key=lambda c: cv2.contourArea(c))
    pts = cnt.reshape(-1, 2)
    z = pts[:, 0] + 1j * pts[:, 1]
    Z = fft(z)
    Z1 = np.abs(Z[1]) if np.abs(Z[1]) != 0 else 1.0
    Z = Z / Z1
    fd = np.abs(Z[1:M+1])
    return fd.tolist()

def smooth_offsets(offsets, window_size=3):
    half = window_size // 2
    smoothed, n = [], len(offsets)
    for i in range(n):
        start, end = max(0, i-half), min(n, i+half+1)
        smoothed.append(sum(offsets[start:end]) / (end-start))
    return smoothed

def generate_edges(width, height, rows, cols, jitter_ratio, points_per_edge, smoothing_window):
    cell_w, cell_h = width / cols, height / rows
    max_jx, max_jy = cell_w * jitter_ratio, cell_h * jitter_ratio
    vertical, horizontal = {}, {}
    for c in range(cols):
        for r in range(rows):
            x = c * cell_w
            y0, y1 = r * cell_h, (r + 1) * cell_h
            if c < cols:
                raw = [random.uniform(-max_jx, max_jx) for _ in range(points_per_edge)]
                sx = smooth_offsets(raw, smoothing_window)
                pts = [(int(x + jx), int(y0 + (y1 - y0) * i / (points_per_edge - 1))) for i, jx in enumerate(sx)]
                vertical[(r, c)] = pts
    for r in range(rows):
        for c in range(cols):
            y = r * cell_h
            x0, x1 = c * cell_w, (c + 1) * cell_w
            if r < rows:
                raw = [random.uniform(-max_jy, max_jy) for _ in range(points_per_edge)]
                sy = smooth_offsets(raw, smoothing_window)
                pts = [(int(x0 + (x1 - x0) * i / (points_per_edge - 1)), int(y + jy)) for i, jy in enumerate(sy)]
                horizontal[(r, c)] = pts
    return vertical, horizontal

def slice_and_save_pieces(img_path, pieces_dir, img_id, rows=3, cols=3,
                          jitter_ratio=0.1, points_per_edge=20, smoothing_window=5):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    vert, hor = generate_edges(w, h, rows, cols, jitter_ratio, points_per_edge, smoothing_window)
    os.makedirs(pieces_dir, exist_ok=True)
    meta = []
    for r in range(rows):
        for c in range(cols):
            poly = []
            if r == 0:
                poly += [(int(c * w / cols), int(r * h / rows)), (int((c + 1) * w / cols), int(r * h / rows))]
            else:
                poly += hor[(r, c)]
            if c == cols - 1:
                poly += [(int((c + 1) * w / cols), int(r * h / rows)), (int((c + 1) * w / cols), int((r + 1) * h / rows))]
            else:
                poly += vert[(r, c)]
            if r == rows - 1:
                poly += [(int((c + 1) * w / cols), int((r + 1) * h / rows)), (int(c * w / cols), int((r + 1) * h / rows))]
            else:
                poly += hor[(r + 1, c)][::-1]
            if c == 0:
                poly += [(int(c * w / cols), int((r + 1) * h / rows)), (int(c * w / cols), int(r * h / rows))]
            else:
                poly += vert[(r, c - 1)][::-1]
            xs, ys = zip(*poly)
            min_x, max_x = max(0, min(xs)), min(w, max(xs))
            min_y, max_y = max(0, min(ys)), min(h, max(ys))
            local = [(x - min_x, y - min_y) for x, y in poly]
            mask = Image.new('L', (max_x - min_x, max_y - min_y), 0)
            draw = ImageDraw.Draw(mask)
            draw.polygon(local, fill=255)
            crop = img.crop((min_x, min_y, max_x, max_y))
            piece = Image.new('RGBA', crop.size)
            piece.paste(crop, (0, 0), mask)
            path = os.path.join(pieces_dir, f"piece_{img_id}_{r}_{c}.png")
            piece.save(path)
            meta.append({'img_id': img_id, 'r': r, 'c': c, 'path': path})
    return meta

def get_adjacent_indices(meta, rows=3, cols=3):
    pos = {(m['r'], m['c']): i for i, m in enumerate(meta)}
    adj = []
    for (r, c), i in pos.items():
        if c < cols - 1: adj.append((i, pos[(r, c + 1)], 'right'))
        if r < rows - 1: adj.append((i, pos[(r + 1, c)], 'below'))
    return adj

def generate_all_pairs_jsonl(input_dir, pieces_dir, output_dir,
                            rows=3, cols=3, jitter_ratio=0.2,
                            points_per_edge=15, smoothing_window=3,
                            fd_M=32):
    os.makedirs(pieces_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    angles = list(range(0, 360, 15))
    dir_map = {'left': 0, 'right': 1, 'above': 2, 'below': 3}
    global_meta, adjacency = [], []
    for img_i, fname in enumerate(sorted(os.listdir(input_dir))):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')): continue
        meta = slice_and_save_pieces(os.path.join(input_dir, fname), pieces_dir, img_i,
                                     rows, cols, jitter_ratio, points_per_edge, smoothing_window)
        base = len(global_meta); global_meta.extend(meta)
        for i, j, rel in get_adjacent_indices(meta, rows, cols):
            adjacency.append((base + i, base + j, rel))
    num_pos = len(adjacency)
    adj_set = set(tuple(sorted((i, j))) for i, j, _ in adjacency)
    img2idx = defaultdict(list)
    for idx, m in enumerate(global_meta): img2idx[m['img_id']].append(idx)
    out = os.path.join(output_dir, 'train_pairs.jsonl')
    with open(out, 'w') as fout:
        for pi, (i1, i2, rel) in enumerate(adjacency):
            m1, m2 = global_meta[i1], global_meta[i2]
            a1, a2 = random.choice(angles), random.choice(angles)
            diff = ((a2 - a1 + 360) % 360); diff = diff - 360 if diff > 180 else diff
            p1 = Image.open(m1['path']).rotate(a1, expand=True)
            p2 = Image.open(m2['path']).rotate(a2, expand=True)
            fn1, fn2 = f"pos_{pi}_a.png", f"pos_{pi}_b.png"
            p1.save(os.path.join(output_dir, fn1)); p2.save(os.path.join(output_dir, fn2))
            fd_a = compute_fd(m1['path'], M=fd_M)
            fd_b = compute_fd(m2['path'], M=fd_M)
            record = {'frag_a': fn1, 'frag_b': fn2, 'is_adjacent': 1,
                      'direction': dir_map[rel], 'angle_diff': diff,
                      'fd_a': fd_a, 'fd_b': fd_b}
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
        neg, used = 0, set()
        for img_id, idxs in img2idx.items():
            for i1, i2 in combinations(idxs, 2):
                if tuple(sorted((i1, i2))) in adj_set: continue
                key = tuple(sorted([(img_id, global_meta[i1]['r'], global_meta[i1]['c']),
                                    (img_id, global_meta[i2]['r'], global_meta[i2]['c'])]))
                if key in used: continue
                used.add(key)
                a1, a2 = random.choice(angles), random.choice(angles)
                diff = ((a2 - a1 + 360) % 360); diff = diff - 360 if diff > 180 else diff
                p1 = Image.open(global_meta[i1]['path']).rotate(a1, expand=True)
                p2 = Image.open(global_meta[i2]['path']).rotate(a2, expand=True)
                fn1, fn2 = f"neg_same_{neg}_a.png", f"neg_same_{neg}_b.png"
                p1.save(os.path.join(output_dir, fn1)); p2.save(os.path.join(output_dir, fn2))
                fd_a = compute_fd(global_meta[i1]['path'], M=fd_M)
                fd_b = compute_fd(global_meta[i2]['path'], M=fd_M)
                record = {'frag_a': fn1, 'frag_b': fn2, 'is_adjacent': 0,
                          'direction': 0, 'angle_diff': diff,
                          'fd_a': fd_a, 'fd_b': fd_b}
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                neg += 1
                if neg >= num_pos: break
            if neg >= num_pos: break
        trials = 0
        while neg < num_pos and trials < 10 * num_pos:
            p1 = random.choice(global_meta); p2 = random.choice(global_meta)
            if p1['img_id'] == p2['img_id']: trials += 1; continue
            key = tuple(sorted([(p1['img_id'], p1['r'], p1['c']), (p2['img_id'], p2['r'], p2['c'])]))
            if key in used: trials += 1; continue
            used.add(key)
            a1, a2 = random.choice(angles), random.choice(angles)
            diff = ((a2 - a1 + 360) % 360); diff = diff - 360 if diff > 180 else diff
            p_img1 = Image.open(p1['path']).rotate(a1, expand=True)
            p_img2 = Image.open(p2['path']).rotate(a2, expand=True)
            fn1, fn2 = f"neg_cross_{neg}_a.png", f"neg_cross_{neg}_b.png"
            p_img1.save(os.path.join(output_dir, fn1)); p_img2.save(os.path.join(output_dir, fn2))
            fd_a = compute_fd(p1['path'], M=fd_M)
            fd_b = compute_fd(p2['path'], M=fd_M)
            record = {'frag_a': fn1, 'frag_b': fn2, 'is_adjacent': 0,
                      'direction': 0, 'angle_diff': diff,
                      'fd_a': fd_a, 'fd_b': fd_b}
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')
            neg += 1; trials += 1
    print(f"Generated {2 * num_pos} pairs with FD in {out}")

if __name__ == '__main__':
    generate_all_pairs_jsonl('dataset3', 'pieces', 'train_fragments_final')

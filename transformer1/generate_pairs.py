import os
import json
import random
import numpy as np
from PIL import Image

def resize_image(img: Image, max_dim: int = 600) -> Image:
    w, h = img.size
    if w >= h:
        new_w = max_dim
        new_h = int(max_dim * h / w)
    else:
        new_h = max_dim
        new_w = int(max_dim * w / h)
    return img.resize((new_w, new_h), Image.LANCZOS)


def cut_image_random_grid(
    img: Image,
    n: int = 3,
    jitter: float = 0.1,
    smoothness: float = 0.02
) -> dict:
    img = img.convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]

    verticals = np.zeros((n + 1, h))
    verticals[0, :] = 0
    verticals[-1, :] = w
    for i in range(1, n):
        base_x = (i / n) * w
        offset = np.random.uniform(-jitter * w, jitter * w, size=h)
        k = max(3, int(smoothness * h))
        kernel = np.ones(k) / k
        smooth = np.convolve(offset, kernel, mode='same')
        xs = np.clip(base_x + smooth, 0, w)
        verticals[i, :] = xs

    horizontals = np.zeros((n + 1, w))
    horizontals[0, :] = 0
    horizontals[-1, :] = h
    for j in range(1, n):
        base_y = (j / n) * h
        offset = np.random.uniform(-jitter * h, jitter * h, size=w)
        k = max(3, int(smoothness * w))
        kernel = np.ones(k) / k
        smooth = np.convolve(offset, kernel, mode='same')
        ys = np.clip(base_y + smooth, 0, h)
        horizontals[j, :] = ys

    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    segments = {}

    for i in range(n):
        for j in range(n):
            left = verticals[i][Y]
            right = verticals[i + 1][Y]
            top = horizontals[j][X]
            bottom = horizontals[j + 1][X]
            mask = (X >= left) & (X < right) & (Y >= top) & (Y < bottom)

            cell_arr = arr.copy()
            cell_arr[~mask] = [0, 0, 0, 0]

            ys = np.any(mask, axis=1)
            xs = np.any(mask, axis=0)
            if not ys.any() or not xs.any():
                continue
            y_min, y_max = np.where(ys)[0][[0, -1]]
            x_min, x_max = np.where(xs)[0][[0, -1]]
            cell_img = Image.fromarray(cell_arr).crop((x_min, y_min, x_max + 1, y_max + 1))
            segments[(i, j)] = cell_img

    return segments


def process_folder(
    input_dir: str,
    output_dir: str,
    n: int = 3,
    jitter: float = 0.1,
    smoothness: float = 0.02,
    max_dim: int = 600
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "pair_info.jsonl")

    with open(jsonl_path, "a", encoding="utf-8") as f:
        for fname in os.listdir(input_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            base, _ = os.path.splitext(fname)
            img_path = os.path.join(input_dir, fname)

            img = Image.open(img_path)
            img = resize_image(img, max_dim)

            segments = cut_image_random_grid(img, n, jitter, smoothness)

            pos_pairs = []
            for (i, j) in segments:
                if (i + 1, j) in segments:
                    pos_pairs.append(((i, j), (i + 1, j)))
                if (i, j + 1) in segments:
                    pos_pairs.append(((i, j), (i, j + 1)))

            cells = list(segments.keys())
            neg_candidates = []
            for idx, (i1, j1) in enumerate(cells):
                for (i2, j2) in cells[idx + 1:]:
                    if not ((abs(i1 - i2) == 1 and j1 == j2) or (abs(j1 - j2) == 1 and i1 == i2)):
                        neg_candidates.append(((i1, j1), (i2, j2)))

            neg_pairs = random.sample(neg_candidates, min(len(neg_candidates), len(pos_pairs)))

            all_pairs = [(p, 1) for p in pos_pairs] + [(p, 0) for p in neg_pairs]
            for ((i1, j1), (i2, j2)), label in all_pairs:
                angle1 = random.choice([0, 90, 180, 270])
                angle2 = random.choice([0, 90, 180, 270])
                frag1 = segments[(i1, j1)].rotate(angle1, expand=True)
                frag2 = segments[(i2, j2)].rotate(angle2, expand=True)

                out1 = f"{base}_cell_{i1}_{j1}_rot{angle1}.png"
                out2 = f"{base}_cell_{i2}_{j2}_rot{angle2}.png"
                path1 = os.path.join(output_dir, out1)
                path2 = os.path.join(output_dir, out2)
                frag1.save(path1, format="PNG")
                frag2.save(path2, format="PNG")

                di, dj = i2 - i1, j2 - j1
                if di < 0 and dj == 0:
                    direction = "up"
                elif di > 0 and dj == 0:
                    direction = "down"
                elif di == 0 and dj > 0:
                    direction = "right"
                elif di == 0 and dj < 0:
                    direction = "left"
                elif di < 0 and dj > 0:
                    direction = "up-right"
                elif di < 0 and dj < 0:
                    direction = "up-left"
                elif di > 0 and dj > 0:
                    direction = "down-right"
                elif di > 0 and dj < 0:
                    direction = "down-left"
                else:
                    direction = ""

                entry = {
                    "image1": path1,
                    "image2": path2,
                    "label": label,
                    "direction": direction,
                    "rotation1": angle1,
                    "rotation2": angle2
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    process_folder(
        input_dir="../dataset",
        output_dir="pairs_train",
        n=3,
        jitter=0.5,
        smoothness=0.2,
        max_dim=600
    )
    print("Сегменты, пары и файл pair_info.jsonl сохранены.")

import cv2
import numpy as np
import math
import os

# -------------------------------
# BRAILLE DICTIONARY (For Auto-Labeling)
# -------------------------------
braille = {
    (1, 0, 0, 0, 0, 0): 'A', (1, 1, 0, 0, 0, 0): 'B', (1, 0, 0, 1, 0, 0): 'C', (1, 0, 0, 1, 1, 0): 'D',
    (1, 0, 0, 0, 1, 0): 'E', (1, 1, 0, 1, 0, 0): 'F', (1, 1, 0, 1, 1, 0): 'G', (1, 1, 0, 0, 1, 0): 'H',
    (0, 1, 0, 1, 0, 0): 'I', (0, 1, 0, 1, 1, 0): 'J', (1, 0, 1, 0, 0, 0): 'K', (1, 1, 1, 0, 0, 0): 'L',
    (1, 0, 1, 1, 0, 0): 'M', (1, 0, 1, 1, 1, 0): 'N', (1, 0, 1, 0, 1, 0): 'O', (1, 1, 1, 1, 0, 0): 'P',
    (1, 1, 1, 1, 1, 0): 'Q', (1, 1, 1, 0, 1, 0): 'R', (0, 1, 1, 1, 0, 0): 'S', (0, 1, 1, 1, 1, 0): 'T',
    (1, 0, 1, 0, 0, 1): 'U', (1, 1, 1, 0, 0, 1): 'V', (0, 1, 0, 1, 1, 1): 'W', (1, 0, 1, 1, 0, 1): 'X',
    (1, 0, 1, 1, 1, 1): 'Y', (1, 0, 1, 0, 1, 1): 'Z'
}

# Create dataset directories
os.makedirs("dataset", exist_ok=True)
for letter in braille.values():
    os.makedirs(f"dataset/{letter}", exist_ok=True)

# -------------------------------
# LOAD & PREPROCESS
# -------------------------------
img = cv2.imread("braille.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, th = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

# Find Dots
num, labels, stats, centroids = cv2.connectedComponentsWithStats(th)
dots = [centroids[i] for i in range(1, num) if 20 < stats[i, cv2.CC_STAT_AREA] < 2500]

# Spacing Calculation
nnd = [min([math.hypot(d1[0] - d2[0], d1[1] - d2[1]) for j, d2 in enumerate(dots) if i != j]) for i, d1 in
       enumerate(dots)]
base_dist = np.median(nnd)

dxs = [abs(d1[0] - d2[0]) for d1 in dots for d2 in dots if
       0 < abs(d1[0] - d2[0]) < base_dist * 2.5 and abs(d1[1] - d2[1]) < base_dist * 0.6]
dys = [abs(d1[1] - d2[1]) for d1 in dots for d2 in dots if
       0 < abs(d1[1] - d2[1]) < base_dist * 2.5 and abs(d1[0] - d2[0]) < base_dist * 0.6]

d_x, d_y = int(np.median(dxs)), int(np.median(dys))

# Group into Lines
dots.sort(key=lambda d: d[1])
lines, curr_line = [], [dots[0]]
for d in dots[1:]:
    if d[1] - curr_line[-1][1] > d_y * 1.8:
        lines.append(curr_line)
        curr_line = [d]
    else:
        curr_line.append(d)
lines.append(curr_line)


def cluster_vals(vals, threshold):
    vals = sorted(vals)
    clusters, curr = [], [vals[0]]
    for v in vals[1:]:
        if v - curr[-1] <= threshold:
            curr.append(v)
        else:
            clusters.append(int(np.mean(curr))); curr = [v]
    clusters.append(int(np.mean(curr)))
    return clusters


# -------------------------------
# CROP AND SAVE IMAGES
# -------------------------------
saved_count = 0
for line_dots in lines:
    line_dots.sort(key=lambda d: d[0])

    chars, curr_char = [], [line_dots[0]]
    for d in line_dots[1:]:
        if d[0] - curr_char[-1][0] > d_x * 1.5:
            chars.append(curr_char)
            curr_char = [d]
        else:
            curr_char.append(d)
    chars.append(curr_char)

    y_clusters = cluster_vals([d[1] for d in line_dots], threshold=d_y * 0.6)
    y_rows = [y_clusters[0]]
    if len(y_clusters) > 1:
        if y_clusters[1] - y_rows[0] > d_y * 1.5:
            y_rows.extend([y_rows[0] + d_y, y_clusters[1]])
        else:
            y_rows.append(y_clusters[1])
            y_rows.append(y_clusters[2] if len(y_clusters) > 2 else y_rows[1] + d_y)
    else:
        y_rows.extend([y_rows[0] + d_y, y_rows[0] + 2 * d_y])
    y_rows = y_rows[:3]

    for char_dots in chars:
        x_clusters = cluster_vals([d[0] for d in char_dots], threshold=d_x * 0.6)
        x0 = x_clusters[0]
        x_cols = [x0, x0 + d_x]

        pattern = [0] * 6
        for d in char_dots:
            c_idx = 0 if abs(d[0] - x_cols[0]) < abs(d[0] - x_cols[1]) else 1
            r_diffs = [abs(d[1] - yr) for yr in y_rows]
            r_idx = r_diffs.index(min(r_diffs))
            pattern[c_idx * 3 + r_idx] = 1

        letter = braille.get(tuple(pattern), None)

        if letter:
            # Calculate bounding box with padding
            pad = 15
            x_min, x_max = max(0, x_cols[0] - pad), min(th.shape[1], x_cols[1] + pad)
            y_min, y_max = max(0, y_rows[0] - pad), min(th.shape[0], y_rows[2] + pad)

            # Crop and save
            crop = th[y_min:y_max, x_min:x_max]
            # Resize to standard 32x32 for ML
            crop = cv2.resize(crop, (32, 32))
            cv2.imwrite(f"dataset/{letter}/{saved_count}.png", crop)
            saved_count += 1

print(f"Successfully generated {saved_count} labeled Braille training images!")
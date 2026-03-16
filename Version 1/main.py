import cv2
import numpy as np
import math

# -------------------------------
# BRAILLE DICTIONARY (A-Z)
# -------------------------------
braille = {
    (1, 0, 0, 0, 0, 0): 'A', (1, 1, 0, 0, 0, 0): 'B', (1, 0, 0, 1, 0, 0): 'C', (1, 0, 0, 1, 1, 0): 'D',
    (1, 0, 0, 0, 1, 0): 'E', (1, 1, 0, 1, 0, 0): 'F', (1, 1, 0, 1, 1, 0): 'G', (1, 1, 0, 0, 1, 0): 'H',
    (0, 1, 0, 1, 0, 0): 'I', (0, 1, 0, 1, 1, 0): 'J',
    (1, 0, 1, 0, 0, 0): 'K', (1, 1, 1, 0, 0, 0): 'L', (1, 0, 1, 1, 0, 0): 'M', (1, 0, 1, 1, 1, 0): 'N',
    (1, 0, 1, 0, 1, 0): 'O', (1, 1, 1, 1, 0, 0): 'P', (1, 1, 1, 1, 1, 0): 'Q', (1, 1, 1, 0, 1, 0): 'R',
    (0, 1, 1, 1, 0, 0): 'S', (0, 1, 1, 1, 1, 0): 'T',
    (1, 0, 1, 0, 0, 1): 'U', (1, 1, 1, 0, 0, 1): 'V', (0, 1, 0, 1, 1, 1): 'W', (1, 0, 1, 1, 0, 1): 'X',
    (1, 0, 1, 1, 1, 1): 'Y', (1, 0, 1, 0, 1, 1): 'Z',
    (0, 0, 0, 0, 0, 0): ' '
}

# -------------------------------
# LOAD & PREPROCESS
# -------------------------------
img = cv2.imread("braille.png")

if img is None:
    print("Image not found. Make sure braille.png is in the folder.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, th = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

kernel = np.ones((3, 3), np.uint8)
th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

num, labels, stats, centroids = cv2.connectedComponentsWithStats(th)
dots = []

for i in range(1, num):
    area = stats[i, cv2.CC_STAT_AREA]
    # Wide area bounds to handle large resolutions safely
    if 20 < area < 2500:
        dots.append(centroids[i])

print("Dots detected:", len(dots))
if not dots:
    print("No dots detected. Check your thresholding.")
    exit()

# -------------------------------
# SCALE-INVARIANT SPACING CALC
# -------------------------------
# Calculate the distance to each dot's nearest neighbor to dynamically find cell sizing
nnd = []
for i, d1 in enumerate(dots):
    dists = [math.hypot(d1[0] - d2[0], d1[1] - d2[1]) for j, d2 in enumerate(dots) if i != j]
    nnd.append(min(dists))
base_dist = np.median(nnd)

dxs, dys = [], []
for i, d1 in enumerate(dots):
    min_dx, min_dy = 9999, 9999
    for j, d2 in enumerate(dots):
        if i == j: continue
        dx = abs(d1[0] - d2[0])
        dy = abs(d1[1] - d2[1])
        if dy < base_dist * 0.6 and dx > base_dist * 0.5: min_dx = min(min_dx, dx)
        if dx < base_dist * 0.6 and dy > base_dist * 0.5: min_dy = min(min_dy, dy)
    if min_dx < base_dist * 2.5: dxs.append(min_dx)
    if min_dy < base_dist * 2.5: dys.append(min_dy)

d_x = int(np.median(dxs)) if dxs else int(base_dist)
d_y = int(np.median(dys)) if dys else int(base_dist)

# -------------------------------
# GROUP INTO LINES (Y-Clustering)
# -------------------------------
dots.sort(key=lambda d: d[1])
lines = []
curr_line = [dots[0]]
for d in dots[1:]:
    # If the vertical jump is large, it signifies a new text line
    if d[1] - curr_line[-1][1] > d_y * 1.8:
        lines.append(curr_line)
        curr_line = [d]
    else:
        curr_line.append(d)
lines.append(curr_line)


def cluster_vals(vals, threshold):
    vals = sorted(vals)
    clusters = []
    curr = [vals[0]]
    for v in vals[1:]:
        if v - curr[-1] <= threshold:
            curr.append(v)
        else:
            clusters.append(int(np.mean(curr)))
            curr = [v]
    clusters.append(int(np.mean(curr)))
    return clusters


# -------------------------------
# PROCESS EACH LINE INDIVIDUALLY
# -------------------------------
recognized_text = ""

for line_dots in lines:
    line_dots.sort(key=lambda d: d[0])

    # 1. Group dots into characters based on X-distance
    chars = []
    curr_char = [line_dots[0]]
    for d in line_dots[1:]:
        if d[0] - curr_char[-1][0] > d_x * 1.5:
            chars.append(curr_char)
            curr_char = [d]
        else:
            curr_char.append(d)
    chars.append(curr_char)

    # 2. Estimate spacing for missing word spaces
    char_starts = [min([d[0] for d in c]) for c in chars]
    strides = []
    for i in range(1, len(char_starts)):
        dist = char_starts[i] - char_starts[i - 1]
        if dist < d_x * 3.5:  # Standard inter-character distance
            strides.append(dist)
    base_stride = np.median(strides) if strides else d_x * 2.5

    # 3. Determine strict Y rows for this specific line
    y_clusters = cluster_vals([d[1] for d in line_dots], threshold=d_y * 0.6)
    y_rows = [y_clusters[0]]
    if len(y_clusters) > 1:
        if y_clusters[1] - y_rows[0] > d_y * 1.5:
            y_rows.extend([y_rows[0] + d_y, y_clusters[1]])  # Handle missing middle row
        else:
            y_rows.append(y_clusters[1])
            if len(y_clusters) > 2:
                y_rows.append(y_clusters[2])
            else:
                y_rows.append(y_rows[1] + d_y)
    else:
        y_rows.extend([y_rows[0] + d_y, y_rows[0] + 2 * d_y])
    y_rows = y_rows[:3]

    line_text = ""
    for i, char_dots in enumerate(chars):
        # 4. Inject word spaces based on distance
        if i > 0:
            dist = char_starts[i] - char_starts[i - 1]
            num_spaces = int(round(dist / base_stride)) - 1
            for _ in range(max(0, num_spaces)):
                line_text += " "

        # 5. Map columns for this specific character
        x_clusters = cluster_vals([d[0] for d in char_dots], threshold=d_x * 0.6)
        x0 = x_clusters[0]
        x_cols = [x0, x0 + d_x]

        # 6. Assign dots to the 2x3 Braille grid
        pattern = [0] * 6
        for d in char_dots:
            c_idx = 0 if abs(d[0] - x_cols[0]) < abs(d[0] - x_cols[1]) else 1
            r_diffs = [abs(d[1] - yr) for yr in y_rows]
            r_idx = r_diffs.index(min(r_diffs))
            pattern[c_idx * 3 + r_idx] = 1

        line_text += braille.get(tuple(pattern), "?")

    recognized_text += line_text.rstrip() + "\n"

print("\nRecognized Text:\n")
print(recognized_text)
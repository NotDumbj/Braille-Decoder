import cv2
import numpy as np
import math
import tensorflow as tf
import os

# -------------------------------
# LOAD ML MODEL
# -------------------------------
if not os.path.exists('braille_cnn.keras'):
    print("Model not found! Run train_model.py first.")
    exit()

model = tf.keras.models.load_model('braille_cnn.keras')
# Keep this ordered exactly as your dataset folder sorted them (usually alphabetical A-Z)
class_names = sorted(os.listdir("dataset"))

# -------------------------------
# LOAD & PREPROCESS IMAGE
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
# PREDICT WITH AI
# -------------------------------
recognized_text = ""

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

    char_starts = [min([d[0] for d in c]) for c in chars]
    strides = [char_starts[i] - char_starts[i - 1] for i in range(1, len(char_starts)) if
               char_starts[i] - char_starts[i - 1] < d_x * 3.5]
    base_stride = np.median(strides) if strides else d_x * 2.5

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

    line_text = ""
    for i, char_dots in enumerate(chars):
        # Inject word spaces
        if i > 0:
            dist = char_starts[i] - char_starts[i - 1]
            for _ in range(max(0, int(round(dist / base_stride)) - 1)):
                line_text += " "

        # Determine strict column bounds for this character
        x_clusters = cluster_vals([d[0] for d in char_dots], threshold=d_x * 0.6)
        x_cols = [x_clusters[0], x_clusters[0] + d_x]

        # Crop the cell from the image
        pad = 15
        x_min, x_max = max(0, x_cols[0] - pad), min(th.shape[1], x_cols[1] + pad)
        y_min, y_max = max(0, y_rows[0] - pad), min(th.shape[0], y_rows[2] + pad)

        cell_crop = th[y_min:y_max, x_min:x_max]

        # Ask the AI to predict the letter
        if cell_crop.size > 0:
            cell_resized = cv2.resize(cell_crop, (32, 32))
            # Reshape for the CNN (1 image, 32x32 pixels, 1 grayscale channel)
            cell_array = np.expand_dims(cell_resized, axis=[0, -1])

            prediction = model.predict(cell_array, verbose=0)
            predicted_class_index = np.argmax(prediction)
            line_text += class_names[predicted_class_index]
        else:
            line_text += "?"

    recognized_text += line_text.rstrip() + "\n"

print("\nAI Recognized Text:\n")
print(recognized_text)
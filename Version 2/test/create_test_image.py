import cv2
import numpy as np

braille_dict = {
    'A': (1, 0, 0, 0, 0, 0), 'B': (1, 1, 0, 0, 0, 0), 'C': (1, 0, 0, 1, 0, 0), 'D': (1, 0, 0, 1, 1, 0),
    'E': (1, 0, 0, 0, 1, 0), 'F': (1, 1, 0, 1, 0, 0), 'G': (1, 1, 0, 1, 1, 0), 'H': (1, 1, 0, 0, 1, 0),
    'I': (0, 1, 0, 1, 0, 0), 'J': (0, 1, 0, 1, 1, 0), 'K': (1, 0, 1, 0, 0, 0), 'L': (1, 1, 1, 0, 0, 0),
    'M': (1, 0, 1, 1, 0, 0), 'N': (1, 0, 1, 1, 1, 0), 'O': (1, 0, 1, 0, 1, 0), 'P': (1, 1, 1, 1, 0, 0),
    'Q': (1, 1, 1, 1, 1, 0), 'R': (1, 1, 1, 0, 1, 0), 'S': (0, 1, 1, 1, 0, 0), 'T': (0, 1, 1, 1, 1, 0),
    'U': (1, 0, 1, 0, 0, 1), 'V': (1, 1, 1, 0, 0, 1), 'W': (0, 1, 0, 1, 1, 1), 'X': (1, 0, 1, 1, 0, 1),
    'Y': (1, 0, 1, 1, 1, 1), 'Z': (1, 0, 1, 0, 1, 1), ' ': (0, 0, 0, 0, 0, 0)
}

text = "MACHINE LEARNING IS AWESOME"

# 1. FIXED SPACING & THICKNESS
dot_radius = 12  # Thicker dots to match the real photograph
dx, dy = 35, 35  # Space between dots in a single cell
char_space = 100  # Wider space between characters (prevents blob merging)
word_space = 150  # Space between words

# Make the canvas slightly larger to fit the wider text
img = np.ones((250, 3200, 3), dtype=np.uint8) * 255
x_offset, y_offset = 50, 50

for char in text:
    if char == ' ':
        x_offset += word_space
        continue

    pattern = braille_dict.get(char, (0, 0, 0, 0, 0, 0))
    for i in range(6):
        if pattern[i]:
            cx = x_offset + (i // 3) * dx
            cy = y_offset + (i % 3) * dy
            cv2.circle(img, (cx, cy), dot_radius, (0, 0, 0), -1)

    x_offset += char_space

# 2. ADD REALISM
# Apply a slight blur so the digital dots mimic camera/ink softness
img = cv2.GaussianBlur(img, (9, 9), 0)

cv2.imwrite("test_braille.png", img)
print("Saved realistic test_braille.png!")
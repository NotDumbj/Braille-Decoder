import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import math
import tensorflow as tf
import os
import threading

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class BrailleDecoderApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Braille Decoder")
        self.root.geometry("900x600")
        self.root.configure(bg="#1e1e2e")

        self.image_path = None
        self.model = None
        self.class_names = sorted(os.listdir("dataset")) if os.path.exists("dataset") else []

        self.setup_ui()
        self.load_model()

    def setup_ui(self):
        # Header
        header = tk.Label(self.root, text="Neural Network Braille Translator",
                          font=("Segoe UI", 20, "bold"), bg="#1e1e2e", fg="#cdd6f4")
        header.pack(pady=20)

        # Main Layout Frame
        main_frame = tk.Frame(self.root, bg="#1e1e2e")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20)

        # Left Column (Image)
        left_frame = tk.Frame(main_frame, bg="#313244", width=400, height=300)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        left_frame.pack_propagate(False)

        self.img_label = tk.Label(left_frame, text="No Image Loaded", bg="#313244", fg="#a6adc8", font=("Segoe UI", 12))
        self.img_label.pack(expand=True)

        btn_frame = tk.Frame(left_frame, bg="#313244")
        btn_frame.pack(pady=10)

        self.btn_load = tk.Button(btn_frame, text="Load Image", command=self.load_image,
                                  bg="#89b4fa", fg="#11111b", font=("Segoe UI", 10, "bold"), padx=10)
        self.btn_load.pack(side=tk.LEFT, padx=5)

        self.btn_decode = tk.Button(btn_frame, text="Decode with AI", command=self.run_inference_thread,
                                    bg="#a6e3a1", fg="#11111b", font=("Segoe UI", 10, "bold"), padx=10,
                                    state=tk.DISABLED)
        self.btn_decode.pack(side=tk.LEFT, padx=5)

        # Right Column (Output Text)
        right_frame = tk.Frame(main_frame, bg="#1e1e2e")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(right_frame, text="Decoded Output:", bg="#1e1e2e", fg="#cdd6f4", font=("Segoe UI", 14, "bold")).pack(
            anchor="w")

        self.text_output = tk.Text(right_frame, bg="#181825", fg="#a6e3a1", font=("Consolas", 14),
                                   wrap=tk.WORD, relief=tk.FLAT, padx=10, pady=10)
        self.text_output.pack(fill=tk.BOTH, expand=True, pady=10)

    def load_model(self):
        if os.path.exists('braille_cnn.keras'):
            self.model = tf.keras.models.load_model('braille_cnn.keras')
            self.text_output.insert(tk.END, "Model loaded successfully! Ready to decode.\n")
        else:
            self.text_output.insert(tk.END, "Error: 'braille_cnn.keras' not found. Run training script first.\n")

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.image_path = file_path
            img = Image.open(file_path)

            # Resize for display
            img.thumbnail((400, 300))
            self.tk_img = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.tk_img, text="")
            self.btn_decode.config(state=tk.NORMAL)
            self.text_output.delete(1.0, tk.END)

    def run_inference_thread(self):
        if not self.model:
            messagebox.showerror("Error", "Model not loaded.")
            return

        self.btn_decode.config(state=tk.DISABLED, text="Decoding...")
        self.text_output.delete(1.0, tk.END)
        self.text_output.insert(tk.END, "Processing image with Convolutional Neural Network...\n\n")

        # Run inference in a separate thread to keep GUI responsive
        threading.Thread(target=self.decode_image, daemon=True).start()

    def decode_image(self):
        try:
            img = cv2.imread(self.image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, th = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)

            num, labels, stats, centroids = cv2.connectedComponentsWithStats(th)
            dots = [centroids[i] for i in range(1, num) if 10 < stats[i, cv2.CC_STAT_AREA] < 2500]

            if not dots:
                self.update_output("No dots detected in image.")
                return

            nnd = [min([math.hypot(d1[0] - d2[0], d1[1] - d2[1]) for j, d2 in enumerate(dots) if i != j]) for i, d1 in
                   enumerate(dots)]
            base_dist = np.median(nnd)

            dxs = [abs(d1[0] - d2[0]) for d1 in dots for d2 in dots if
                   0 < abs(d1[0] - d2[0]) < base_dist * 2.5 and abs(d1[1] - d2[1]) < base_dist * 0.6]
            dys = [abs(d1[1] - d2[1]) for d1 in dots for d2 in dots if
                   0 < abs(d1[1] - d2[1]) < base_dist * 2.5 and abs(d1[0] - d2[0]) < base_dist * 0.6]

            # Strict array slicing forces it to measure the inside of the letter, not the spaces
            d_x = int(np.median(np.sort(dxs)[:max(1, len(dxs) // 3)])) if dxs else int(base_dist)
            d_y = int(np.median(np.sort(dys)[:max(1, len(dys) // 3)])) if dys else int(base_dist)

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

            crops_to_predict = []
            text_layout = []

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

                line_meta = []
                for i, char_dots in enumerate(chars):
                    spaces = 0
                    if i > 0:
                        dist = char_starts[i] - char_starts[i - 1]
                        spaces = max(0, int(round(dist / base_stride)) - 1)

                    # Fixed Grid Anchor (Preserves the exact 2x3 Aspect Ratio for the AI)
                    x_anchor = min([d[0] for d in char_dots])
                    y_anchor = min([d[1] for d in char_dots])

                    pad = 15
                    x_min = int(max(0, x_anchor - pad))
                    x_max = int(min(th.shape[1], x_anchor + d_x + pad))
                    y_min = int(max(0, y_anchor - pad))
                    y_max = int(min(th.shape[0], y_anchor + 2 * d_y + pad))

                    cell_crop = th[y_min:y_max, x_min:x_max]

                    if cell_crop.size > 0:
                        cell_resized = cv2.resize(cell_crop, (32, 32))
                        crops_to_predict.append(cell_resized)
                        line_meta.append({"spaces": spaces, "valid": True})
                    else:
                        line_meta.append({"spaces": spaces, "valid": False})

                text_layout.append(line_meta)

            if crops_to_predict:
                crops_array = np.array(crops_to_predict).reshape(-1, 32, 32, 1)
                predictions = self.model.predict(crops_array, verbose=0)
                predicted_classes = np.argmax(predictions, axis=1)
            else:
                predicted_classes = []

            recognized_text = ""
            pred_idx = 0
            for line_meta in text_layout:
                line_text = ""
                for meta in line_meta:
                    line_text += " " * meta["spaces"]
                    if meta["valid"]:
                        line_text += self.class_names[predicted_classes[pred_idx]]
                        pred_idx += 1
                    else:
                        line_text += "?"
                recognized_text += line_text.rstrip() + "\n"

            self.update_output(recognized_text)

        except Exception as e:
            self.update_output(f"Error during decoding: {str(e)}")

    def update_output(self, text):
        # Safely update GUI from the thread
        self.root.after(0, self._set_text_and_reset_btn, text)

    def _set_text_and_reset_btn(self, text):
        self.text_output.insert(tk.END, text)
        self.btn_decode.config(state=tk.NORMAL, text="Decode with AI")


if __name__ == "__main__":
    root = tk.Tk()
    app = BrailleDecoderApp(root)
    root.mainloop()
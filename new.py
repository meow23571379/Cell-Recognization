import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy import ndimage as ndi
from scipy.stats import pearsonr
from scipy.ndimage import center_of_mass
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import gaussian
import pandas as pd
from readlif.reader import LifFile

def convert_lif_to_tif():
    file_paths = filedialog.askopenfilenames(filetypes=[("LIF files", "*.lif")])
    if not file_paths:
        return
    
    for filepath in file_paths:
        try:
            filename = os.path.basename(filepath)
            status_var.set(f"Converting: {filename}...")
            root.update() # 即時更新 GUI

            new_file = LifFile(filepath)
            image = new_file.get_image(0)
            
            frames = [np.array(image.get_frame(z=0, t=0, c=c)) for c in range(image.channels)]
            matrix_data = np.array(frames)
            
            tif_file_name = os.path.splitext(filename)[0] + ".tif"
            output_tif_path = os.path.join(os.path.dirname(filepath), tif_file_name)
            
            tiff.imwrite(output_tif_path, matrix_data, imagej=True, metadata={'axes': 'CYX'})
        except Exception as e:
            print(f"Error converting {filename}: {e}")
            
    status_var.set("Ready")
    messagebox.showinfo("Complete", "LIF to TIF conversion finished!")

def process_images():
    file_paths = filedialog.askopenfilenames(filetypes=[("TIFF files", "*.tif *.tiff")])
    if not file_paths:
        return

    for filepath in file_paths:
        try:
            filename = os.path.basename(filepath)
            status_var.set(f"Processing: {filename}...")
            root.update() # 即時更新 GUI

            raw_data = tiff.imread(filepath).astype(float)
            if len(raw_data.shape) != 3 or raw_data.shape[0] < 3:
                print(f"Skipping {filename}: Script requires at least 3 channels")
                continue
            
            ch1, ch2, ch3 = raw_data[0], raw_data[1], raw_data[2]

            smoothed_data1 = gaussian(ch1, sigma=20.0)
            smoothed_data2 = gaussian(ch2, sigma=20.0)
            data = (smoothed_data1 + smoothed_data2*2) / 3
            smoothed_data = gaussian(data, sigma=20.0)

            threshold = np.mean(smoothed_data) * 0.5
            mask = smoothed_data > threshold
            coordinates = peak_local_max(smoothed_data, min_distance=100, labels=mask)

            peaks_mask = np.zeros(data.shape, dtype=bool)
            peaks_mask[tuple(coordinates.T)] = True
            markers, _ = ndi.label(peaks_mask)
            labels = watershed(-smoothed_data, markers, mask=mask)

            image_height, image_width = data.shape
            unique_labels = np.unique(labels)[1:] 
            
            filtered_original_labels = []
            for label_val in unique_labels:
                coords = np.argwhere(labels == label_val)
                is_edge = any(r == 0 or r == image_height - 1 or c == 0 or c == image_width - 1 for r, c in coords)
                if not is_edge:
                    filtered_original_labels.append(label_val)

            correlation_results = []
            for label_val in filtered_original_labels:
                region_mask = (labels == label_val)
                ch1_px = ch1[region_mask]
                ch3_px = ch3[region_mask]

                if len(ch1_px) > 1 and len(ch3_px) > 1:
                    corr, p_val = pearsonr(ch1_px, ch3_px)
                    m1 = np.sum(ch1_px[ch3_px > 0]) / np.sum(ch1_px) if np.sum(ch1_px) > 0 else np.nan
                    m2 = np.sum(ch3_px[ch1_px > 0]) / np.sum(ch3_px) if np.sum(ch3_px) > 0 else np.nan

                    correlation_results.append({
                        'Region Label': label_val,
                        'Pearson Correlation': corr,
                        'P-value': p_val,
                        'M1 Coefficient': m1,
                        'M2 Coefficient': m2
                    })

            base_name = os.path.splitext(filepath)[0]
            pd.DataFrame(correlation_results).to_csv(f"{base_name}.csv", index=False)

            filtered_img = np.zeros_like(labels, dtype=int)
            for label_val in filtered_original_labels:
                filtered_img[labels == label_val] = label_val

            plt.figure(figsize=(10, 8))
            plt.imshow(filtered_img, cmap='nipy_spectral')
            
            for label_val in filtered_original_labels:
                cy, cx = center_of_mass(data, labels, label_val)
                plt.text(cx, cy, str(int(label_val)), color='white', fontsize=8, ha='center', va='center')

            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{base_name}.png", dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    status_var.set("Ready")
    messagebox.showinfo("Complete", "All selected TIF files have been processed!")

root = tk.Tk()
root.title("Microscopy Batch Processor")
root.geometry("350x200")

tk.Label(root, text="Select an action:").pack(pady=5)
tk.Button(root, text="Convert LIF to TIF", command=convert_lif_to_tif, width=20, height=2).pack(pady=5)
tk.Button(root, text="Select TIF & Process", command=process_images, width=20, height=2).pack(pady=5)

# Status Panel
status_var = tk.StringVar()
status_var.set("Ready")
tk.Label(root, textvariable=status_var, fg="blue", wraplength=330).pack(pady=5)

root.mainloop()
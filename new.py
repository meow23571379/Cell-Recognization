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
from skimage.filters import gaussian, threshold_otsu
from skimage.exposure import equalize_adapthist
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




def stretch_to_255(channel):
    """Linearly stretch a channel's intensity to the 0-255 range."""
    channel = channel.astype(float)
    ch_min = np.min(channel)
    ch_max = np.max(channel)

    print(ch_min, ch_max)
    if ch_max > ch_min:
        stretched = (channel - ch_min) * 255.0 / (ch_max - ch_min) 
    else:
        stretched = np.zeros_like(channel)
    return stretched

def binarize_channel(channel, fill_holes=False):
    """
    Binarize a channel using a mean-based threshold so that any pixel with
    signal becomes 1 (255), regardless of how dim it is relative to the
    brightest nuclei. This prevents dimmer nuclei from being lost when
    channel 2 is combined with the other channels for segmentation.

    If fill_holes=True, scipy.ndimage.binary_fill_holes is applied
    afterward. This is meant for channel 2 (nucleus): nuclear signal often
    has an interior hole (e.g. nucleolus, dim center), and if left unfilled
    the watershed/threshold step below can be misled by that hole. Filling
    it gives a solid nucleus mask.
    """
    if np.max(channel) <= np.min(channel):
        return np.zeros_like(channel)
    thresh_val = np.mean(channel) * 0.1
    binary_bool = channel > thresh_val
    if fill_holes:
        binary_bool = ndi.binary_fill_holes(binary_bool)
    binary = binary_bool.astype(float) * 255.0
    return binary

def local_contrast_enhance(channel, target_mask=None, clip_limit=0.03):
    """
    Brighten locally underexposed regions of a channel using CLAHE
    (Contrast Limited Adaptive Histogram Equalization).

    Rationale: channel 2 marks where signal is expected to be. If, within
    a region where channel 2 shows signal, this channel's local intensity
    is much lower than channel 2's, that region is likely underexposed
    rather than truly negative. CLAHE re-normalizes contrast locally
    (in tiles) instead of globally, so dim sub-regions get brightened
    without blowing out already-bright areas.

    If target_mask is provided (e.g. binarized channel 2), the enhanced
    version is only applied inside that mask; outside of it, the original
    values are kept so background noise isn't amplified.
    """
    norm = channel / 255.0
    norm = np.clip(norm, 0, 1)
    enhanced = equalize_adapthist(norm, clip_limit=clip_limit) * 255.0
    if target_mask is not None:
        enhanced = np.where(target_mask, enhanced, channel)
    return enhanced

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
            
            # Original channels: used for quantification (Pearson / M1 / M2)
            ch1, ch2, ch3 = raw_data[0], raw_data[1], raw_data[2]
            print(ch1.shape[1])
            Small_pic = (ch1.shape[1] < 1000)
            # Stretched channels (0-255): used only for segmentation, so that
            # lower-expression cells are not lost during thresholding/watershed
            stretched_ch1 = stretch_to_255(ch1)
            print("----")
            stretched_ch2 = stretch_to_255(ch2)
            print("----")
            stretched_ch3 = stretch_to_255(ch3)
            print("====")

            # Channel 2 = target signal. Binarize it (Otsu) so any pixel with
            # signal becomes 1/255, even if it's dim relative to the brightest
            # nuclei - this stops dim nuclei from being washed out when
            # combined with the other channels below.
            binary_ch2 = binarize_channel(stretched_ch2, fill_holes=True)
            target_mask = binary_ch2 > 0
            region_mask = (stretched_ch1 + binary_ch2 + stretched_ch3) > 0

            # Channel 1 and channel 3 can be locally underexposed: where
            # channel 2 shows signal but ch1/ch3 is much dimmer in that same
            # area, that area is likely underexposed rather than truly
            # negative. Locally brighten (CLAHE) ch1/ch3 within the channel 2
            # signal footprint to correct for this.
            enhanced_ch1 = local_contrast_enhance(stretched_ch1, target_mask=region_mask)
            enhanced_ch3 = local_contrast_enhance(stretched_ch3, target_mask=region_mask)

            # Save stretched/binarized/enhanced channels for visual verification
            base_name = os.path.splitext(filepath)[0]
            stretched_stack = np.stack([stretched_ch1, stretched_ch2, stretched_ch3]).astype(np.uint8)
            tiff.imwrite(f"{base_name}_stretched.tif", stretched_stack, imagej=True, metadata={'axes': 'CYX'})
            seg_input_stack = np.stack([(enhanced_ch1*2 + binary_ch2)/3, binary_ch2, enhanced_ch3]).astype(np.uint8)
            tiff.imwrite(f"{base_name}_seg_input.tif", seg_input_stack, imagej=True, metadata={'axes': 'CYX'})

            smoothed_data1 = gaussian(enhanced_ch1, sigma=20.0)
            smoothed_data2 = gaussian(binary_ch2, sigma=20.0)
            smoothed_data3 = gaussian(enhanced_ch3, sigma=20.0)
            data = (smoothed_data1 + smoothed_data2+ smoothed_data3) / 3
            smoothed_data = gaussian(data, sigma=10.0)

            threshold = np.mean(smoothed_data) * 0.5
            mask = smoothed_data > threshold

            # Save mask as a TIF for visual verification
            tiff.imwrite(f"{base_name}_mask.tif", (mask.astype(np.uint8) * 255))

            # --- Nucleus-based watershed markers ---
            # Instead of peak_local_max on the combined smoothed intensity
            # (which can merge two touching cells into one peak when their
            # local maxima aren't distinct enough), use each individual
            # nucleus - a connected component in the filled channel 2 binary
            # mask - as its own watershed seed. One nucleus = one cell, so
            # this gives more reliable separation of cells sitting close
            # together.
            if(Small_pic) :
                MIN_NUCLEUS_AREA = 300
            else:
                MIN_NUCLEUS_AREA = 500  # px, nuclei smaller than this are treated as noise and removed

            # binary_ch2 was already thresholded + hole-filled. Erode it by
            # 5 px to shrink/eliminate small noise blobs and separate nuclei
            # that are only touching by a thin bridge, then label connected
            # components and take coordinates. Areas that are still too small
            # after erosion get dropped in Step 2 below.
            EROSION_PX = 5
            nucleus_bool_mask = ndi.binary_erosion(binary_ch2 > 0, iterations=EROSION_PX)

            raw_nucleus_labels, num_raw_nuclei = ndi.label(nucleus_bool_mask)

            # Step 1: get the coordinate + area of every detected nucleus first
            all_nuclei = []
            for nuc_label in range(1, num_raw_nuclei + 1):
                nuc_region = (raw_nucleus_labels == nuc_label)
                area = np.sum(nuc_region)
                cy, cx = center_of_mass(nuc_region)
                all_nuclei.append({'label': nuc_label, 'row': cy, 'col': cx, 'area': area})

            # Save the full (unfiltered) list for verification, so you can see
            # exactly which ones get dropped as too small in the next step
            if all_nuclei:
                pd.DataFrame(all_nuclei).to_csv(f"{base_name}_nucleus_coordinates_all.csv", index=False)

            # Step 2: remove nuclei that are too small to be real (noise blobs)
            # before they're turned into watershed markers
            kept_nuclei = [n for n in all_nuclei if n['area'] >= MIN_NUCLEUS_AREA]

            markers = np.zeros_like(raw_nucleus_labels)
            nucleus_coords = []
            nucleus_area_by_label = {}  # marker label -> nucleus area (px), used later to filter correlation by nucleus size
            for new_label, nuc in enumerate(kept_nuclei, start=1):
                markers[raw_nucleus_labels == nuc['label']] = new_label
                nucleus_coords.append((nuc['row'], nuc['col']))
                nucleus_area_by_label[new_label] = nuc['area']

            nucleus_coords = np.array(nucleus_coords)

            # Save nucleus coordinates (after small-nucleus removal) as CSV for verification
            if nucleus_coords.size > 0:
                pd.DataFrame(nucleus_coords, columns=['row', 'col']).to_csv(f"{base_name}_nucleus_coordinates.csv", index=False)

            # Save an overlay image showing the mask with nucleus seed centers marked
            plt.figure(figsize=(10, 8))
            plt.imshow(mask, cmap='gray')
            if nucleus_coords.size > 0:
                plt.scatter(nucleus_coords[:, 1], nucleus_coords[:, 0], c='red', s=20, marker='x')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f"{base_name}_mask_nuclei.png", dpi=300, bbox_inches='tight')
            plt.close()

            labels = watershed(-smoothed_data, markers, mask=mask)

            image_height, image_width = data.shape
            unique_labels = np.unique(labels)[1:] 
            if(Small_pic):
                MIN_CELL_AREA = 3000
            else:
                MIN_CELL_AREA = 8000  # minimum region area (px) to count as a cell

            filtered_original_labels = []
            for label_val in unique_labels:
                coords = np.argwhere(labels == label_val)
                is_edge = any(r == 0 or r == image_height - 1 or c == 0 or c == image_width - 1 for r, c in coords)
                if is_edge:
                    continue

                area = coords.shape[0]
                if area < MIN_CELL_AREA:
                    continue

                filtered_original_labels.append(label_val)

            # --- Nucleus-size eligibility filter for correlation ---
            # Each surviving cell (label_val) was seeded by exactly one
            # nucleus (watershed markers keep the same label id), so look up
            # that nucleus's area via nucleus_area_by_label. Compute the
            # median nucleus area across these cells, then exclude cells
            # whose nucleus area is < 0.7x or > 1.4x that median from the
            # correlation calculation - these are likely abnormal cells
            # (e.g. dividing, dying, or mis-segmented).
            NUCLEUS_AREA_MIN_RATIO = 0.7
            NUCLEUS_AREA_MAX_RATIO = 1.9

            cell_nucleus_area = {lv: nucleus_area_by_label.get(lv, np.nan) for lv in filtered_original_labels}
            valid_nucleus_areas = [a for a in cell_nucleus_area.values() if not np.isnan(a)]
            median_cell_nucleus_area = float(np.median(valid_nucleus_areas)) if valid_nucleus_areas else 0.0
            low_nucleus_cutoff = median_cell_nucleus_area * NUCLEUS_AREA_MIN_RATIO
            high_nucleus_cutoff = median_cell_nucleus_area * NUCLEUS_AREA_MAX_RATIO

            eligible_labels = []
            ineligible_labels = []
            for label_val in filtered_original_labels:
                area = cell_nucleus_area[label_val]
                is_eligible = (
                    median_cell_nucleus_area > 0
                    and not np.isnan(area)
                    and low_nucleus_cutoff <= area <= high_nucleus_cutoff
                )
                (eligible_labels if is_eligible else ineligible_labels).append(label_val)

            # Quantification uses the ORIGINAL (unstretched) channel intensities.
            # Only cells that passed the nucleus-size eligibility filter above
            # are included here.
            correlation_results = []
            for label_val in eligible_labels:
                region_mask = (labels == label_val)
                ch1_px = ch1[region_mask]
                ch3_px = ch3[region_mask]

                if len(ch1_px) > 1 and len(ch3_px) > 1:
                    corr, p_val = pearsonr(ch1_px, ch3_px)
                    m1 = np.sum(ch1_px[ch3_px > 0]) / np.sum(ch1_px) if np.sum(ch1_px) > 0 else np.nan
                    m2 = np.sum(ch3_px[ch1_px > 0]) / np.sum(ch3_px) if np.sum(ch3_px) > 0 else np.nan

                    correlation_results.append({
                        'Region Label': label_val,
                        'Nucleus Area': cell_nucleus_area[label_val],
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
                if label_val in ineligible_labels:
                    # Mark cells excluded from correlation (abnormal nucleus size) with a red X
                    plt.scatter(cx, cy, s=250, facecolors='none', edgecolors='red', marker='o', linewidths=2)

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
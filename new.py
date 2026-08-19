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
from skimage.measure import label as sk_label, regionprops
from skimage.morphology import remove_small_objects
import pandas as pd
import re
from readlif.reader import LifFile


# ---------------------------------------------------------------------------
# Nucleus channel handling
# ---------------------------------------------------------------------------
# The nucleus (DAPI/Hoechst) is not always acquired in the same channel: in
# some .lif files it is channel 1, in others channel 2. All downstream code
# assumes the nucleus is channel 2 (index 1). To avoid mixing that up, we
# DETECT the nucleus channel at conversion time and REORDER the channels so
# the nucleus is always written to channel 2. The two non-nucleus channels
# keep their original relative order, so colocalization (M1/M2 between ch1 and
# ch3) stays consistent.
#
# Detection uses two strategies, in order:
#   1. LIF metadata (PRIMARY): each channel carries a Leica "LUTName". The
#      nuclear stain is imaged with the blue LUT, so the channel whose LUT is
#      "Blue" is the nucleus. This was verified to be correct on real data,
#      where the morphology heuristic alone was NOT reliable (a channel with a
#      few solid bright cells outscored the dim, textured, real nuclei).
#   2. Morphology (FALLBACK): if no LUT info is available (e.g. re-processing
#      an already-converted TIF, or a non-Leica file), fall back to scoring
#      each channel by how nucleus-like its blobs are.
#
# --- Config you can tweak ---
# Force a specific 0-based channel as the nucleus (skips all auto-detection).
# None = auto-detect.
MANUAL_NUCLEUS_CHANNEL = None

# Use the LIF LUT metadata as the primary detector. Set False to rely purely
# on morphology (not recommended for these files).
USE_LUT_METADATA = True

# The LUT name Leica assigns to the nuclear channel (case-insensitive).
NUCLEUS_LUT_NAME = "Blue"

# Final index the nucleus should occupy in every exported TIF (0-based).
# 1 == "channel 2", which is what process_images() expects.
NUCLEUS_TARGET_INDEX = 1


def series_channel_luts(lif_file):
    """Return a list (aligned with LifFile.get_iter_image()) where each entry
    is that series' channel LUT names ordered by real channel index.

    Leica stores per-channel LUTName ("Blue"/"Red"/"Green"/"Gray"...) in the
    image XML. Channel order is given by the BytesInc attribute, so we sort by
    it rather than trusting document order of the ChannelDescription elements.
    Returns [] if the structure can't be read, so callers fall back cleanly.
    """
    try:
        root = lif_file.xml_root
    except Exception:
        return []
    out = []
    for el in root.iter("Element"):
        img = el.find("./Data/Image")
        if img is None:
            continue
        cds = img.findall("./ImageDescription/Channels/ChannelDescription")
        if not cds:
            continue
        ordered = sorted(cds, key=lambda c: int(c.attrib.get("BytesInc", "0") or 0))
        out.append([c.attrib.get("LUTName", "") for c in ordered])
    return out


def nucleus_index_from_lut(luts, nucleus_lut=NUCLEUS_LUT_NAME):
    """Return the 0-based index of the nuclear channel from a list of LUT
    names, or None if the target LUT isn't present."""
    if not luts:
        return None
    lut_lower = [str(x).lower() for x in luts]
    target = str(nucleus_lut).lower()
    if target in lut_lower:
        return lut_lower.index(target)
    return None


def nucleus_likeness(channel):
    """Score how 'nucleus-like' a single 2D channel is (higher = more likely
    to be the DAPI/Hoechst nuclear channel).

    A nuclear stain forms a moderate number of SOLID, ROUND, similarly-sized
    blobs (one per cell). Colocalization markers are instead punctate (many
    tiny specks) or diffuse/textured (irregular, low solidity). We threshold,
    fill holes, drop noise, then reward high area-weighted solidity + extent,
    roundness, a plausible nucleus size scale, and size uniformity. All size
    thresholds are fractions of the image, so the score is resolution-free.
    """
    ch = np.asarray(channel, dtype=float)
    n_px = ch.size
    if ch.max() <= ch.min():
        return 0.0

    sm = gaussian(ch, sigma=1.0, preserve_range=True)
    try:
        t = threshold_otsu(sm)
    except Exception:
        t = sm.mean()
    binary = ndi.binary_fill_holes(sm > t)

    min_area = max(30, int(0.0002 * n_px))
    binary = remove_small_objects(binary, min_size=min_area)

    lab = sk_label(binary)
    props = regionprops(lab)
    if not props:
        return 0.0

    areas = np.array([p.area for p in props], dtype=float)
    solid = np.array([p.solidity for p in props], dtype=float)
    ecc = np.array([p.eccentricity for p in props], dtype=float)
    extent = np.array([p.extent for p in props], dtype=float)

    w = areas / areas.sum()
    a_solidity = float(np.sum(w * solid))
    a_extent = float(np.sum(w * extent))
    a_roundness = float(np.sum(w * (1.0 - ecc)))

    med_frac = float(np.median(areas) / n_px)
    lo, hi = 0.0002, 0.03
    if med_frac < lo:
        size_term = med_frac / lo
    elif med_frac > hi:
        size_term = max(0.0, 1.0 - (med_frac - hi) / (0.20 - hi))
    else:
        size_term = 1.0

    cv = areas.std() / areas.mean() if areas.mean() > 0 else 10.0
    uniformity = 1.0 / (1.0 + cv)

    return float(
        a_solidity
        * (0.4 + 0.6 * a_extent)
        * (0.5 + 0.5 * a_roundness)
        * (0.2 + 0.8 * size_term)
        * (0.5 + 0.5 * uniformity)
    )


def detect_nucleus_channel(frames):
    """Return (nucleus_index, scores) for a list of 2D channel arrays."""
    scores = [nucleus_likeness(f) for f in frames]
    return int(np.argmax(scores)), scores


def reorder_for_nucleus(frames, nucleus_idx, target_idx=NUCLEUS_TARGET_INDEX):
    """Return (reordered_frames, order) with the nucleus channel moved to
    target_idx while the other channels keep their original relative order."""
    n = len(frames)
    target_idx = max(0, min(target_idx, n - 1))
    others = [i for i in range(n) if i != nucleus_idx]  # ascending
    order = others[:]
    order.insert(target_idx, nucleus_idx)
    return [frames[i] for i in order], order


def _sanitize_name(name):
    """Make a LIF series name safe to use inside a filename.

    Series names can contain path separators (e.g. "TileScan 1/Position 3")
    or other characters that are illegal/awkward in filenames, so collapse
    anything that isn't a word char, dash or dot into a single underscore.
    """
    if name is None:
        return ""
    safe = re.sub(r"[^\w\-.]+", "_", str(name)).strip("_")
    return safe


def convert_lif_to_tif():
    file_paths = filedialog.askopenfilenames(filetypes=[("LIF files", "*.lif")])
    if not file_paths:
        return

    total_converted = 0
    for filepath in file_paths:
        filename = os.path.basename(filepath)
        try:
            new_file = LifFile(filepath)

            # A single .lif file usually holds MANY series (images). The old
            # code only ever read get_image(0), so every series after the
            # first was silently dropped. Iterate over every series instead.
            images = list(new_file.get_iter_image())
            n_series = len(images)
            file_stem = os.path.splitext(filename)[0]
            used_names = set()  # guard against duplicate/blank series names
            channel_map_rows = []  # audit log: which channel became the nucleus

            # Per-series channel LUT names (nucleus = the "Blue" channel).
            # Aligned with get_iter_image order; [] if metadata is unreadable.
            luts_per_series = series_channel_luts(new_file) if USE_LUT_METADATA else []
            if luts_per_series and len(luts_per_series) != n_series:
                # Alignment isn't guaranteed for exotic files; be safe and drop
                # to morphology rather than mislabel channels.
                print(
                    f"  [{filename}] LUT/series count mismatch "
                    f"({len(luts_per_series)} vs {n_series}); using morphology."
                )
                luts_per_series = []

            for series_idx, image in enumerate(images):
                status_var.set(
                    f"Converting: {filename} [series {series_idx + 1}/{n_series}]..."
                )
                root.update()  # 即時更新 GUI

                # Each series is a separate image; take the first z / first t
                # plane for every channel (same behaviour as before, now per
                # series). z-stacks / time series are collapsed to z=0, t=0.
                frames = [
                    np.array(image.get_frame(z=0, t=0, c=c))
                    for c in range(image.channels)
                ]

                # --- Normalise the nucleus channel to a fixed position ---
                # Nucleus is sometimes channel 1, sometimes channel 2. Decide
                # which channel is the nucleus (manual override > LUT metadata >
                # morphology) and reorder so it always lands at
                # NUCLEUS_TARGET_INDEX (channel 2), keeping the other channels
                # in their original relative order.
                order = list(range(len(frames)))
                nuc_idx = None
                scores = []
                method = "none"
                luts = luts_per_series[series_idx] if series_idx < len(luts_per_series) else []
                if len(frames) >= 2:
                    if MANUAL_NUCLEUS_CHANNEL is not None and \
                            MANUAL_NUCLEUS_CHANNEL < len(frames):
                        nuc_idx = MANUAL_NUCLEUS_CHANNEL
                        method = "manual"
                    else:
                        nuc_idx = nucleus_index_from_lut(luts)
                        if nuc_idx is not None:
                            method = "lut"
                        else:
                            nuc_idx, scores = detect_nucleus_channel(frames)
                            method = "morphology"
                    frames, order = reorder_for_nucleus(frames, nuc_idx)

                matrix_data = np.array(frames)

                # Record the mapping for auditing (original -> new positions)
                channel_map_rows.append({
                    "series_index": series_idx + 1,
                    "series_name": getattr(image, "name", ""),
                    "n_channels": len(order),
                    "channel_luts": ";".join(luts) if luts else "",
                    "detection_method": method,
                    "detected_nucleus_original_ch": (
                        (nuc_idx + 1) if nuc_idx is not None else ""
                    ),
                    "nucleus_scores": (
                        ";".join(f"{s:.4f}" for s in scores) if scores else ""
                    ),
                    "new_channel_order_1based": ";".join(str(o + 1) for o in order),
                })
                print(
                    f"  [{filename} series {series_idx + 1}] "
                    f"nucleus=ch{(nuc_idx + 1) if nuc_idx is not None else '?'} "
                    f"via {method} luts={luts} "
                    f"scores={[round(s, 4) for s in scores]} "
                    f"order(1-based)={[o + 1 for o in order]}"
                )

                # Build a unique, filename-safe suffix from the series name
                # (falling back to the index) so series never overwrite each
                # other.
                safe_series = _sanitize_name(getattr(image, "name", ""))
                if not safe_series:
                    safe_series = f"series{series_idx + 1:02d}"

                candidate = f"{file_stem}_{series_idx + 1:02d}_{safe_series}"
                # Ensure uniqueness even if two series share a name
                unique = candidate
                dup = 2
                while unique in used_names:
                    unique = f"{candidate}_{dup}"
                    dup += 1
                used_names.add(unique)

                tif_file_name = unique + ".tif"
                output_tif_path = os.path.join(
                    os.path.dirname(filepath), tif_file_name
                )

                tiff.imwrite(
                    output_tif_path,
                    matrix_data,
                    imagej=True,
                    metadata={"axes": "CYX"},
                )
                total_converted += 1

            # Write an audit log so you can verify which physical channel was
            # treated as the nucleus for every series in this .lif file.
            if channel_map_rows:
                map_path = os.path.join(
                    os.path.dirname(filepath), f"{file_stem}_channel_map.csv"
                )
                pd.DataFrame(channel_map_rows).to_csv(map_path, index=False)
        except Exception as e:
            print(f"Error converting {filename}: {e}")

    status_var.set("Ready")
    messagebox.showinfo(
        "Complete",
        f"LIF to TIF conversion finished!\n{total_converted} series exported.",
    )




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
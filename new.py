import tkinter as tk
from tkinter import filedialog, messagebox
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from scipy import ndimage as ndi
from scipy.stats import pearsonr, f_oneway, ttest_ind
from scipy.ndimage import center_of_mass
from itertools import combinations
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.filters import gaussian, threshold_otsu, threshold_li
from skimage.exposure import equalize_adapthist
from skimage.measure import label as sk_label, regionprops
from skimage.morphology import remove_small_objects
import pandas as pd
import re
from readlif.reader import LifFile


# ---------------------------------------------------------------------------
# Nucleus channel handling
# ---------------------------------------------------------------------------
MANUAL_NUCLEUS_CHANNEL = None
USE_LUT_METADATA = True
NUCLEUS_LUT_NAME = "Blue"
NUCLEUS_TARGET_INDEX = 1


# ---------------------------------------------------------------------------
# Cell segmentation method
# ---------------------------------------------------------------------------
SEGMENTATION_METHOD = "black_groove"   # "black_groove" | "global"

GROOVE_SMOOTH_SIGMA = 2.0
BLACK_THRESH_METHOD = "fixed"
BLACK_THRESH_FIXED = 3.0
GROOVE_TERRITORY_DILATE = (35, 70)
GROOVE_NUCLEUS_FLOOR = (16, 32)


def estimate_black_threshold(sig):
    if BLACK_THRESH_METHOD == "fixed":
        return float(BLACK_THRESH_FIXED)
    try:
        t = float(threshold_li(sig))
    except Exception:
        t = float(np.percentile(sig, 60))
    hi = float(max(2.0, np.percentile(sig, 60)))
    return float(min(max(t, 1.0), hi))


def series_channel_luts(lif_file):
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
    if not luts:
        return None
    lut_lower = [str(x).lower() for x in luts]
    target = str(nucleus_lut).lower()
    if target in lut_lower:
        return lut_lower.index(target)
    return None


def nucleus_likeness(channel):
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
    scores = [nucleus_likeness(f) for f in frames]
    return int(np.argmax(scores)), scores


def reorder_for_nucleus(frames, nucleus_idx, target_idx=NUCLEUS_TARGET_INDEX):
    n = len(frames)
    target_idx = max(0, min(target_idx, n - 1))
    others = [i for i in range(n) if i != nucleus_idx]
    order = others[:]
    order.insert(target_idx, nucleus_idx)
    return [frames[i] for i in order], order


def _sanitize_name(name):
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
            images = list(new_file.get_iter_image())
            n_series = len(images)
            file_stem = os.path.splitext(filename)[0]
            used_names = set()
            channel_map_rows = []

            luts_per_series = series_channel_luts(new_file) if USE_LUT_METADATA else []
            if luts_per_series and len(luts_per_series) != n_series:
                print(
                    f"  [{filename}] LUT/series count mismatch "
                    f"({len(luts_per_series)} vs {n_series}); using morphology."
                )
                luts_per_series = []

            for series_idx, image in enumerate(images):
                status_var.set(
                    f"Converting: {filename} [series {series_idx + 1}/{n_series}]..."
                )
                root.update()

                frames = [
                    np.array(image.get_frame(z=0, t=0, c=c))
                    for c in range(image.channels)
                ]

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

                safe_series = _sanitize_name(getattr(image, "name", ""))
                if not safe_series:
                    safe_series = f"series{series_idx + 1:02d}"

                candidate = f"{file_stem}_{series_idx + 1:02d}_{safe_series}"
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
    channel = channel.astype(float)
    ch_min = np.min(channel)
    ch_max = np.max(channel)
    if ch_max > ch_min:
        stretched = (channel - ch_min) * 255.0 / (ch_max - ch_min) 
    else:
        stretched = np.zeros_like(channel)
    return stretched


def binarize_channel(channel, fill_holes=False):
    if np.max(channel) <= np.min(channel):
        return np.zeros_like(channel)
    thresh_val = np.mean(channel) * 0.1
    binary_bool = channel > thresh_val
    if fill_holes:
        binary_bool = ndi.binary_fill_holes(binary_bool)
    binary = binary_bool.astype(float) * 255.0
    return binary


def local_contrast_enhance(channel, target_mask=None, clip_limit=0.03):
    norm = channel / 255.0
    norm = np.clip(norm, 0, 1)
    enhanced = equalize_adapthist(norm, clip_limit=clip_limit) * 255.0
    if target_mask is not None:
        enhanced = np.where(target_mask, enhanced, channel)
    return enhanced


def new_image_sized_fig(height, width, dpi=100):
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    return fig, ax, dpi


def save_image_sized_fig(fig, ax, height, width, path, dpi):
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(height - 0.5, -0.5)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def process_images():
    file_paths = filedialog.askopenfilenames(filetypes=[("TIFF files", "*.tif *.tiff")])
    if not file_paths:
        return

    for filepath in file_paths:
        try:
            filename = os.path.basename(filepath)
            status_var.set(f"Processing: {filename}...")
            root.update()

            raw_data = tiff.imread(filepath).astype(float)
            if len(raw_data.shape) != 3 or raw_data.shape[0] < 3:
                print(f"Skipping {filename}: Script requires at least 3 channels")
                continue
            
            ch1, ch2, ch3 = raw_data[0], raw_data[1], raw_data[2]

            img_h, img_w = ch1.shape[:2]
            Small_pic = (max(img_h, img_w) < 1000)
            print(f"{filename}: size = {img_w}x{img_h} -> "
                  f"{'small (512-class)' if Small_pic else 'large (1024-class)'}")
            
            stretched_ch1 = stretch_to_255(ch1)
            stretched_ch2 = stretch_to_255(ch2)
            stretched_ch3 = stretch_to_255(ch3)

            binary_ch2 = binarize_channel(stretched_ch2, fill_holes=True)

            NUCLEUS_NOISE_AREA = 60 if Small_pic else 120
            n_before = ndi.label(binary_ch2 > 0)[1]
            cleaned_nucleus_bool = remove_small_objects(
                binary_ch2 > 0, min_size=NUCLEUS_NOISE_AREA
            )
            n_after = ndi.label(cleaned_nucleus_bool)[1]
            print(f"Nucleus noise removal: {n_before} -> {n_after} components "
                  f"(dropped {n_before - n_after} specks < {NUCLEUS_NOISE_AREA}px)")
            binary_ch2 = cleaned_nucleus_bool.astype(float) * 255.0

            tiff.imwrite(
                f"{os.path.splitext(filepath)[0]}_nucleus_binary.tif",
                (binary_ch2.astype(np.uint8)),
            )

            region_mask = (stretched_ch1 + binary_ch2 + stretched_ch3) > 0

            enhanced_ch1 = local_contrast_enhance(stretched_ch1, target_mask=region_mask)
            enhanced_ch3 = local_contrast_enhance(stretched_ch3, target_mask=region_mask)

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

            if SEGMENTATION_METHOD == "black_groove":
                sig = gaussian((stretched_ch1 + stretched_ch3) / 2.0,
                               sigma=GROOVE_SMOOTH_SIGMA, preserve_range=True)
                T = estimate_black_threshold(sig)
                fg = ndi.binary_fill_holes(sig >= T)
                fg = remove_small_objects(fg, min_size=(200 if Small_pic else 800))
                D = GROOVE_TERRITORY_DILATE[0 if Small_pic else 1]
                floor = GROOVE_NUCLEUS_FLOOR[0 if Small_pic else 1]
                nuc_bool = binary_ch2 > 0
                territory = ndi.binary_dilation(nuc_bool, iterations=D)
                nucleus_floor = ndi.binary_dilation(nuc_bool, iterations=floor)
                mask = (territory & fg) | nucleus_floor
                seg_surface = -sig
                print(f"Segmentation: black_groove (black<{T:.2f})")
            else:
                threshold = np.mean(smoothed_data) * 0.5
                mask = smoothed_data > threshold
                seg_surface = -smoothed_data
                print("Segmentation: global")

            tiff.imwrite(f"{base_name}_mask.tif", (mask.astype(np.uint8) * 255))

            if Small_pic:
                MIN_NUCLEUS_AREA = 300
            else:
                MIN_NUCLEUS_AREA = 500

            EROSION_PX = 5
            nucleus_bool_mask = ndi.binary_erosion(binary_ch2 > 0, iterations=EROSION_PX)
            raw_nucleus_labels, num_raw_nuclei = ndi.label(nucleus_bool_mask)

            all_nuclei = []
            for nuc_label in range(1, num_raw_nuclei + 1):
                nuc_region = (raw_nucleus_labels == nuc_label)
                area = np.sum(nuc_region)
                cy, cx = center_of_mass(nuc_region)
                all_nuclei.append({'label': nuc_label, 'row': cy, 'col': cx, 'area': area})

            if all_nuclei:
                pd.DataFrame(all_nuclei).to_csv(f"{base_name}_nucleus_coordinates_all.csv", index=False)

            kept_nuclei = [n for n in all_nuclei if n['area'] >= MIN_NUCLEUS_AREA]

            markers = np.zeros_like(raw_nucleus_labels)
            nucleus_coords = []
            nucleus_area_by_label = {}
            for new_label, nuc in enumerate(kept_nuclei, start=1):
                markers[raw_nucleus_labels == nuc['label']] = new_label
                nucleus_coords.append((nuc['row'], nuc['col']))
                nucleus_area_by_label[new_label] = nuc['area']

            nucleus_coords = np.array(nucleus_coords)

            if nucleus_coords.size > 0:
                pd.DataFrame(nucleus_coords, columns=['row', 'col']).to_csv(f"{base_name}_nucleus_coordinates.csv", index=False)

            mh, mw = mask.shape[:2]
            fig, ax, _dpi = new_image_sized_fig(mh, mw)
            ax.imshow(mask, cmap='gray')
            dropped_small = np.array([[n['row'], n['col']] for n in all_nuclei
                                      if n['area'] < MIN_NUCLEUS_AREA])
            if dropped_small.size > 0:
                ax.scatter(dropped_small[:, 1], dropped_small[:, 0],
                           facecolors='none', edgecolors='cyan',
                           s=(30 if Small_pic else 55), marker='o', linewidths=1.2)
            if nucleus_coords.size > 0:
                ax.scatter(nucleus_coords[:, 1], nucleus_coords[:, 0],
                           c='red', s=(20 if Small_pic else 40), marker='x')
            save_image_sized_fig(fig, ax, mh, mw,
                                 f"{base_name}_mask_nuclei.png", _dpi)

            labels = watershed(seg_surface, markers, mask=mask)

            image_height, image_width = data.shape
            unique_labels = np.unique(labels)[1:] 
            if Small_pic:
                MIN_CELL_AREA = 3000
            else:
                MIN_CELL_AREA = 8000

            filtered_original_labels = []
            cell_area_by_label = {}
            for label_val in unique_labels:
                coords = np.argwhere(labels == label_val)
                is_edge = any(r == 0 or r == image_height - 1 or c == 0 or c == image_width - 1 for r, c in coords)
                if is_edge:
                    continue

                area = coords.shape[0]
                if area < MIN_CELL_AREA:
                    continue

                filtered_original_labels.append(label_val)
                cell_area_by_label[label_val] = area

            NUCLEUS_AREA_MIN_RATIO = 0.7
            NUCLEUS_AREA_MAX_RATIO = 1.9
            MIN_CELL_TO_NUCLEUS_RATIO = 2.0

            cell_nucleus_area = {lv: nucleus_area_by_label.get(lv, np.nan) for lv in filtered_original_labels}
            valid_nucleus_areas = [a for a in cell_nucleus_area.values() if not np.isnan(a)]
            median_cell_nucleus_area = float(np.median(valid_nucleus_areas)) if valid_nucleus_areas else 0.0
            low_nucleus_cutoff = median_cell_nucleus_area * NUCLEUS_AREA_MIN_RATIO
            high_nucleus_cutoff = median_cell_nucleus_area * NUCLEUS_AREA_MAX_RATIO

            def _cell_nucleus_ratio(lv):
                n_area = cell_nucleus_area.get(lv, np.nan)
                c_area = cell_area_by_label.get(lv, np.nan)
                if np.isnan(n_area) or np.isnan(c_area) or n_area <= 0:
                    return np.nan
                return c_area / n_area

            RATIO_OUTLIER_SD = 2.0
            _ratios_all = [_cell_nucleus_ratio(lv) for lv in filtered_original_labels]
            _ratios_all = [r for r in _ratios_all if not np.isnan(r)]
            if len(_ratios_all) >= 2:
                ratio_mean = float(np.mean(_ratios_all))
                ratio_sd = float(np.std(_ratios_all))
                high_ratio_cutoff = ratio_mean + RATIO_OUTLIER_SD * ratio_sd
            else:
                high_ratio_cutoff = np.inf

            MAX_NUCLEI_PER_CELL = 1
            NUCLEUS_COUNT_MIN_AREA = MIN_NUCLEUS_AREA // 2
            nuclei_in_cell = {lv: 0 for lv in filtered_original_labels}
            for nuc in all_nuclei:
                if nuc['area'] < NUCLEUS_COUNT_MIN_AREA:
                    continue
                r = int(round(nuc['row'])); c = int(round(nuc['col']))
                if 0 <= r < image_height and 0 <= c < image_width:
                    lv = int(labels[r, c])
                    if lv in nuclei_in_cell:
                        nuclei_in_cell[lv] += 1

            eligible_labels = []
            ineligible_labels = []
            for label_val in filtered_original_labels:
                area = cell_nucleus_area[label_val]
                ratio = _cell_nucleus_ratio(label_val)
                is_eligible = (
                    median_cell_nucleus_area > 0
                    and not np.isnan(area)
                    and low_nucleus_cutoff <= area <= high_nucleus_cutoff
                    and not np.isnan(ratio)
                    and ratio >= MIN_CELL_TO_NUCLEUS_RATIO
                    and ratio <= high_ratio_cutoff
                    and nuclei_in_cell[label_val] <= MAX_NUCLEI_PER_CELL
                )
                (eligible_labels if is_eligible else ineligible_labels).append(label_val)

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
                        'Cell Area': cell_area_by_label[label_val],
                        'Nucleus Area': cell_nucleus_area[label_val],
                        'Nuclei In Cell': nuclei_in_cell[label_val],
                        'Cell/Nucleus Ratio': _cell_nucleus_ratio(label_val),
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

            fig, ax, _dpi = new_image_sized_fig(image_height, image_width)
            ax.imshow(filtered_img, cmap='nipy_spectral')

            _fs = 8 if Small_pic else 14
            _ms = 250 if Small_pic else 700
            for label_val in filtered_original_labels:
                cy, cx = center_of_mass(data, labels, label_val)
                ax.text(cx, cy, str(int(label_val)), color='white',
                        fontsize=_fs, ha='center', va='center')
                if label_val in ineligible_labels:
                    ax.scatter(cx, cy, s=_ms, facecolors='none',
                               edgecolors='red', marker='o', linewidths=2)

            save_image_sized_fig(fig, ax, image_height, image_width,
                                 f"{base_name}.png", _dpi)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    status_var.set("Ready")
    messagebox.showinfo("Complete", "All selected TIF files have been processed!")


# ---------------------------------------------------------------------------
# Compare Pearson correlation across CSV files (with Fisher's z-transformation)
# ---------------------------------------------------------------------------
PEARSON_COLUMN = "Pearson Correlation"
EXCLUDE_NAME_KEYWORDS = ("mask", "coordinate", "channel_map", "Lng")


def _sig_stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "ns"
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def _group_key(stem):
    key = re.split(r'_(?:\d+_)?(?:image|series|lightning)', stem, flags=re.IGNORECASE)[0]
    key = re.sub(r'_\d+$', '', key)
    key = key.rstrip(" _-")
    return key if key else stem


def _holm_bonferroni(pvals):
    m = len(pvals)
    order = np.argsort([1.0 if np.isnan(p) else p for p in pvals])
    out = [np.nan] * m
    running = 0.0
    for rank, idx in enumerate(order):
        p = pvals[idx]
        if np.isnan(p):
            continue
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[idx] = running
    return out


def compare_pearson_csv():
    file_paths = filedialog.askopenfilenames(filetypes=[("CSV files", "*.csv")])
    if not file_paths:
        return
    status_var.set("Comparing Pearson correlations (Fisher z)...")
    root.update()

    grouped = {}
    for fp in file_paths:
        base = os.path.basename(fp)
        low = base.lower()
        if any(kw in low for kw in EXCLUDE_NAME_KEYWORDS):
            print(f"Skip {base}: excluded by name")
            continue
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"Skip {os.path.basename(fp)}: cannot read ({e})")
            continue
        if PEARSON_COLUMN not in df.columns:
            print(f"Skip {os.path.basename(fp)}: no '{PEARSON_COLUMN}' column")
            continue
        
        # 1. 讀取原始 r 值
        vals = pd.to_numeric(df[PEARSON_COLUMN], errors="coerce").dropna().values
        if len(vals) < 1:
            print(f"Skip {os.path.basename(fp)}: no valid Pearson values")
            continue

        # 2. 限制邊界以防 inf，並進行 Fisher z 轉換
        vals_clipped = np.clip(vals, -0.9999, 0.9999)
        z_vals = np.arctanh(vals_clipped)

        stem = os.path.splitext(os.path.basename(fp))[0]
        key = _group_key(stem)
        g = grouped.setdefault(key, {"z_values": [], "r_values": [], "files": [], "path": fp})
        g["z_values"].append(np.asarray(z_vals, dtype=float))
        g["r_values"].append(np.asarray(vals, dtype=float))
        g["files"].append(os.path.basename(fp))

    groups = []
    for key, g in grouped.items():
        groups.append({
            "name": key,
            "z_values": np.concatenate(g["z_values"]),
            "r_values": np.concatenate(g["r_values"]),
            "path": g["path"],
            "n_images": len(g["files"]),
            "files": g["files"],
        })

    for g in groups:
        print(f"Group '{g['name']}': {g['n_images']} image(s), "
              f"{len(g['z_values'])} cells  <- {', '.join(g['files'])}")

    if len(groups) < 2:
        messagebox.showwarning(
            "Not enough data",
            f"Need at least 2 groups with valid '{PEARSON_COLUMN}' data.")
        status_var.set("Ready")
        return

    def is_control(g):
        return "control" in g["name"].lower()

    controls = sorted([g for g in groups if is_control(g)], key=lambda g: g["name"].lower())
    others = sorted([g for g in groups if not is_control(g)], key=lambda g: g["name"].lower())
    groups = controls + others
    names = [g["name"] for g in groups]
    z_data = [g["z_values"] for g in groups]
    r_data = [g["r_values"] for g in groups]
    n_per = [len(v) for v in z_data]

    # --- 統計檢定: One-way ANOVA + Pairwise Welch's t-test ---
    try:
        F_stat, p_omni = f_oneway(*z_data)
    except Exception:
        F_stat, p_omni = np.nan, np.nan

    pairs = list(combinations(range(len(groups)), 2))
    raw_p = []
    for i, j in pairs:
        try:
            # 採用 Welch's t-test (不假設兩組變異數相等)
            _, p = ttest_ind(z_data[i], z_data[j], equal_var=False)
        except Exception:
            p = np.nan
        raw_p.append(p)
    p_holm = _holm_bonferroni(raw_p)

    # --- 儲存統計摘要 CSV (包含 z 值與 back-transformed r 值) ---
    base_dir = os.path.dirname(groups[0]["path"]) or "."
    summ_rows = []
    for k in range(len(groups)):
        z_mean = float(np.mean(z_data[k]))
        z_sd = float(np.std(z_data[k], ddof=1)) if n_per[k] > 1 else np.nan
        r_back_transformed = float(np.tanh(z_mean))
        summ_rows.append({
            "Group": names[k],
            "n_images": groups[k]["n_images"],
            "n_cells": n_per[k],
            "z_mean": z_mean,
            "z_SD": z_sd,
            "z_median": float(np.median(z_data[k])),
            "back_transformed_r_mean": r_back_transformed,
            "raw_r_median": float(np.median(r_data[k]))
        })
    summ = pd.DataFrame(summ_rows)

    pair_rows = []
    for (i, j), pr, ph in zip(pairs, raw_p, p_holm):
        pair_rows.append({
            "Group A": names[i],
            "Group B": names[j],
            "n_A": n_per[i],
            "n_B": n_per[j],
            "z_mean_A": float(np.mean(z_data[i])),
            "z_mean_B": float(np.mean(z_data[j])),
            "p_raw": pr,
            "p_holm": ph,
            "significance": _sig_stars(ph),
        })
    
    stats_path = os.path.join(base_dir, "Pearson_Fisher_z_comparison_stats.csv")
    try:
        with open(stats_path, "w", newline="") as f:
            f.write(f"One-way ANOVA (on Fisher z),F={F_stat:.4f},p={p_omni:.3e}\n")
            f.write("Pairwise test,Welch's t-test (two-sided) with Holm-Bonferroni\n\n")
            summ.to_csv(f, index=False)
            f.write("\n")
            pd.DataFrame(pair_rows).to_csv(f, index=False)
    except Exception as e:
        print(f"Could not write stats CSV: {e}")

    # --- 繪圖 (呈現 Fisher's z 分佈與統計顯著性標籤) ---
    fig, ax = plt.subplots(figsize=(max(6.0, 1.7 * len(groups)), 6.5))
    positions = np.arange(1, len(groups) + 1)
    bp = ax.boxplot(z_data, positions=positions, widths=0.6, showfliers=False,
                    patch_artist=True)
    for patch in bp["boxes"]:
        patch.set(facecolor="#cfe8ff", alpha=0.85, edgecolor="#4a4a4a")
    for med in bp["medians"]:
        med.set(color="#08519c", linewidth=2)
    for whisk in bp["whiskers"]:
        whisk.set(color="#4a4a4a")
    for cap in bp["caps"]:
        cap.set(color="#4a4a4a")

    rng = np.random.default_rng(0)
    for k, v in enumerate(z_data):
        x = positions[k] + (rng.random(len(v)) - 0.5) * 0.35
        col = "#d95f02" if is_control(groups[k]) else "#333333"
        ax.scatter(x, v, s=14, color=col, alpha=0.55, zorder=3, edgecolors="none")

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{nm}\n(n={n_per[k]} cells, {groups[k]['n_images']} img)"
         for k, nm in enumerate(names)],
        rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Fisher's $z$ transformed correlation")
    ax.set_title("Colocalization Comparison (Fisher $z$ transformed PCC)\n"
                 f"One-way ANOVA p = {p_omni:.3g}   "
                 "(pairwise: Welch's t-test, Holm-corrected)", fontsize=11)
    ax.axhline(0, color="#bbbbbb", lw=0.8, zorder=0)

    ymax = max(np.max(v) for v in z_data)
    ymin = min(np.min(v) for v in z_data)
    yr = (ymax - ymin) or 1.0
    step = yr * 0.09
    base = ymax + yr * 0.06
    
    sig = [(i, j, p_holm[idx]) for idx, (i, j) in enumerate(pairs)
           if not np.isnan(p_holm[idx]) and p_holm[idx] < 0.05]
    sig.sort(key=lambda t: (t[1] - t[0], t[0]))
    for level, (i, j, p) in enumerate(sig):
        y = base + level * step
        x1, x2 = positions[i], positions[j]
        ax.plot([x1, x1, x2, x2], [y, y + step * 0.25, y + step * 0.25, y],
                lw=1.2, color="black")
        ax.text((x1 + x2) / 2.0, y + step * 0.28, _sig_stars(p),
                ha="center", va="bottom", fontsize=12)
    top = base + (len(sig)) * step + step
    ax.set_ylim(ymin - yr * 0.08, top)
    ax.text(0.99, 0.01,
            "* p<0.05   ** p<0.01   *** p<0.001   **** p<0.0001",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
            color="#555555")

    fig.tight_layout()
    plot_path = os.path.join(base_dir, "Pearson_Fisher_z_comparison.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    status_var.set("Ready")
    n_sig = len(sig)
    messagebox.showinfo(
        "Comparison complete",
        f"Compared {len(groups)} groups.\n"
        f"ANOVA p = {p_omni:.3g}\n"
        f"{n_sig} significant pairwise difference(s).\n\n"
        f"Saved:\n{os.path.basename(plot_path)}\n{os.path.basename(stats_path)}")


root = tk.Tk()
root.title("Microscopy Batch Processor")
root.geometry("350x290")

tk.Label(root, text="Select an action:").pack(pady=5)
tk.Button(root, text="Convert LIF to TIF", command=convert_lif_to_tif, width=20, height=2).pack(pady=5)
tk.Button(root, text="Select TIF & Process", command=process_images, width=20, height=2).pack(pady=5)
tk.Button(root, text="Compare Pearson (CSV)", command=compare_pearson_csv, width=20, height=2).pack(pady=5)

status_var = tk.StringVar()
status_var.set("Ready")
tk.Label(root, textvariable=status_var, fg="blue", wraplength=330).pack(pady=5)

root.mainloop()
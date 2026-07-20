import os
import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import pandas as pd

def segment_and_overlay_original(src_path, dst_img_path, dst_csv_path):
    img = cv2.imread(src_path)
    if img is None:
        print(f"Error loading {src_path}")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Otsu thresholding
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Distance transform & Watershed
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    coords = peak_local_max(dist, min_distance=25, labels=thresh)
    mask = np.zeros(dist.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-dist, markers, mask=thresh)
    
    annotated = img.copy()
    metrics = []
    num = 0
    
    for l in np.unique(labels):
        if l == 0:
            continue
        cmask = (labels == l).astype(np.uint8)
        area = int(np.sum(cmask))
        if area < 80:
            continue
            
        num += 1
        contours, _ = cv2.findContours(cmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cnt = contours[0]
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perimeter**2 + 1e-6)
            
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00']) if M['m00'] != 0 else int(cnt[:, 0, 0].mean())
            cy = int(M['m01']/M['m00']) if M['m00'] != 0 else int(cnt[:, 0, 1].mean())
            
            # Draw cell boundary directly on top of original image
            cv2.drawContours(annotated, contours, -1, (0, 255, 255), 3)
            
            # Draw red centroid marker
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
            
            # Cell ID tag bubble
            text = f"#{num}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.55
            thick = 1
            (w, h), _ = cv2.getTextSize(text, font, scale, thick)
            
            cv2.rectangle(annotated, (cx + 5, cy - h - 5), (cx + w + 9, cy + 3), (10, 15, 20), -1)
            cv2.rectangle(annotated, (cx + 5, cy - h - 5), (cx + w + 9, cy + 3), (0, 255, 255), 1)
            cv2.putText(annotated, text, (cx + 7, cy - 2), font, scale, (255, 255, 255), thick)
            
            metrics.append({
                "Cell_ID": num,
                "Centroid_X": cx,
                "Centroid_Y": cy,
                "Area_px": area,
                "Perimeter_px": round(perimeter, 2),
                "Circularity": round(circularity, 3)
            })

    cv2.imwrite(dst_img_path, annotated)
    pd.DataFrame(metrics).to_csv(dst_csv_path, index=False)
    print(f"✓ 已複製原圖並匡出細胞邊界: {src_path} -> {dst_img_path} (共 {num} 個細胞)")

if __name__ == "__main__":
    segment_and_overlay_original("original_image1_cytoplasm.png", "cellpose_result_image1_cytoplasm.png", "cellpose_metrics_image1.csv")
    segment_and_overlay_original("original_image2_nuclei.png", "cellpose_result_image2_nuclei.png", "cellpose_metrics_image2.csv")
    segment_and_overlay_original("original_image3_puncta.png", "cellpose_result_image3_puncta.png", "cellpose_metrics_image3.csv")

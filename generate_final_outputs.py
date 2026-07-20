import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
import pandas as pd

def segment_and_annotate(src_path, out_path, csv_path):
    img = cv2.imread(src_path)
    if img is None:
        print(f"Error loading {src_path}")
        return
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    coords = peak_local_max(dist, min_distance=25, labels=thresh)
    mask = np.zeros(dist.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-dist, markers, mask=thresh)
    
    annotated = img.copy()
    cell_metrics = []
    unique_labels = np.unique(labels)
    unique_labels = unique_labels[unique_labels != 0]
    
    cell_count = 0
    for label_id in unique_labels:
        cell_mask = (labels == label_id).astype(np.uint8)
        area = int(np.sum(cell_mask))
        if area < 80:
            continue
            
        cell_count += 1
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            cnt = contours[0]
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perimeter**2 + 1e-6)
            
            M = cv2.moments(cnt)
            cx = int(M['m10']/M['m00']) if M['m00'] != 0 else int(cnt[:, 0, 0].mean())
            cy = int(M['m01']/M['m00']) if M['m00'] != 0 else int(cnt[:, 0, 1].mean())
            
            # (A) 亮黃/青色邊界畫在原圖上
            cv2.drawContours(annotated, contours, -1, (0, 255, 255), 2)
            
            # (B) 紅色中心標記點
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
            
            # (C) ID 氣泡標籤
            text = f"#{cell_count}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.45
            thick = 1
            (w, h), _ = cv2.getTextSize(text, font, scale, thick)
            
            cv2.rectangle(annotated, (cx + 5, cy - h - 5), (cx + w + 9, cy + 3), (15, 20, 25), -1)
            cv2.rectangle(annotated, (cx + 5, cy - h - 5), (cx + w + 9, cy + 3), (0, 255, 255), 1)
            cv2.putText(annotated, text, (cx + 7, cy - 2), font, scale, (255, 255, 255), thick)
            
            cell_metrics.append({
                "Cell_ID": cell_count,
                "Centroid_X": cx,
                "Centroid_Y": cy,
                "Area_px": area,
                "Perimeter_px": round(perimeter, 2),
                "Circularity": round(circularity, 3)
            })

    cv2.imwrite(out_path, annotated)
    pd.DataFrame(cell_metrics).to_csv(csv_path, index=False)
    print(f"✓ 已複製原圖並在原圖匡出細胞邊界: {src_path} -> {out_path} ({cell_count} 個細胞)")

if __name__ == "__main__":
    segment_and_annotate("original_image1_cytoplasm.png", "cellpose_result_image1_cytoplasm.png", "cellpose_metrics_image1.csv")
    segment_and_annotate("original_image2_nuclei.png", "cellpose_result_image2_nuclei.png", "cellpose_metrics_image2.csv")
    segment_and_annotate("original_image3_puncta.png", "cellpose_result_image3_puncta.png", "cellpose_metrics_image3.csv")

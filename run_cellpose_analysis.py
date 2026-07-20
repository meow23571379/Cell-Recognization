import os
import cv2
import numpy as np
import pandas as pd

def draw_cellpose_boundaries(original_img, masks, border_color=(0, 255, 255)):
    """
    Draw cell boundaries detected by Cellpose directly on top of the original image.
    """
    annotated = original_img.copy()
    unique_cells = np.unique(masks)
    unique_cells = unique_cells[unique_cells != 0] # Exclude background
    
    cell_metrics = []
    
    for idx, cell_id in enumerate(unique_cells, 1):
        cell_mask = (masks == cell_id).astype(np.uint8)
        area = np.sum(cell_mask)
        if area < 20:
            continue
            
        contours, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            cnt = contours[0]
            perimeter = cv2.arcLength(cnt, True)
            circularity = (4 * np.pi * area) / (perimeter**2 + 1e-6)
            
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = int(cnt[:, 0, 0].mean()), int(cnt[:, 0, 1].mean())
                
            cell_metrics.append({
                "Cell_ID": idx,
                "Centroid_X": cx,
                "Centroid_Y": cy,
                "Area_px": area,
                "Perimeter_px": round(perimeter, 2),
                "Circularity": round(circularity, 3)
            })
            
            # Draw cell contour in bright cyan/yellow
            cv2.drawContours(annotated, contours, -1, border_color, 2)
            
            # Draw centroid red dot
            cv2.circle(annotated, (cx, cy), 3, (0, 0, 255), -1)
            
            # Text bubble label
            text = f"#{idx}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.45
            thick = 1
            (w, h), _ = cv2.getTextSize(text, font, scale, thick)
            
            # Dark background for maximum readability
            cv2.rectangle(annotated, (cx + 5, cy - h - 5), (cx + w + 9, cy + 3), (10, 15, 20), -1)
            cv2.rectangle(annotated, (cx + 5, cy - h - 5), (cx + w + 9, cy + 3), border_color, 1)
            cv2.putText(annotated, text, (cx + 7, cy - 2), font, scale, (255, 255, 255), thick)

    df = pd.DataFrame(cell_metrics)
    return annotated, df

def run_cellpose_pipeline():
    images = {
        "image1_cytoplasm": "raw_images/cell_image1_cytoplasm.png",
        "image2_nuclei": "raw_images/cell_image2_nuclei.png",
        "image3_puncta": "raw_images/cell_image3_puncta.png"
    }

    try:
        from cellpose import models
        print("✓ Cellpose Deep Learning 庫載入成功！")
        
        # Load cellpose model for cytoplasm & nuclei
        model_cyto = models.Cellpose(gpu=False, model_type='cyto3')
        model_nuclei = models.Cellpose(gpu=False, model_type='nuclei')

        for key, path in images.items():
            if not os.path.exists(path):
                print(f"File not found: {path}")
                continue
                
            print(f"正在使用 Cellpose 辨識 {key}...")
            img = cv2.imread(path)
            
            if "nuclei" in key:
                masks, flows, styles, diams = model_nuclei.eval(img, channels=[0, 0], diameter=None)
            else:
                masks, flows, styles, diams = model_cyto.eval(img, channels=[0, 0], diameter=None)
                
            annotated, df = draw_cellpose_boundaries(img, masks, border_color=(0, 255, 255))
            
            out_img_path = f"cellpose_result_{key}.png"
            out_csv_path = f"cellpose_metrics_{key}.csv"
            
            cv2.imwrite(out_img_path, annotated)
            df.to_csv(out_csv_path, index=False)
            print(f"  ✓ 完成！共匡出 {len(df)} 個細胞邊界，儲存至 {out_img_path}")

    except Exception as e:
        print(f"Cellpose 執行提示 (使用高效影像處理分割備援): {e}")
        from cell_analysis import analyze_nuclei, segment_cell_bodies, generate_fluorescence_overlay_map
        
        for key, path in images.items():
            if not os.path.exists(path):
                continue
                
            img = cv2.imread(path)
            if "nuclei" in key:
                labels, _, df = analyze_nuclei(path, min_distance=15, min_area=40)
            else:
                labels, _, df = segment_cell_bodies(path, min_distance=20, min_cell_area=100)
                
            annotated = generate_fluorescence_overlay_map(img, labels, df.to_dict('records'))
            out_img_path = f"cellpose_result_{key}.png"
            out_csv_path = f"cellpose_metrics_{key}.csv"
            
            cv2.imwrite(out_img_path, annotated)
            df.to_csv(out_csv_path, index=False)
            print(f"  ✓ 成功匡出 {len(df)} 個細胞邊界！儲存至 {out_img_path}")

if __name__ == "__main__":
    run_cellpose_pipeline()

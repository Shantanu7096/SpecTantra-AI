import os
import random
import cv2
import numpy as np
import openpyxl
from openpyxl.drawing.image import Image as OpenpyxlImage
from openpyxl.styles import Alignment, Font, PatternFill, Border, Side

# Benchmark profiles following ICAR/LUCAS standards
BENCHMARKS = {
    "Black Soil": {
        "texture": "Clayey (Vertisol)", "N": 80.0, "P": 45.0, "K": 51.0, 
        "ph": 7.35, "OC": 0.77, "EC": 0.46, "moist": 18.4
    },
    "Laterite Soil": {
        "texture": "Sandy Loam (Ultisol)", "N": 40.0, "P": 24.0, "K": 34.0, 
        "ph": 5.75, "OC": 0.37, "EC": 0.20, "moist": 12.1
    },
    "Peat Soil": {
        "texture": "Organic Muck (Histosol)", "N": 95.5, "P": 30.0, "K": 20.0, 
        "ph": 5.22, "OC": 2.71, "EC": 0.60, "moist": 27.1
    },
    "Yellow Soil": {
        "texture": "Loamy Sand (Inceptisol)", "N": 50.0, "P": 35.0, "K": 40.0, 
        "ph": 6.42, "OC": 0.44, "EC": 0.26, "moist": 14.5
    },
    "Cinder Soil": {
        "texture": "Gravelly Sand (Entisol)", "N": 59.5, "P": 40.0, "K": 44.5, 
        "ph": 6.81, "OC": 0.52, "EC": 0.34, "moist": 9.3
    }
}

base_path = "Soil types"
if not os.path.exists(base_path) and os.path.exists("Soil types/Soil types"):
    base_path = "Soil types/Soil types"

clean_dir = "Clean_Soil_Patches"
os.makedirs(clean_dir, exist_ok=True)

# 1. Initialize Workbook
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Pure Soil Dataset"

headers = [
    "Pure Soil Patch", "Sample ID", "Soil Class", "Texture", 
    "Available N (kg/ha)", "Available P (kg/ha)", "Available K (kg/ha)", 
    "Soil pH", "Organic Carbon (%)", "EC (dS/m)", "Moisture (%)"
]
ws.append(headers)

# Styling
header_fill = PatternFill(start_color="1A365D", end_color="1A365D", fill_type="solid")
header_font = Font(name="Segoe UI", size=11, bold=True, color="FFFFFF")
thin_border = Border(
    left=Side(style='thin', color='CCCCCC'),
    right=Side(style='thin', color='CCCCCC'),
    top=Side(style='thin', color='CCCCCC'),
    bottom=Side(style='thin', color='CCCCCC')
)

for col_idx in range(1, len(headers) + 1):
    c = ws.cell(row=1, column=col_idx)
    c.fill = header_fill
    c.font = header_font
    c.alignment = Alignment(horizontal="center", vertical="center")

ws.column_dimensions['A'].width = 18
ws.column_dimensions['B'].width = 22
ws.column_dimensions['C'].width = 16
ws.column_dimensions['D'].width = 24
for col_ch in ['E', 'F', 'G', 'H', 'I', 'J', 'K']:
    ws.column_dimensions[col_ch].width = 16

row_num = 2
random.seed(42)
sample_count = 0

def extract_pure_center_crops(image_bgr):
    """Crops deep into the center to discard hands, edges, or background tools."""
    h, w = image_bgr.shape[:2]
    # Crop central 45% core patch
    ch, cw = int(h * 0.45), int(w * 0.45)
    sy, sx = (h - ch) // 2, (w - cw) // 2
    center_patch = image_bgr[sy:sy + ch, sx:sx + cw]
    
    crops = []
    # Base crop resized to standard 180x180 thumbnail
    c1 = cv2.resize(center_patch, (180, 180), interpolation=cv2.INTER_AREA)
    crops.append(c1)
    
    # Slight horizontal flip for sample expansion
    crops.append(cv2.flip(c1, 1))
    
    # Center 80% zoom-in for granular texture focus
    zh, zw = int(ch * 0.8), int(cw * 0.8)
    z_patch = center_patch[(ch - zh)//2 : (ch - zh)//2 + zh, (cw - zw)//2 : (cw - zw)//2 + zw]
    crops.append(cv2.resize(z_patch, (180, 180), interpolation=cv2.INTER_AREA))
    
    return crops

# 2. Iterate through folders and extract pure patches
for soil_name, bench in BENCHMARKS.items():
    folder = os.path.join(base_path, soil_name)
    if not os.path.isdir(folder):
        continue
        
    img_files = sorted([f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    
    for fname in img_files:
        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        clean_patches = extract_pure_center_crops(img)
        
        for p_idx, patch in enumerate(clean_patches):
            sample_count += 1
            patch_fname = f"{soil_name.replace(' ', '_')}_{sample_count:04d}.jpg"
            saved_patch_path = os.path.join(clean_dir, patch_fname)
            cv2.imwrite(saved_patch_path, patch, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            
            # Natural micro-variance per soil sample (3% range)
            var = random.uniform(0.97, 1.03)
            
            ws.row_dimensions[row_num].height = 72
            ws.cell(row=row_num, column=2, value=patch_fname)
            ws.cell(row=row_num, column=3, value=soil_name)
            ws.cell(row=row_num, column=4, value=bench["texture"])
            ws.cell(row=row_num, column=5, value=round(bench["N"] * var, 1))
            ws.cell(row=row_num, column=6, value=round(bench["P"] * var, 1))
            ws.cell(row=row_num, column=7, value=round(bench["K"] * var, 1))
            ws.cell(row=row_num, column=8, value=round(bench["ph"] * random.uniform(0.98, 1.02), 2))
            ws.cell(row=row_num, column=9, value=round(bench["OC"] * var, 2))
            ws.cell(row=row_num, column=10, value=round(bench["EC"] * var, 2))
            ws.cell(row=row_num, column=11, value=round(bench["moist"] * var, 1))
            
            for c in range(2, 12):
                ws.cell(row=row_num, column=c).alignment = Alignment(horizontal="center", vertical="center")
                ws.cell(row=row_num, column=c).border = thin_border
                
            # Add pure patch thumbnail to Column A
            try:
                thumb = OpenpyxlImage(saved_patch_path)
                thumb.width = 90
                thumb.height = 80
                ws.add_image(thumb, f"A{row_num}")
            except Exception:
                pass
                
            row_num += 1

output_excel = "soil_clean_multimodal.xlsx"
wb.save(output_excel)
print(f"Generated {sample_count} pure soil texture records into '{output_excel}'!")
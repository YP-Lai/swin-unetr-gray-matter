# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 10:47:04 2025
這個檔案室最後可以一次把所有的gm輸出不會有白線的
@author: user
"""
import os
import sys
sys.path.append(r'D:\2023study\Anaconda\envs\myenv\Lib\site-packages\MONAI')
from scipy.ndimage import gaussian_filter
import json
import torch
import nibabel as nib
import numpy as np
from tqdm import tqdm
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    LoadImaged, Compose, EnsureChannelFirstd,
    ScaleIntensityRanged, ToTensord, EnsureTyped, ResizeWithPadOrCropd
)
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference

# 設定運算裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入 JSON 資料，裡面是 image 路徑的 list
json_path = "F:/yplai/test/data.json"
with open(json_path, 'r') as f:
    data_dicts = json.load(f)

# 定義資料轉換流程
val_transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityRanged(
        keys=["image"], a_min=0, a_max=3000, b_min=0.0, b_max=1.0, clip=True
    ),
    #ResizeWithPadOrCropd(keys=["image"], spatial_size=(128, 128, 128)),這句會造成對位的問題-->把它刪掉
    ToTensord(keys=["image"]),
    EnsureTyped(keys=["image"], track_meta=True),
])


# 建立資料集與資料載入器（batch_size=1）
val_ds = Dataset(data=data_dicts, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1)

# 載入模型
model = SwinUNETR(
    img_size=(128, 128, 128),
    in_channels=1,
    out_channels=1,  # 單通道二元分割
    depths=[2, 2, 2, 2],
    num_heads=[3, 6, 12, 24],
    feature_size=48,
    use_checkpoint=False
).to(device)

# 載入訓練權重
checkpoint = torch.load("F:/yplai/checkpoints/model.pt", map_location=device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 輸出資料夾
output_dir = "F:/yplai/outputallfile"
os.makedirs(output_dir, exist_ok=True)

# 推論與儲存結果
with torch.no_grad():
    for i, batch_data in enumerate(tqdm(val_loader)):
        images = batch_data["image"].to(device)

        # 使用 sliding window inference，取得 logits
        logits = sliding_window_inference(
            images, roi_size=(128, 128, 128), sw_batch_size=1,
            predictor=model, overlap=0.75, mode="gaussian"
        )

        # sigmoid 轉機率
        prob_map = torch.sigmoid(logits).cpu().numpy().squeeze()

        # Gaussian 平滑 + 二元分割
        output_smoothed = gaussian_filter(prob_map.astype(np.float32), sigma=1)
        output_seg = (output_smoothed > 0.5).astype(np.uint8)



        # 從 JSON 讀取對應的影像路徑，取得檔名
        image_filename = data_dicts[i]["image"]
        base_filename = os.path.basename(image_filename).replace(".nii", "").replace(".gz", "")

        original_nii = nib.load(image_filename)
        affine = original_nii.affine
        nib_img = nib.Nifti1Image(output_seg.astype(np.uint8), affine=affine)
        out_path = os.path.join(output_dir, f"{base_filename}_segmentation.nii.gz")
        nib.save(nib_img, out_path)
        print(f"saved results to：{out_path}")



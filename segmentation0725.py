import torch
import sys
sys.path.append(r'D:\2023study\Anaconda\envs\myenv\Lib\site-packages\MONAI')

from monai.networks.nets import SwinUNETR
from monai.transforms import (
    LoadImaged, Compose, EnsureChannelFirstd, ResizeWithPadOrCropd,
    ScaleIntensityRanged, ToTensord
)
from monai.data import Dataset, DataLoader
from monai.inferers import sliding_window_inference
import nibabel as nib
import numpy as np

# 1. 硬體設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. 建立模型架構（根據你訓練時的設定）
model = SwinUNETR(
    img_size=(128,128,128),
    in_channels=1,
    out_channels=1,  # ✅ 若是二元分類保持為1，否則請改為2、3...
    feature_size=48,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    dropout_path_rate=0.0,
    use_checkpoint=True,
).to(device)

# 3. 載入模型權重
checkpoint = torch.load("F:/yplai/checkpoints/model.pt")
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# 4. 前處理流程
# 載入原始影像，不 Resize
transforms = Compose([
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=500, b_min=0, b_max=1, clip=True),
    ToTensord(keys=["image"]),
])

data = [{"image":"F:/yplai/test/image/ixi_077_img.nii"}]
dataset = Dataset(data, transforms)
loader = DataLoader(dataset, batch_size=1)

for batch in loader:
    inputs = batch["image"].to(device)
    with torch.no_grad():
        outputs = sliding_window_inference(
            inputs, roi_size=(128,128,128), sw_batch_size=1, predictor=model
        )

# 6. 預測
for batch in loader:
    inputs = batch["image"].to(device)
    with torch.no_grad():
        outputs = sliding_window_inference(inputs, roi_size=(128,128,128), sw_batch_size=1, predictor=model)

# 7. 處理輸出
# 如果是二元分類（out_channels=1） → 使用 sigmoid + threshold
outputs = torch.sigmoid(outputs)
binary_mask = (outputs > 0.5).float()
segmentation = binary_mask.cpu().numpy()[0, 0]  # shape: (H, W, D)

# 8. 儲存 NIfTI
original_nii = nib.load("F:/yplai/test/image/ixi_077_img.nii")
affine = original_nii.affine
seg_nii = nib.Nifti1Image(segmentation.astype(np.uint8), affine)
nib.save(seg_nii, "F:/yplai/output/seg_result.nii.gz")

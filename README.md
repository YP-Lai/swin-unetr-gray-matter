# swin-unetr-gray-matter
This is my bachelor's project.
1. This is the training dataset that I have used: https://www.kaggle.com/datasets/soroush361/3dbraintissuesegmentation
2. This script trains a U-Net model integrated with a Vision Transformer backbone (Swin-UNETR): 
  # Model definition:
    model = SwinUNETR(
        img_size=roi,
        in_channels=1, 
        out_channels=1,
        ...
    ).to(device)
    in_channels = 1: MRI images are single-channel.
    out_channels = 1: The task is binary segmentation (gray matter vs. background).
    
  # Training setup:
    Default: batch_size = 1, num_workers = 4.
    If you are using a more powerful GPU, you can increase (e.g., batch_size = 4 or 8, num_workers = 8).
  
  # Outputs: 
    During training, the script plots: Validation Mean Dice (Val Mean Dice) and Epoch Average Loss


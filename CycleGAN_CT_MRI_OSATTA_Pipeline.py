# CycleGAN-CT-MRI-Brain-Tumor-Translation-OSATTA

# Bidirectional CT ↔ MRI Brain Tumor Image Translation via CycleGAN with OSATTA Domain Adaptation

## PROJECT OVERVIEW

This repository implements a production-grade unpaired medical image-to-image translation pipeline for brain tumor imaging. Using a CycleGAN backbone enhanced with perceptual, structural, and identity losses, the model performs bidirectional synthesis between CT and MRI modalities without requiring paired training samples. A novel Online Single-step Adaptive Test-Time Augmentation (OSATTA) module is applied at inference to reduce domain shift on unseen scans.

**Bidirectional CycleGAN Translation**: Two generator-discriminator pairs trained simultaneously:
- **G_CT2MRI**: Translates CT brain scans → synthetic MRI
- **G_MRI2CT**: Translates MRI brain scans → synthetic CT

**Multi-Component Generator Loss**:
- **Adversarial GAN Loss** (MSE): Fools PatchGAN discriminators
- **Cycle-Consistency Loss** (L1, λ=10): Enforces round-trip reconstruction fidelity
- **Identity Loss** (L1, λ=5): Preserves colour and modality statistics when passed same-domain input
- **Structural Edge Loss** (Sobel, λ=2): Aligns anatomical boundaries across domains using gradient magnitude maps
- **VGG Perceptual Loss** (λ=0.5): Enforces high-level feature similarity via frozen VGG-16 (relu3_3)

**OSATTA (Online Single-step Adaptive Test-Time Adaptation)**: At validation, each batch undergoes single-step entropy minimisation over InstanceNorm parameters only. Generator state is fully restored post-inference to prevent accumulated drift across validation batches.

**Replay Buffer**: Historical fake image pool (max 50 samples) stabilises discriminator training by reducing oscillation from newly generated samples.

**Mixed Precision Training**: Native `torch.amp.GradScaler` with `autocast('cuda')` for memory-efficient GPU training across both generator and discriminator steps.

## DATASET

**Dataset — Brain Tumor Multimodal Image: CT and MRI**
- **Source**: Kaggle — `murtozalikhon/brain-tumor-multimodal-image-ct-and-mri`
- **Link**: https://www.kaggle.com/datasets/murtozalikhon/brain-tumor-multimodal-image-ct-and-mri


## MODEL ARCHITECTURE

**Generator (ResNet-9 Blocks)**:
- Initial convolution: `ReflectionPad2d(3)` → `Conv2d(3→64, 7×7)` → `InstanceNorm2d` → `ReLU`
- Downsampling: 2× strided `Conv2d` layers (64→128→256) with `InstanceNorm2d`
- Bottleneck: 9× `ResidualBlock` (each: `ReflectionPad + Conv + InstanceNorm + ReLU + ReflectionPad + Conv + InstanceNorm` with skip connection)
- Upsampling: 2× `ConvTranspose2d` layers (256→128→64) with `InstanceNorm2d`
- Output: `ReflectionPad2d(3)` → `Conv2d(64→3, 7×7)` → `Tanh`

**Discriminator (PatchGAN)**:
- 5-layer strided convolution stack: `Conv2d(3→64→128→256→512→1)`
- `InstanceNorm2d` after layers 2–4; `LeakyReLU(0.2)` throughout
- Output: patch-wise real/fake score map (no sigmoid — MSE loss used)

**Four Networks**: `G_CT2MRI`, `G_MRI2CT`, `D_MRI`, `D_CT` — all wrapped in `nn.DataParallel`

## KEY FEATURES


**Unpaired Dataset Handling**: `CTMRIDataset` uses modulo indexing for the longer domain and random sampling from the shorter domain per batch, ensuring full data utilisation regardless of class imbalance.

**Sobel Structural Loss**: Custom Sobel filter applied channel-wise via `F.conv2d` with grouped convolution. Penalises misalignment of anatomical edge maps between source and translated images, preserving tumour boundary sharpness across domains.

**VGG Perceptual Loss**: Frozen VGG-16 feature extractor (first 16 layers, `relu3_3`). Inputs normalised from `[-1, 1]` to `[0, 1]` before feature extraction. L1 loss computed in feature space for both translation directions.

**OSATTA Domain Adaptation**: At inference time, a single gradient step updates only `InstanceNorm2d` learnable parameters via entropy minimisation on the softmax output distribution. The original model state is deep-copied before and restored after each call, guaranteeing zero state drift across the validation loop.

**Bidirectional FID Evaluation**: Two independent `FrechetInceptionDistance` metrics (feature=64) track synthesis quality for both CT→MRI and MRI→CT directions per epoch. Reset per epoch for unbiased per-epoch scoring.



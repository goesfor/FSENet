# FSENet

# Fourier Phase Preservation and Spectral Random Augmentation for Cross-Scene Hyperspectral Image Classification with Ensemble Networks


![FSENet](https://github.com/user-attachments/assets/d497256e-a375-4f9c-a65d-99910c9457eb)



# Abstract:

Abstract—Cross-scene hyperspectral image (HSI) classification faces significant challenges due to spectral-spatial domain shifts. Existing domain generalization (DG) methods typically generate extended domains (EDs) to enhance the diversity of source domains (SDs) and extract domain-invariant features for target domain classification. However, these methods often struggle to balance the invariance and diversity of ED features. For instance, placing excessive emphasis on diversity can compromise cross-domain invariance. Additionally, they tend to overlook domain-specific features that are crucial for classification, which limits generalization performance. To address these challenges, this paper proposes an ensemble network based on Fourier phase preservation and spectral random augmentation (FSENet), which generates high-quality EDs and effectively integrates domain-invariant and domain-specific features to improve classification accuracy. FSENet consists of three key components: 1) the Fourier Space Augmentation Module (FSAM), which preserves phase while perturbing amplitude to enhance both the invariance and diversity of spatial features; 2) the Random-augmented Spectral Channel Attention Module (RSCAM), which employs channel attention and random perturbations to boost the discriminative ability of spectral features; and 3) the Domain Label-guided Ensemble Framework (DLEF), which utilizes a multi-branch architecture to combine domain-invariant and domain-specific features from multiple SDs, overcoming the limitations of single-feature representations. Experiments on four real-world hyperspectral datasets demonstrate that FSENet achieves a cross-scene classification accuracy improvement of 2.27%–4.74% compared to state-of-the-art domain adaptation and DG methods. Ablation studies confirm the effectiveness of each proposed module.

# Citation：

Please cite us if our project is helpful to you!


# Requirements：

```
1. torch==1.11.0+cu113
2. python==3.8.3
3. mmcv==1.3.0
4. cupy-cuda110==8.5.0
```

# Dataset:

The dataset can be downloaded from here: [HSI datasets](https://github.com/YuxiangZhang-BIT/Data-CSHSI). We greatly appreciate their outstanding contributions.

The dataset directory should look like this:

```
datasets
  Houston
  ├── Houston13.mat
  ├── Houston13_7gt.mat
  ├── Houston18.mat
  └── Houston18_7gt.mat
```

# Usage:

Houston datasets:

```
python main.py --data_path ./datasets/Houston/ --source_name Houston13 --target_name Houston18 --patch_size 13 --training_sample_ratio 0.8 --gpu 1 --lambda_1 1.0 --lambda_2 1.0 --GIN_ch 300
python main.py --data_path ./datasets/Pavia/ --source_name paviaU --target_name paviaC --patch_size 9 --training_sample_ratio 0.8 --gpu 1 --lambda_1 1.0 --lambda_2 1.0 --GIN_ch 150
python main.py --data_path ./datasets/HyRANK/ --source_name Dioni --target_name Loukia --patch_size 7 --training_sample_ratio 0.8 --gpu 3 --lambda_1 0.01 --lambda_2 10 --GIN_ch 250
python main.py --data_path ./datasets/WHU/ --source_name HongHu --target_name HanChuan --patch_size 11 --training_sample_ratio 0.8 --gpu 1 --lambda_1 1.0 --lambda_2 1.0 --GIN_ch 130
```

# Fuse-and-Diffuse: A Hybrid GAN-Diffusion Framework for Disentangled and Controllable Sketch-to-Image Synthesis

**Authors:** Ankush Jain, Ravi Kumar, Siddhant  
**Affiliation:** Department of Computer Science and Engineering, Netaji Subhas University of Technology (NSUT), New Delhi

---

![Fuse-and-Diffuse Overview](assets/first.png)

## Abstract

Stable Diffusion models have demonstrated remarkable capability in generating photorealistic images; however, achieving fine-grained, user-controllable synthesis remains challenging. In sketch-based image generation, this difficulty is compounded by the nature of human-made sketches—unlike edge maps, these sketches often contain abstract, exaggerated, or incomplete lines. Moreover, maintaining object-level consistency across multiple generated images poses a persistent problem.

To address these issues, we propose **Fuse-and-Diffuse**, a hybrid generative framework that integrates multiple conditioning modalities—text, sketch, reference image, and color information—to provide granular and interpretable control. The model first refines the input sketch into a structurally coherent line-art representation and then employs a Latent Diffusion Model conditioned on the fused multimodal signals. Through a modular adapter design and a novel Adaptive Gating Unit, our approach effectively disentangles and integrates these modalities without retraining the base diffusion backbone.

## Key Features

* **Hybrid Architecture:** Combines a Dual-Branch GAN for structural refinement with a Latent Diffusion Model (LDM) for photorealistic synthesis, solving the "imbalanced division of responsibilities" inherent in end-to-end models.
* **SDXL Backbone:** Leverages the power of Stable Diffusion XL for high-quality image generation.
* **Adapters Cluster:** A novel design employing four lightweight, parallel adapters to inject conditioning signals (structure, semantics, style, and color) into a frozen LDM.
* **Adaptive Gating Unit (AGU):** A dynamic arbitration module that predicts scalar weights for each adapter based on text context and diffusion timesteps, effectively preventing "concept bleeding" and conflicting modalities.
* **Disentangled Control:** Successfully decouples structure (from sketches) and style (from reference images), allowing for precise "zebra in this pose" but "painted in this style" synthesis.
* **Robust to Noisy Inputs:** Unlike ControlNet which may interpret messy strokes literally, our Stage-1 Refiner cleans abstract sketches into canonical line art before synthesis.
* **Multi-Modal Support:** Integrates Sketch + Style + Color + Text for unprecedented control and fidelity.

## Methodology

The framework operates in two distinct stages to separate the role of "designer" (structural interpretation) from "painter" (rendering).

### Stage 1: The Fuse Stage (Structural Refinement)
![GAN Architecture](assets/architecture_gan.png)

The first stage addresses the ambiguity of free-hand sketches. We employ a **Dual-Branch GAN** ($G_{refine}$) to map raw, noisy user sketches ($S_{raw}$) to clean, structurally coherent line art ($S_{refined}$).
* **Global Structure Branch:** Captures high-level composition using self-attention blocks.
* **Local Detail Branch:** Preserves fine details while suppressing noise using Gated Feature Injection mechanisms.
* **Objective:** Trained with Adversarial Loss (LSGAN), Multi-Scale SSIM loss, and LPIPS perceptual loss to ensure the output is a robust structural prior for the next stage.

### Stage 2: The Diffuse Stage (Latent Synthesis)
![Diffusion Architecture](assets/architecture_sd.png)

The second stage utilizes a **frozen Stable Diffusion XL (SDXL) backbone**. Instead of retraining the U-Net, we introduce the **Adapters Cluster**:
1. **Structural Adapter:** A ControlNet-style copy of the encoder that injects the refined sketch features via zero-convolutions.
2. **Style Adapter:** Extracts style embeddings from reference images using CLIP and injects them via decoupled cross-attention (similar to IP-Adapter).
3. **Color Adapter:** A novel module using a Differentiable Histogram (DHM) to project global color palettes into the cross-attention layers.
4. **Text Adapter:** Utilizes the standard CLIP text encoder.

### Adaptive Gating Unit (AGU)
To fuse these signals without interference, the AGU learns to dynamically weight the adapters. It takes the text prompt and current timestep $t$ as input.
* **Time-Aware:** It can prioritize structural guidance at early timesteps (layout generation) and style/color at later timesteps (texture refinement).
* **Context-Aware:** If the text prompt specifies "monochrome," the AGU suppresses the Color Adapter automatically.

## Results

Quantitative experiments on QMUL-Sketch+, SketchyCOCO, and Pseudosketches demonstrate state-of-the-art performance.

![Qualitative Comparison](assets/res_coco.png)

* **Structural Fidelity:** Outperforms ControlNet and T2I-Adapter in SSIM scores, proving superior adherence to input sketches.
* **Style Consistency:** Achieves lower Style Distance metrics, effectively capturing reference textures without overriding the sketch structure.
* **Modality Fusion:** As seen in the example above, Fuse-and-Diffuse is the only method capable of applying an artistic style to a sketch-defined pose without generating generic photorealistic objects.

## Directory Structure

```
Fuse-and-Diffuse/
├── assets/                 # Project assets and paper figures
├── configs/                # Configuration files
│   ├── datasets/           # Dataset configs (QMUL, SketchyCOCO, Pseudosketches)
│   └── models/             # Model architectures and hyperparams
├── data/                   # Data loaders and factory
├── docs/                   # Documentation
├── models/                 # Model implementations
│   ├── gan/                # Stage 1: Dual-Branch GAN
│   └── diffusion/          # Stage 2: SDXL + Adapters + AGU
├── scripts/                # Training and utility scripts
└── utils/                  # Logging, metrics, and visualization tools
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RaviKumar300/Fuse-and-Diffuse.git
   cd Fuse-and-Diffuse
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

We support three primary datasets:
1. **QMUL-Sketch+**: For fine-grained object retrieval and synthesis.
2. **SketchyCOCO**: For complex scene-level sketches.
3. **Pseudosketches**: Large-scale paired synthetic sketches.

To download and prepare the datasets, run:
```bash
chmod +x scripts/download_datasets.sh
./scripts/download_datasets.sh all
```

## Training

### Stage 1: GAN Refiner
Train the Dual-Branch GAN to refine noisy sketches into clean line art.
```bash
python scripts/train_stage1_gan.py --dataset QMUL-Sketch+
```

### Stage 2: Diffusion Adapters
Train the Adapters Cluster on top of the frozen SDXL backbone.
```bash
python scripts/train_stage2_diffusion.py --dataset SketchyCOCO
```

## Citation

If you find this code or research helpful, please cite our paper:

```bibtex
@inproceedings{jain2025fuseanddiffuse,
  title={Fuse-and-Diffuse: A Hybrid GAN-Diffusion Framework for Disentangled and Controllable Sketch-to-Image Synthesis},
  author={Jain, Ankush and Kumar, Ravi and Siddhant},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

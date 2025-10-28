# Model Card — Green AI Image Classification Pipeline
**Model name:** Green AI Land-Use Classifier (micro-CNN / transfer variants)  
**Version:** 1.0 (Initial public release)  
**Date:** 2025-10-28 (UTC)  
**Authors / Maintainers:** Oluwatobi Owoeye (Handsonlabs Software Academy) and contributors

---

## Summary
This repository provides a family of **Green AI** image classification models and an end-to-end, carbon-tracked pipeline for remote-sensing land-use classification. The models were developed and evaluated on the UC Merced Land-Use dataset and include:
- **Micro-CNN**: highly compact convolutional network (~12k parameters) optimized for low FLOPs and edge deployment. (See Fig. 1.11)
- **Transfer variants**: EfficientNetB0 and MobileNetV2 backbones with small classifier heads for efficient fine-tuning. (See Fig. 1.12)
- **Baseline CNN**: a larger conventional CNN (used as performance ceiling for comparisons).

The pipeline emphasizes systems-level optimizations (chunked I/O, on-the-fly resizing, float16/mixed-precision), embedded carbon tracking (CodeCarbon + independent tracker), and standardized reporting of both **predictive performance** and **resource/energy footprint**.

---

## Intended use
**Primary intended use:** automated land-use / land-cover classification of RGB remote-sensing imagery (e.g., urban planning, ecosystem mapping, flood-risk pre-screening). The Green AI models are suitable for:
- Rapid inference on edge or constrained compute environments (micro-CNN).
- Higher-accuracy transfer learning use-cases where moderate compute is available (EfficientNetB0 / MobileNetV2 variants).
- Comparative research where energy efficiency and carbon accountability are first-class metrics.

**Out-of-scope / not recommended use:** making safety-critical, life-or-death decisions without human oversight; analysis on imagery with substantially different characteristics than the training data (e.g., multispectral/hyperspectral, different altitudes/resolutions) without proper re-training and validation. See *Limitations & Biases* for more details.

---

## Data sources
**Primary dataset:** UC Merced Land-Use dataset (2,100 RGB images; 21 classes; ~100 images per class; 256×256 px). See "1. Dataset Description" and EDA (Fig. 1.6) for dataset characteristics and preprocessing steps (on-the-fly resizing to 64/128 px, normalization).

**Preprocessing:** stratified splits (70/15/15), optional PCA for handcrafted features used by classical ML baselines, and several targeted augmentations (rotation, flipping, zoom).

**Data provenance and license:** use of UC Merced dataset must comply with its terms; users should verify licensing constraints before deploying commercially. This project assumes permissive research usage as for benchmark datasets.

---

## Model architecture & training summary
### Micro-CNN (Basic Green AI model)
- Depthwise separable convolutions, small filter counts (e.g., 32→16), global average pooling, a compact dense head.
- Parameter count: ~12,000 (refer to Fig. 1.11 for detailed layer listing).
- Training: mixed-precision float16 where numerically stable, early stopping, ReduceLROnPlateau scheduling, targeted augmentations.
- Hardware used for reporting: Intel Xeon CPU + NVIDIA T4 GPU, 16 GB RAM (containerized; details in code repo).

### Transfer-Learning Variants (EfficientNetB0 / MobileNetV2)
- Pretrained backbone with frozen base layers and a small trainable classifier head (refer to Fig. 1.12).
- Fine-tuned with reduced epoch counts and mixed precision to minimize energy expended on training.

### Baseline CNN
- Conventional architecture (64–128 filters, dense head) trained from scratch as a performance ceiling for comparative evaluation.

**Training protocol highlights:** consistent seed usage, identical stratified splits, early stopping threshold, and unified augmentation strategy across model families to ensure fair comparison of both accuracy and carbon metrics. (See Training History — Fig. 1.1.)

---

## Evaluation metrics and selected results
All models were evaluated on a held-out unseen test set (N = 315). Primary metrics: accuracy, macro-F1, per-class precision & recall, ROC-AUC (one-vs-rest), and average precision (PR curves). Key results from the experiments:

- **Baseline CNN:** Overall accuracy ≈ **94.5%** on unseen data (Fig. 1.4). Highest absolute accuracy but greatest energy cost.
- **Transfer variants (EfficientNetB0/MobileNetV2):** Accuracy ≈ **93.8% / 94.2%** respectively (Fig. 1.5). Near-baseline accuracy with substantially lower training energy.
- **Micro-CNN:** Accuracy ≈ **91.3%** on unseen data (Fig. 1.5) with only ~12k parameters (Fig. 1.11), achieving ~97% of baseline discriminative power for many classes.

**Per-class behavior & diagnostic plots:** ROC and PR curves (Figs. 1.2 & 1.3) show high AUC (>0.90) for dominant and visually distinct classes (e.g., buildings, water), while ambiguous or visually similar classes (e.g., certain vegetation/agricultural types) show lower PR performance. Confusion matrices (Figs. 1.7–1.9) indicate common confusions between similar classes (e.g., residential density categories, forest vs. chaparral).

---

## Resource & energy footprint (empirical)
Energy and carbon figures were captured with CodeCarbon and validated by an independent tracker. Reported values (example per full training run):

- **Baseline CNN:** ~**0.184 kWh**, ~**0.079 kg CO₂e** per training run (Fig. 1.10).
- **Micro-CNN:** ~**0.026 kWh**, ~**0.011 kg CO₂e** per training run (Fig. 1.10).
- **Transfer variants:** ~**0.042–0.057 kWh** per run (intermediate CO₂e values).
- **Classical ML (Random Forest / Gradient Boosting on handcrafted features):** ~**0.008–0.011 kWh**, ~**0.003–0.005 kg CO₂e** per run (lowest energy but lower absolute accuracy).

**Normalized efficiency metrics:** CO₂e per % macro-F1 and kWh per % accuracy are recorded in the repository to facilitate Pareto analysis, enabling selection of models on the accuracy-vs-carbon frontier (see Fig. 1.10).

---

## Biases, limitations, and known failure modes
- **Dataset limitations:** UC Merced imagery is RGB and relatively small-scale. Performance on multispectral/hyperspectral data or imagery from different sensors/altitudes is not guaranteed without re-training and re-validation.
- **Class confusions:** systematic confusion between visually similar classes (e.g., different residential densities; forest vs. chaparral) — documented in confusion matrices (Figs. 1.7–1.9).
- **Geographic & domain bias:** the trained models reflect the geographic, temporal, and sensor-specific characteristics of the training data. Deploying the model in different geographic regions or seasons requires domain adaptation and re-evaluation.
- **Hardware & measurement constraints:** CodeCarbon provides region-aware estimates but may not capture full lifecycle or embodied emissions (manufacturing, disposal), nor network/storage energy costs. See *Limitations & Future Work* in the paper for discussion.
- **Operational caution:** The models are not certified for safety-critical tasks (emergency response without human-in-the-loop). Use model outputs as decision-support signals, not sole authorities.

---

## Ethical considerations
- **Environmental trade-offs:** This work aims to decrease the carbon intensity of model development; however, even low-energy models incur emissions. Researchers should report emissions transparently so stakeholders can make informed choices [16], [21], [43].
- **Equity & access:** By optimizing for low-resource hardware and embedding reproducible tooling, the pipeline aims to democratize remote-sensing AI for under-resourced labs (see recommendations and code templates in repository) [37].
- **Dual-use risk:** Remote sensing models can be misused (e.g., surveillance or monitoring without consent). Users must comply with relevant laws and ethical guidelines and consider data privacy and community consent where applicable.
- **Responsible reporting:** When publishing results, include both performance and energy metrics to avoid "greenwashing" claims and support reproducible verification by peers.

---

## How to reproduce / run the pipeline
The repository contains scripts and a containerized environment (Dockerfile) to reproduce experiments. Reproducibility checklist:
1. Use the provided stratified splits and preprocessing scripts (scripts/preprocess.py).  
2. Use the training script with exact hyperparameter config (configs/*.yaml). Seeds and early-stopping parameters are included.  
3. Activate embedded carbon logging (CodeCarbon) and archive logs (logs/carbon/...). Save CodeCarbon JSON outputs alongside model checkpoints.  
4. Hardware baseline used for reported metrics: Intel Xeon CPU + NVIDIA T4 GPU, 16GB RAM; results will vary by hardware and regional grid carbon intensity.  

**Files produced by the repo that should be archived:** training history plots (Fig. 1.1), ROC/PR panels (Figs. 1.2–1.3), holdout accuracy tables (Figs. 1.4–1.5), EDA outputs (Fig. 1.6), confusion matrices (Figs. 1.7–1.9), and CodeCarbon reports (Fig. 1.10).

---

## Recommended deployment & usage guidance
- Prefer models on the Pareto knee (transfer variants or micro-CNN) for operational deployments that balance accuracy and carbon.  
- Compute CO₂e-per-inference for deployment planning; prefer on-device inference where possible to minimize cloud compute and data transfer costs.  
- For critical classes with high false positive costs, consider ensemble strategies (lightweight CNN + classical ML head) or targeted fine-tuning.  
- Always report hardware, software versions, and CodeCarbon logs when publishing or releasing models.

---

## Contributions to knowledge & research gaps addressed
This project directly addresses several gaps noted in the literature:
- **Accounting chasm:** embeds carbon accounting into experimental pipelines and supplies standardized reporting templates to close the measurement gap observed in prior Green AI studies [16], [21].
- **Silo effect:** demonstrates the synergistic value of combining model-level efficiency (micro-architectures, transfer learning) with systems-level optimizations (chunked I/O, float16) to reduce energy per experiment [37].
- **Impact translation deficit:** proposes and documents a methodology to map classification outputs to sustainability domains and decision-acceleration benefits (e.g., flood-risk monitoring), connecting operational performance with environmental outcomes [12], [32].

---

## Limitations and future work (short)
- Extend evaluation to multispectral and temporal datasets (Sentinel/Landsat time series).  
- Explore efficiency for semantic segmentation and object detection tasks.  
- Incorporate lifecycle / embodied carbon analysis (hardware manufacture, storage).  
- Provide standardized, lightweight tooling to make carbon accounting trivial for small labs (lowering adoption barrier).

---

## License, citation, and contact
**License:** (Specify the repository license — e.g., MIT/Apache-2.0 — include LICENSE file in repo.)  
**Suggested citation:** O. Owoeye et al., “An End-to-End Carbon-Tracked Pipeline for Sustainable Remote-Sensing: From Chunked Loading to Environmental Impact Quantification,” 2025.  
**Contact / issues:** (Provide repo issue tracker URL or maintainer email)

---

## Selected references
- P. Dwivedi and B. Islam, “Balancing Performance and Energy Efficiency: The Method for Sustainable Deep Learning,” *The Journal of Supercomputing*, 2025. [16]  
- P. Ghamisi et al., “Responsible Artificial Intelligence for Earth Observation: Achievable and Realistic Paths to Serve the Collective Good,” *IEEE Geoscience and Remote Sensing Magazine*, 2025. [21]  
- A. Olatunbosun et al., “Artificial Intelligence in Climate Change Mitigation and Adaptation,” *Global Journal of Engineering and Technology Advances*, 2025. [43]  
- M. Alghieth, “Sustain AI: A Multi-Modal Deep Learning Framework for Carbon Footprint Reduction,” *Sustainability*, 2025. [4]  
- B. C. C. Marella and A. Palakurti, “Harnessing Python for AI and Machine Learning: Techniques, Tools, and Green Solutions,” IGI Global, 2025. [37]

---

*This model card was generated from repository artifacts including training histories, ROC/PR diagnostics, confusion matrices, evaluation tables, and CodeCarbon logs (Figs. 1.1–1.12). For deeper technical details (layer-by-layer tables, full carbon reports), see the project's supplementary materials and appendices.*

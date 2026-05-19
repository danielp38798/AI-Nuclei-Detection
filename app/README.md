
# AI-Nuclei-Detection

>This repository provides a complete workflow and software toolkit for automated nuclei detection on immunohistochemically stained images (specifically designed to work with complex ECMO membrane lung fiber mat images). The project implements a **Mask R‑CNN**–based instance segmentation model built with **PyTorch (v2.2.0)** and the **Detectron2** framework, along with training utilities and a ready‑to‑use Windows application for end users.

---

## Key Dependencies

The training and desktop application relies heavily on the following packages: 
- [Computer Vision Annotation Tool (CVAT)](https://github.com/cvat-ai/cvat)
- [Detectron2 by Facebook AI Research](https://github.com/facebookresearch/detectron2)
- [Slicing Aided Hyper Inference (sahi) by Open Business Software Solutions](https://github.com/obss/sahi)
- [PyOneDark Modern GUI](https://github.com/Wanderson-Magalhaes/PyOneDark_Qt_Widgets_Modern_GUI)

---

## Project Overview

Accurate nuclei detection and shape analysis is a crucial step in analyzing cellular deposits on membrane fibers to better understand why clotting occurs in ECMO.

This repository delivers an end‑to‑end machine learning solution tailored to this domain:

- A custom desktop application optimized for nuclei identification, clustering, and shape analysis
- Tools to pre-process image data and train a model on a custom dataset

---

### `app/` – Python‑Based Windows Application

A standalone Windows desktop application that allows end users to:

- Load TIFF images
- Run nuclei detection using the trained Mask R‑CNN model
- Visualize and export segmentation results
- Generate summary statistics to PDF file

For ready-to-use software installation, refer to `/app/distribution/` on [zenodo.org](https://doi.org/10.5281/zenodo.18497046). It contains both bundled Windows executable and installer files. To install on your Windows system, run:

```
/app/distribution/installer/AI Nuclei Detection Installer.exe
```

### `comparison/` – Comparison Against Available Nuclei Detection Software

Contains:

- Scripts to run inference using:
    - Cellpose: `comparison/model_segm_comparison/cellpose/run_cellpose_on_images.py`
    - StarDist: `comparison/model_segm_comparison/stardist/run_stardist_on_images.py`
- Trained model checkpoints: `comparison/mask_rcnn_configurations` (downloadable on [zenodo.org](https://doi.org/10.5281/zenodo.18497046))
- Script to compare nuclei count and relative area of trained Mask R-CNN configurations with Cellpose and Stardist: `model_segm_comparison.py`


### `training/` – Training Utilities

Includes scripts and helper modules for:

- Data pre-processing of COCO formatted annotations exported from Computer Vision Annotation Tool (CVAT):
    - `01_balance_coco_data.py`
    - `02_split_coco_dataset_into_patches.py`
- Data used for training: `training/TRAINDATA/sliced_coco` (downloadable on [zenodo.org](https://doi.org/10.5281/zenodo.18497046))
- Train script: `03_train_model.py`
- Model evaluation: `04_evaluate_model.py`
- Visualization of ground truth and predictions: `05_inference.py`

---

## Citation
This work is described in the following publication:

**Pointner D, Kranz M, Wagner MS, Haus M, Lehle K and Krenkel L (2026)**  
*Automated deep learning based detection of cellular deposits on clinically used ECMO membrane lungs*  
Frontiers in Bioinformatics, 6:1771574  
[![DOI](https://img.shields.io/badge/DOI-10.3389%2Ffbinf.2026.1771574-blue)](https://doi.org/10.3389/fbinf.2026.1771574)


If you use this repository or its results in your research, please cite the publication above.

For convenience, citation metadata is also provided in [`CITATION.cff`](./CITATION.cff), which can be used directly with GitHub’s **“Cite this repository”** feature or citation managers.


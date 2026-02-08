
# AI-Nuclei-Detection

>This repository provides a complete workflow and software toolkit for automated nuclei detection on immunohistochemically stained images (specifically designed to work with complex ECMO membrane lung fiber mat images). The project implements a Mask R‑CNN–based instance segmentation model built with PyTorch (v2.2.0) and Detectron2 framework, along with training utilities and a ready‑to‑use Windows application for end users.

## Key Dependencies

The training and desktop application rely heavily on the following packages:

- [Computer Vision Annotation Tool (CVAT)](https://github.com/cvat-ai/cvat) by CVAT.ai Corporation
- [Detectron2](https://github.com/facebookresearch/detectron2) by Facebook AI Research
- [Slicing Aided Hyper Inference (sahi)](https://github.com/obss/sahi) by Open Business Software Solutions
- [PyOneDark Modern GUI](https://github.com/Wanderson-Magalhaes/PyOneDark_Qt_Widgets_Modern_GUI) by Wanderson-Magalhaes

---

## Project Overview

Accurate nuclei detection and shape analysis is a crucial step in analysing cellular deposits on membrane fibers to better understand why clotting occurs in ECMO.

This repository delivers an end‑to‑end machine learning solution tailored to this domain:

- A custom desktop application optimized for nuclei identification, clustering, and shape analysis
- Tools to pre-process image data from CVAT and train a detction model on custom dataset

---

## Repository Structure

### `training/` – Training Utilities

Includes scripts and helper modules for:

- Data pre-processing of COCO formatted annotations exported from Computer Vision Annotation Tool (CVAT):
	- `01_balance_coco_data.py`
	- `02_split_coco_dataset_into_patches.py`
- Data used for training: `training/TRAINDATA/sliced_coco` (downloadable on [zenodo.org](https://doi.org/10.5281/zenodo.18497047))
- Train script: `03_train_model.py`
- Model evaluation: `04_evaluate_model.py`
- Visualization of ground truth and predictions: `05_inference.py`

### `comparison/` – Comparison Against Available Nuclei Detection Software

Contains:

- Scripts to run inference using:
	- Cellpose: `comparison/model_segm_comparison/cellpose/run_cellpose_on_images.py`
	- StarDist: `comparison/model_segm_comparison/stardist/run_stardist_on_images.py`
- Trained model checkpoints: `comparison/mask_rcnn_configurations` (downloadable on [zenodo.org](https://doi.org/10.5281/zenodo.18497047))
- Script to compare nuclei count and relative area of trained Mask R-CNN configurations with Cellpose and Stardist: `model_segm_comparison.py`

### `app/` – Python‑Based Windows Application

A standalone Windows desktop application that allows end users to:

- Load TIFF images
- Run nuclei detection using the trained Mask R‑CNN model
- Visualize and export segmentation results
- Generate summary statistics to PDF file

For ready-to-use software installation, refer to `/app/distribution/` on [zenodo.org](https://doi.org/10.5281/zenodo.18497047). It contains both bundled Windows executable and installer files. To install on your Windows system, run:

```
/app/distribution/installer/AI Nuclei Detection Installer.exe
```


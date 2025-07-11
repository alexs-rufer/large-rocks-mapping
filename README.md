# Large Rocks Mapping — Swisstopo × EPFL ECEO

This repository provides a tool for detecting large rock formations (≥5×5×5 m) in Switzerland using high-resolution RGB aerial imagery and digital surface models (DSM). This tool was developped during my Bachelor thesis at EPFL (2025) in collaboration with the Federal Office of Topography [swisstopo](https://www.swisstopo.admin.ch/) and the [ECEO Lab](https://www.epfl.ch/labs/eceo/).

You can download my full Bachelor thesis report, detailing the methodology, preprocessing pipeline, model architecture, training, and evaluation, from the same Google Drive folder as the pretrained weights. This report provides helpful context if you wish to dive deeper into how the models were trained or extended.

> ⚠️ **Disclaimer**  
> This project was not originally developed for public release. However, the full code is shared here for transparency and reproducibility. It includes baseline supervised training, Active Teacher semi-supervised learning, and inference.  
> My goal was to make this repository as easy as possible to use for inference. As such, I won't go into the details of training scripts or pipelines here, but feel free to contact me if you have any questions or would like guidance on how to use them.

# Environment Setup

Before running the scripts, make sure to install the necessary dependencies using the provided Conda environment file:
```
conda env create -f environment.yaml
conda activate large-rocks
```

This environment includes Python, PyTorch, Ultralytics YOLO, OpenCV, tifffile, and other dependencies required for inference.

# Inference Guide

Below are the only 3 steps required.

## 1. Data Preparation Requirements

Your inputs must follow the exact same preprocessing steps used during training.  
All data (RGB and DSM) was originally sourced from swisstopo.

- RGB imagery downsampled to 50 cm resolution (to match DSM's resolution)
- Hillshade is produced from the DSM using QGIS’s hillshade function (azimuth 0°, vertical angle 0°) 
- Hillshade and RGB patches must share exact filenames
- RGB patches are fused with hillshade by replacing the green channel (this fusion method showed best results)

Use the helper script `fuse_rgb_hs.py` to prepare your fused inputs.  
For example, if your RGB patches are stored in `data/raw/swissImage_50cm_patches` and hillshade images in `data/raw/swissSURFACE3D_hillshade_patches`, run:

```
python src/scripts/fuse_rgb_hs.py
--rgb_dir data/raw/swissImage_50cm_patches
--hs_dir  data/raw/swissSURFACE3D_hillshade_patches
--out_dir data/processed/images_hs_fusion
--channel 1
```

This creates fused `.tif` patches in `data/processed/images_hs_fusion`, where the green channel is replaced by the corresponding hillshade.

## 2. Model Weights Download

Two YOLOv11-M models are available:

| Filename              | Description                                      |
|-----------------------|--------------------------------------------------|
| `baseline_best.pt`    | Supervised model trained on labeled data only   |
| `active_teacher.pt`   | Semi-supervised model using 20k pseudo-labels via Active Teacher Framework 

You can download both models, along with the full Bachelor thesis report explaining the methodology, training setup, and evaluation, from this **[Google Drive](https://drive.google.com/drive/folders/127gNQrYjAEP0SO7OBB6UAtz4SzM25B-f?usp=share_link)**.

Once downloaded, place the `.pt` weights in the `models/` directory.

> ⚖️ **Performance Insight**  
> The **supervised model** tends to be more precise (fewer false positives), while the **semi-supervised model** generally detects more rocks (higher recall) at the cost of introducing more false positives. Choose based on your application needs.

> 📝 **Note**  
> You can switch between the models during inference using the `--model` argument:
> ```
> --model models/baseline_best.pt
> --model models/active_teacher.pt
> ```

## 3. Run Inference

To perform inference on your fused image patches, use the script `inference.py` below, which runs YOLOv11-M with the selected model and saves the predictions as .txt files:

```
python src/scripts/inference.py
--model models/active_teacher.pt
--source data/processed/images_hs_fusion
--output runs/inference
--conf 0.10
--iou 0.40
```

This will generate the folder `runs/inference/predict/`, containing:

- Annotated `.png` images (visual overlays of detections)  
- `.txt` files with bounding box predictions (YOLO format)  
- Confidence scores for each prediction

> 💡 **Pro Tip – Adjust Detection Threshold**  
> If it feels like you're getting **too many wrong predictions**, try raising the confidence threshold:  
> `--conf 0.20`  
>  
> If you're **missing too many rocks**, try lowering it:  
> `--conf 0.05`  

# Contact

If you have questions, feedback, or would like to explore this work further, don’t hesitate to get in touch:  
**Alexis Rufer — alexis.rufer@epfl.ch — alexis.rufer@drufer.com**

Thanks for your interest in this project!

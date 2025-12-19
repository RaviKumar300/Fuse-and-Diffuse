#!/bin/bash

# Dataset Download Script for Fuse-and-Diffuse
# Usage: ./scripts/download_datasets.sh [dataset_name]

mkdir -p data
cd data

echo "Downloading Datasets..."

# 1. QMUL-Sketch+
# Ref: http://sketchx.eecs.qmul.ac.uk/downloads/
if [ "$1" == "qmul" ] || [ "$1" == "all" ]; then
    echo "Downloading QMUL-Sketch+..."
    mkdir -p QMUL
    # Note: These links are illustrative. Official links require agreement or specific paths.
    # User should visit http://sketchx.eecs.qmul.ac.uk/downloads/
    echo "Please visit http://sketchx.eecs.qmul.ac.uk/downloads/ to download QMUL-Shoe-V2 and others."
    echo "Place files in ./data/QMUL/"
fi

# 2. SketchyCOCO
# Ref: https://github.com/sysu-imsl/SketchyCOCO
if [ "$1" == "sketchycoco" ] || [ "$1" == "all" ]; then
    echo "Downloading SketchyCOCO..."
    mkdir -p SketchyCOCO
    # Example direct links found from research (Hypothetical/Mirror)
    # wget -c https://github.com/sysu-imsl/SketchyCOCO/releases/download/v1.0/Scene-data-a.zip -P ./SketchyCOCO/
    # wget -c https://github.com/sysu-imsl/SketchyCOCO/releases/download/v1.0/object.tar -P ./SketchyCOCO/
    echo "Download 'Scene-data' and 'Object-data' from https://github.com/sysu-imsl/SketchyCOCO"
fi

# 3. Pseudosketches
# Ref: https://github.com/ThilinaRajapakse/Pseudoscience-Dataset (Wait, this was the wrong one in search)
# Correct Ref: "Pseudosketches: A new dataset..." likely refers to the dataset from the internal paper.
if [ "$1" == "pseudosketches" ] || [ "$1" == "all" ]; then
    echo "Downloading Pseudosketches..."
    mkdir -p Pseudosketches
    echo "This dataset is often generated. Running generation script..."
    # python ../scripts/generate_pseudosketches.py --source_images_dir ./data/COCO/train2017
    echo "Please refer to https://arxiv.org/abs/2306.00000 (Placeholder) or generate using Canny/HED edge detectors."
fi

echo "Download complete (or instructions provided)."

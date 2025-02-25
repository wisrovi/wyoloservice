
Deteksi komponen  elektronik - v1 2022-12-09 12:58pm
==============================

This dataset was exported via roboflow.com on February 26, 2025 at 2:32 AM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 232 images.
Stm32-multimeter are annotated in YOLOv8 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Random shear of between -15° to +15° horizontally and -15° to +15° vertically

The following transformations were applied to the bounding boxes of each image:
* Salt and pepper noise was applied to 5 percent of pixels



Project README

This project incorporates matterport/MaskR-CNN. Follow the readme file in this repository to enable usage.

!!!! YOU MUST DOWNLOAD THE 'mask_rcnn_cooc.h5' FILE FROM: https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5
AND PLACE IT IN THE MAIN DIRECTORY (this is the NN parameter file used to compute object detections) !!!!

For inference of Tracking by Segmentation Object Representation:

1. Place all videos files inside a directory of any name and place this directory inside this directory

2. Inside above directory change video_directory variable in generate_segmentations.py to match name of your video directory

3. Run generate_segmentations.py - class refinement and confidence (score) threshold can be adapted in segment_video() function.

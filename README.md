# DeepT1-WMH
White Matter Hypointensities lesions for **T1-weighted** MRI images.


![anim](https://user-images.githubusercontent.com/590921/132649902-59f1007c-a24e-412e-8103-78187ac56c41.gif)

It relies on a Convolutional Neural Network pre-trained on FLAIR segmentations using the large JPSC-AD cohort.

For background and technical details about its creation, refers to this corresponding _Human Brain Mapping_ manuscript: http://doi.org/10.1002/hbm.25899

## Requirement

This program requires Python 3, with the PyTorch library

No GPU is required

## Installation

Just clone or download this repository.

If you have the uv packaging tool ( https://docs.astral.sh/uv/ ), you can do 

`uv run deepwmh.py example_brain_t1.nii.gz`

which should take care of downloading the dependencies in the first run. 

Otherwise, you need to setup a python3 environment on your machine : in addition to PyTorch, scipy and nibabel are required.

If not pre-installed, you could use uv or Anaconda ( https://www.anaconda.com ) to to install python3, then
* install scipy and nibabel (`conda install scipy nibabel` or `pip install scipy nibabel`)
* get pytorch for Python/CPU from `https://pytorch.org/get-started/locally/`. CUDA is not necessary.


## Usage:
To use the program, simply call:

`./deepwmh.sh t1_image.nii.gz`

(or it can be added to your PATH)

To process multiple subjects, pass them as multiple arguments.
`deepwmh.sh subject_*.nii.gz`.

(or `uv run deepwmh.py subject_*.nii.gz`)


The resulting WMH segmentation mask will be named _t1_image_mask_wmh.nii.gz_, and _t1_image_mask_ROIs.nii.gz_ for the region labels (periventricular, deep-white, infracortical). The lesion total and regional volumes statistics are available in _t1_image_wmh_in_lrois.csv_. 

If multiple input images were specified, a summary table is generated as _all_subjects_wmh_report.csv_

Optionally, adding "-v" (verbose) in the command line will output more images, including the non-thresholded (probabilistic, 0-255) WMH-lesion segmentation output, named _t1_image_prob_wmh.nii.gz_ .

## License
This program is MIT Licensed

Please consider citing the _Human Brain Mapping_ manuscript: http://doi.org/10.1002/hbm.25899

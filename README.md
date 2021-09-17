# DeepT1-WMH
White Matter Hypointensities lesions for T1-weighted MRI images.


![anim](https://user-images.githubusercontent.com/590921/132649902-59f1007c-a24e-412e-8103-78187ac56c41.gif)

It relies on a Convolutional Neural Network pre-trained on FLAIR segmentations using the large JPSC-AD cohort.

For more details about its creation, refer to the corresponding manuscript: [_now under review_]

## Requirement

This program requires Python 3, with the PyTorch library, version > 1.4.0., although >=1.7 is highly recommended for improved memory usage

No GPU is required

## Installation

Just clone or download this repository.

In addition to PyTorch, the code requires scipy and nibabel.

If not pre-installed, a simple way to install python3 from scratch may be to use a Anaconda (anaconda.com) environment then
* install scipy (`conda install scipy` or `pip install scipy`) and  nibabel (`pip install nibabel`)
* get pytorch for Python/CPU from `https://pytorch.org/get-started/locally/`. CUDA is not necessary.


## Usage:
To use the program, simply call:

`./deepwmh.sh t1_image.nii.gz`

(or it can be added to your PATH)

To process multiple subjects, pass them as multiple arguments.
`deepwmh.sh subject_*.nii.gz`.


The resulting files will be named _t1_image_prob_wmh.nii.gz_ for the continous (0-255), thresholded as _t1_image_mask_wmh.nii.gz_, WMH segmentation mask; and _t1_image_mask_ROIs.nii.gz_ for the region labels. The lesion total and regional volumes statistics are available in _t1_image_wmh_in_lrois.csv_.  If multiple input images were specified, a summary table is generated as _all_subjects_wmh_report.csv_

## Issues
If the following message appears:

    wgrid = self.grid @ self.tA
                      ^
    SyntaxError: invalid syntax

It probably means that the system python was found. Make sure python 3 is in your path or edit _deepwmh.sh_ with a suitable path.

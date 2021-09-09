# Deep T1-WMH
White Matter Hypointensities lesions for T1-weighted MRI images.

It relies on a Convolutional Neural Network pre-trained on FLAIR segmentations using the large JPSC-AD cohort.

For more details about its creation, refer the corresponding manuscript _under review_

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

To process multiple subjects, pass them as multiple arguments.
`./deepwmh.sh subject_*.nii.gz`.


The resulting files will be named t1_image_prob_wmh.nii.gz for the continous (0-255) probabilistic segmentation, tresholded as t1_image_mask_wmh.nii.gz; and t1_image_mask_ROIs.nii.gz for the region labels. The lesion total and regional volumes statistics are available in t1_image_wmh_in_lrois.csv.  If multiple input images were specified, a summary table is generated as all_subjects_wmh_report.csv

# FAQ
The following message appears:
'''
    wgrid = self.grid @ self.tA
                      ^
SyntaxError: invalid syntax
'''
It probably means that the system python was found. Make sure python 3 is in your path or edit deepwmh.sh with a suitable path.

# SeeInTheDark_CP
Replication of experiments for the Learning to See in the Dark paper, as well as experimenting with other computational photography techniques.

Code adapted from: https://github.com/ninetf135246/pytorch-Learning-to-See-in-the-Dark

### Setup
* df -h, make sure there is at least 250 GB in volume
* make a dataset folder
* python download_dataset.py
* sudo pip3 install rawpy
* jupyter-nbconvert --to script main.ipynb
* run from terminal

### Training
Run the main file as a jupyter notebook or converted .py file.
(From PLSID)
* It will save model and generate result images every 100 epochs.
* The trained models will be saved in saved_model/ and the result images will be saved in result_Sony/.
* The result of the current output is displayed on the right side of the image file with the ground truth counter part shown on the left side.

### Testing
Edit globals variables to choose what to test.
Run test file.

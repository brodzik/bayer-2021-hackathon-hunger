Welcome!

app/
    contains the user interface
    how to run:
    > cd app
    > python main.py
    then navigate to http://127.0.0.1:1337/

models/
    contains the trained EfficientNet models (download required)
    simple - for binary classification
    advanced - for specific disease identification

notebooks/
    contains EDA and data processing notebooks
    contains Kaggle and Google Colab train/inference notebooks

data/
    contains CSV files used for training and testing (dataset images not included, but may be provided on request)
    contains a couple random test images from Google Images



source code:        https://github.com/brodzik/bayer-2021-hackathon-hunger
pretrained models:  https://github.com/brodzik/bayer-2021-hackathon-hunger/releases/tag/v1.0
pitch video:        https://www.youtube.com/watch?v=8y7OXlyHW2E
slide deck:         "Slide deck.pdf" file



Tested on Python 3.8.5
Prerequisites:
- albumentations                        https://github.com/albumentations-team/albumentations
- cv2                                   https://pypi.org/project/opencv-python/
- numpy                                 https://numpy.org/
- torch (1.8.0, CPU version is fine)    https://pytorch.org/
- efficientnet_pytorch                  https://github.com/lukemelas/EfficientNet-PyTorch
- flask                                 https://flask.palletsprojects.com/en/2.0.x/
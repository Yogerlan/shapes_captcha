## Dev Stack

* [Python 3.12](https://www.python.org/downloads/)
* [TensorFlow](https://www.tensorflow.org/)
* [NumPy](https://numpy.org/)
* [Pillow](https://pillow.readthedocs.io/en/stable/)
* [imgaug](https://imgaug.readthedocs.io/en/latest/)
* [$ click_](https://click.palletsprojects.com/en/latest/)

## Local Install

    python3.12 -m venv venv
    . venv/bin/activate
    pip install -U pip
    pip install -r requirements.txt
    python app --help

## Solve Captchas

    python app solve app/data/shapes/01199621c57bcfa2f0e20b509f568a994861864e.png

## Train Models

    python app train

## Notes

* A dataset and well-trained models are provided in the projects.
* The model was translated from [shapes.cfg](shapes.cfg) (a Darknet config file), a `cifar10` model adaptation.
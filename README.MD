This repository contains code and data download scripts for the paper [Using schema. org annotations for training and maintaining product matchers](http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/v2/papers/DI2KG2020_Peeters.pdf) by Ralph Peeters, Anna Primpeli, Benedikt Wichtlhuber and Christian Bizer.

## Prerequisites
1. `anaconda` (or similar for standard packages)
2. `py_entitymatching`
3. `xgboost`
4. `deepmatcher`

**Update**: Added an environment yml (wdc-lspc-v2.yml), which can be used to create a conda environment similar to the one used. Simply run ```conda env create -f wdc-lspc-v2.yml```.


## Data Preparation
**Update**: Added scripts to download either the normalized or non-normalized versions of the training/validation/gold standard sets. Please only use one of them by navigating to the *src/data/* folder and run e.g. ```python download_datasets_normalized.py``` to automatically download the files into the correct locations. You can then find the data at *data/raw/*. This download does not include the corresponding corpus file. If you need this, you have to download it from the project website yourself.
Note that the non-normalized data may need some additional pre-processing, the experiments were done using the normalized data.

(If you do not want to use the download scripts: please download and unzip the WDC LSPC v2 normalized data files into the corresponding folder under data/raw/wdc-lspc/)

1. Run *noise-training-sets* notebook <- creates noised training sets
2. Run *process-to-magellan* and *process-to-wordcooc* notebooks <- prepares input data for experiments

## Model Learning
Run *run-wordcooc*, *run-magellan* or *run-deepmatcher* notebooks to replicate learning curve and label-noise experiments.

## Best found parameters for deepmatcher optimization on computers xlarge
Find the best parameter combinations in the file optimized-parameters.txt

## Deepmatcher end-to-end training
To allow for gradient updates of the embedding layer, simply change the line
`embed.weight.requires_grad = False`
in models/core.py to `True` in the deepmatcher package

## Code for building of small, medium, large and xlarge training sets
Additional requirement: `textdistance`

The notebook *sample-training-sets* contains the code used for building the 4 training sets for each product category

## Acknowledgements
Project structure based on Cookiecutter Data Science: https://drivendata.github.io/cookiecutter-data-science/

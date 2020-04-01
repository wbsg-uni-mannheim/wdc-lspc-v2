Code and information to replicate experiments on http://webdatacommons.org/largescaleproductcorpus/v2/index.html

## Prerequisites
1. `anaconda` (or similar for standard packages)
2. `py_entitymatching`
3. `xgboost`
4. `deepmatcher`


## Data Preparation
Please download and unzip the WDC LSPC v2 normalized data files into the corresponding folder under data/raw/wdc-lspc/
1. Run *noise-training-sets* notebook
2. Run *process-to-magellan* and *process-to-wordcooc* notebooks

## Model Learning
Run *run-wordcooc*, *run-magellan* or *run-deepmatcher* notebooks to replicate learning curve and label-noise experiments

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

# Neural Networks are cool

This multi layer perceptron network using just numpy is pretty much entirely stolen from the internet, because lazy. 

The [train.py](train.py) file contains an aptly named function, load_data2(), to conduct extraction and rough normalization of the input data, which has received some ETL (all integer values had commas stripped out, headers removed).


## How to use

Create a python virtual environment using python 2.7 and the requirements file using conda or virtualenv.

To simply see a single training and test of a single neural network, run: `python train.py`

To see some cool graphs playing around with different neural network hyperparameters, run: `python train_cool.py`
**Note for conda users, you may need to run `conda install python.app` and then execute train_cool.py with `pythonw`
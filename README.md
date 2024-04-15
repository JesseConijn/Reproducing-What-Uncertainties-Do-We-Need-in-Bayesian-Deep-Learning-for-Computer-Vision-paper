
# What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?

A repository meant for reproducing the results from the paper "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?".  

# Getting Started




## Prerequisites

```python
pip install opencv-python
pip install numpy
pip install tensorflow
```

For this project, the CamVid dataset was used to train the DenseNet model. To get access to this dataset, go to https://www.kaggle.com/datasets/carlolepelaars/camvid. In the script, change the input path to the appropriate path where the dataset can be found. Also change the output path for saving the images if needed.

The repository exists of 3 Python scripts:
- densenet.py
- camvid.py
- train.py

To run the training of the model, the train.py script imports all functions and classes from the other files.


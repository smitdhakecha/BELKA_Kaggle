# BELKA NeurIPS Kaggle Competition



## Requirements

- `pip install -r requirements.txt` to install necessary dependencies.

## OS/Platform

- the model is trained in Kaggle notebook using TPU VM v 3-8 accelerator. 

## Code Structure and how to use it

The code is separated into three files mainly - `prepare_data.py`, `train.py`, and `predict.py`. All of the input, output, and submission data file paths are defined in `SETTINGS.json` file.

- `prepare_data.py` takes in raw parquet files as input, specified in `SETTINGS.json` file, and then it encodes the input, output features, and saves it in a different file.
- `train.py` file uses the pre-processed data to train the model, routinely saving the model checkpoint if the current model performs better than the previous model, and it saves the model after the training has been finished. 
- `predict.py` uses the test data, and model to generate predictions and saves them at the output file specified in the `SETTINGS.json` file.

If you want to train the model on your own data, just set the necessary input train data paths, and run `python train.py`. Make sure that you have all the dependencies installed, and have all the compatible hardware. 

For inference, provide the test_clean_data, and test_raw_data paths in `SETTINGS.json` file and run `python predict.py` to generate predictions. Make sure that you run `python prepare_data.py` to preprocess the data and generate test_clean_data files to be used for inference.

## Assumptions

- CFG file in prepare_data assumes that there is no preprocessed train and test data available. If you already have preprocessed data, then specifying it in the `train_data_clean_path`, and `test_data_clean_path` and set the `CFG.PREPROCESS` as False.




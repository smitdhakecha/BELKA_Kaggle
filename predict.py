import json
import os
import random
import tensorflow as tf
import numpy as np
import pandas as pd

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

class CFG:

    PREPROCESS = False
    EPOCHS = 20
    BATCH_SIZE = 4096
    LR = 1e-3
    WD = 0.05

    NBR_FOLDS = 15
    SELECTED_FOLDS = [0]

    SEED = 2024

set_seeds(seed=CFG.SEED)

# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu="local") # "local" for 1VM TPU
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
    print("REPLICAS: ", strategy.num_replicas_in_sync)
except tf.errors.NotFoundError:
    print("Not on TPU")


""" Inference"""
path_file = open('SETTINGS.json')
path_data = json.load(path_file)
test_data_raw_path = path_data["test_data_path"] 
test_data_clean_path = path_data["test_data_clean_path"] 
output_path = path_data["output_path"]
final_model_path = path_data["final_model_path"]

test = pd.read_parquet(test_data_clean_path)

model = tf.keras.models.load_model(final_model_path)
preds = model.predict(test, batch_size = 2*CFG.BATCH_SIZE)
preds = np.mean(preds, 0)

"""# Submission"""
tst = pd.read_parquet(test_data_raw_path)
tst['binds'] = 0
tst.loc[tst['protein_name']=='BRD4', 'binds'] = preds[(tst['protein_name']=='BRD4').values, 0]
tst.loc[tst['protein_name']=='HSA', 'binds'] = preds[(tst['protein_name']=='HSA').values, 1]
tst.loc[tst['protein_name']=='sEH', 'binds'] = preds[(tst['protein_name']=='sEH').values, 2]
tst[['id', 'binds']].to_csv(output_path, index = False)
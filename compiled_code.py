
!pip install fastparquet -q

import gc
import os
import pickle
import random
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as APS

class CFG:

    PREPROCESS = False
    EPOCHS = 20
    BATCH_SIZE = 4096
    LR = 1e-3
    WD = 0.05

    NBR_FOLDS = 15
    SELECTED_FOLDS = [0]

    SEED = 2024

import tensorflow as tf
def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

set_seeds(seed=CFG.SEED)

import tensorflow as tf

# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu="local") # "local" for 1VM TPU
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
    print("REPLICAS: ", strategy.num_replicas_in_sync)
except tf.errors.NotFoundError:
    print("Not on TPU")

"""# Preprocessing"""

train_data_raw_path = '/kaggle/input/leash-BELKA/train.parquet'
test_data_raw_path = '/kaggle/input/leash-BELKA/test.parquet'

train_data_clean_path = 'train_enc.parquet'
test_data_clean_path = 'test_enc.parquet'

output_path = 'submission.csv'

if CFG.PREPROCESS:
    enc = {'l': 1, 'y': 2, '@': 3, '3': 4, 'H': 5, 'S': 6, 'F': 7, 'C': 8, 'r': 9, 's': 10, '/': 11, 'c': 12, 'o': 13,
           '+': 14, 'I': 15, '5': 16, '(': 17, '2': 18, ')': 19, '9': 20, 'i': 21, '#': 22, '6': 23, '8': 24, '4': 25, '=': 26,
           '1': 27, 'O': 28, '[': 29, 'D': 30, 'B': 31, ']': 32, 'N': 33, '7': 34, 'n': 35, '-': 36}
    train_raw = pd.read_parquet(train_data_raw_path)
    smiles = train_raw[train_raw['protein_name']=='BRD4']['molecule_smiles'].values
    assert (smiles!=train_raw[train_raw['protein_name']=='HSA']['molecule_smiles'].values).sum() == 0
    assert (smiles!=train_raw[train_raw['protein_name']=='sEH']['molecule_smiles'].values).sum() == 0
    def encode_smile(smile):
        tmp = [enc[i] for i in smile]
        tmp = tmp + [0]*(142-len(tmp))
        return np.array(tmp).astype(np.uint8)

    smiles_enc = joblib.Parallel(n_jobs=96)(joblib.delayed(encode_smile)(smile) for smile in tqdm(smiles))
    smiles_enc = np.stack(smiles_enc)
    train = pd.DataFrame(smiles_enc, columns = [f'enc{i}' for i in range(142)])
    train['bind1'] = train_raw[train_raw['protein_name']=='BRD4']['binds'].values
    train['bind2'] = train_raw[train_raw['protein_name']=='HSA']['binds'].values
    train['bind3'] = train_raw[train_raw['protein_name']=='sEH']['binds'].values
    train.to_parquet(train_data_clean_path)

    test_raw = pd.read_parquet(test_data_raw_path)
    smiles = test_raw['molecule_smiles'].values

    smiles_enc = joblib.Parallel(n_jobs=96)(joblib.delayed(encode_smile)(smile) for smile in tqdm(smiles))
    smiles_enc = np.stack(smiles_enc)
    test = pd.DataFrame(smiles_enc, columns = [f'enc{i}' for i in range(142)])
    test.to_parquet(test_data_clean_path)

else:
    train = pd.read_parquet(train_data_clean_path)
    test = pd.read_parquet(test_data_clean_path)

"""# Modeling"""

def my_model():
    with strategy.scope():
        INP_LEN = 142
        NUM_FILTERS = 32
        hidden_dim = 128

        inputs = tf.keras.layers.Input(shape=(INP_LEN,), dtype='int32')
        x = tf.keras.layers.Embedding(input_dim=36, output_dim=hidden_dim, input_length=INP_LEN, mask_zero = True)(inputs)
        x = tf.keras.layers.Conv1D(filters=NUM_FILTERS, kernel_size=3,  activation='relu', padding='valid',  strides=1)(x)
        x = tf.keras.layers.Conv1D(filters=NUM_FILTERS*2, kernel_size=3,  activation='relu', padding='valid',  strides=1)(x)
        x = tf.keras.layers.Conv1D(filters=NUM_FILTERS*3, kernel_size=3,  activation='relu', padding='valid',  strides=1)(x)
        x = tf.keras.layers.GlobalMaxPooling1D()(x)

        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        outputs = tf.keras.layers.Dense(3, activation='sigmoid')(x)

        model = tf.keras.models.Model(inputs = inputs, outputs = outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=CFG.LR, weight_decay = CFG.WD)
        loss = 'binary_crossentropy'
        weighted_metrics = [tf.keras.metrics.AUC(curve='PR', name = 'avg_precision')]
        model.compile(
        loss=loss,
        optimizer=optimizer,
        weighted_metrics=weighted_metrics,
        )
        return model

"""# Train & Inference"""

FEATURES = [f'enc{i}' for i in range(142)]
TARGETS = ['bind1', 'bind2', 'bind3']
skf = StratifiedKFold(n_splits = CFG.NBR_FOLDS, shuffle = True, random_state = 42)

all_preds = []
for fold,(train_idx, valid_idx) in enumerate(skf.split(train, train[TARGETS].sum(1))):

    if fold not in CFG.SELECTED_FOLDS:
        continue;

    X_train = train.loc[train_idx, FEATURES]
    y_train = train.loc[train_idx, TARGETS]
    X_val = train.loc[valid_idx, FEATURES]
    y_val = train.loc[valid_idx, TARGETS]

    es = tf.keras.callbacks.EarlyStopping(patience=5, monitor="val_loss", mode='min', verbose=1)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath=f"model-{fold}.h5",
                                                        save_best_only=True, save_weights_only=True,
                                                    mode='min')
    reduce_lr_loss = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, verbose=1)
    model = my_model()
    history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=CFG.EPOCHS,
            callbacks=[checkpoint, reduce_lr_loss, es],
            batch_size=CFG.BATCH_SIZE,
            verbose=1,
        )
    model.load_weights(f"model-{fold}.h5")
    oof = model.predict(X_val, batch_size = 2*CFG.BATCH_SIZE)
    print('fold :', fold, 'CV score =', APS(y_val, oof, average = 'micro'))

    preds = model.predict(test, batch_size = 2*CFG.BATCH_SIZE)
    all_preds.append(preds)

preds = np.mean(all_preds, 0)

"""# Submission"""

tst = pd.read_parquet(test_data_raw_path)
tst['binds'] = 0
tst.loc[tst['protein_name']=='BRD4', 'binds'] = preds[(tst['protein_name']=='BRD4').values, 0]
tst.loc[tst['protein_name']=='HSA', 'binds'] = preds[(tst['protein_name']=='HSA').values, 1]
tst.loc[tst['protein_name']=='sEH', 'binds'] = preds[(tst['protein_name']=='sEH').values, 2]
tst[['id', 'binds']].to_csv(output_path, index = False)
import os
import random
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score as APS

class CFG:

    EPOCHS = 20
    BATCH_SIZE = 4096
    LR = 1e-3
    WD = 0.05

    NBR_FOLDS = 15
    SELECTED_FOLDS = [0]

    SEED = 2024

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

set_seeds(seed=CFG.SEED)

# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu="local") # "local" for 1VM TPU
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU")
    print("REPLICAS: ", strategy.num_replicas_in_sync)
except tf.errors.NotFoundError:
    print("Not on TPU")

path_file = open('SETTINGS.json')
path_data = json.load(path_file)

train_data_clean_path = path_data["train_data_clean_path"] 
test_data_clean_path = path_data["test_data_clean_path"] 
model_directory = path_data["model_dir"] 

train = pd.read_parquet(train_data_clean_path)
test = pd.read_parquet(test_data_clean_path)

final_model_path = path_data["final_model_path"]

output_path = 'submission.csv'

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
    checkpoint = tf.keras.callbacks.ModelCheckpoint(monitor='val_loss', filepath=f"{model_directory}/model-{fold}.h5",
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
    model.load_weights(f"{model_directory}/model-{fold}.h5")
    oof = model.predict(X_val, batch_size = 2*CFG.BATCH_SIZE)
    print('fold :', fold, 'CV score =', APS(y_val, oof, average = 'micro'))

model.save(final_model_path)
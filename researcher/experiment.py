import datetime
import os
import time
import gc
import json

import tensorflow as tf
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from researcher.logging import *
from researcher.assist import *
from researcher.torch_train import *
from researcher.records import *

def write_history(histories, write_fn):
    n_histories = len(histories)
    for k in histories[0]:
        for i in range(len(histories[0][k])):
            s = 0
            value_count = 0
            for h in histories:
                if i < len(h[k]):
                    s += h[k][i]
                    value_count += 1
            
            write_fn(k, s/value_count, i)

def log_histories(history, writer, val_writer):
    train_histories = []
    val_histories = []

    for h in history:
        train_history = {}
        val_history = {}
        for k in h:
            if k[:4] == "val_":
                val_history[k[4:]] = h[k]
            else:
                train_history[k] = h[k]
        train_histories.append(train_history)
        val_histories.append(val_history)

    write_history(train_histories, writer)
    write_history(val_histories, val_writer)   

def get_mean_loss(model, df, loss_fn):
    losses = []
    for X, y in df:
        preds = model.predict(X)
        losses.append(loss_fn(y, preds))
    
    return np.mean(losses)


def kfold_tf_experiment(name, model_hash, make_model, train_gens, val_gens, epochs,steps_per_epoch, batch_size, 
        extra_val_gens, workers, save, model_save_path, log_path):

    assert val_gens is None or len(train_gens) == len(val_gens)
  
    tf.keras.backend.clear_session()
    val_gens = val_gens or [None for i in train_gens]

    start_time = datetime.datetime.now()
    log_dir = "{}{}_{}".format(log_path, name, start_time.strftime("%Y%m%d-%H%M%S"))
    
    tb_writer = tf.summary.create_file_writer(log_dir + "_train")
    tb_writer.set_as_default()
    val_tb_writer = tf.summary.create_file_writer(log_dir + "_val")
    val_tb_writer.set_as_default()

    histories = []

    for fold, gen in enumerate(train_gens):
        print("\n\n========== Starting Fold {} =========".format(fold))
        
        # Logging callbacks
        batch_history = {}
        callbacks = [
            BatchWriter(make_dict_writer(batch_history), batch_metrics=["loss", "accuracy"]),
            LearningRateWriter(make_dict_writer(batch_history)),
        ]

        model = make_model()

        model.fit(
            gen, 
            batch_size=batch_size,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            verbose=1,
            callbacks=callbacks,
            validation_data=val_gens[fold]
        )

        extra_val_history = {}
        for name, val_generator, loss_fn in extra_val_gens:
            extra_val_history[name] = get_mean_loss(model, val_generator, loss_fn) 

        
        histories.append({**model.history.history, **batch_history, **extra_val_history})
        
        if save:
            model.save_weights(model_weights_file(model_save_path, model_hash, fold=fold))
        
        del model
        gc.collect()
        tf.keras.backend.clear_session()

    log_histories(histories, make_write_to_tb(tb_writer), make_write_to_tb(val_tb_writer))

    print("\n\n========== Experiment Duration {} =========".format(datetime.datetime.now() - start_time))


    return histories


# ---------------------------------------------------------------------------------------------------------
#
#                                       PARAMETER EXPERIMENTING 
#
# ---------------------------------------------------------------------------------------------------------

def get_experiment_methods(collection, params):
    assert not (params["model_fn"] in collection.TF_MODELS and params["model_fn"] in collection.TORCH_MODELS)

    if params["model_fn"] in collection.TF_MODELS:
        return collection.get_model_maker(collection, params), kfold_tf_experiment, collection.TF_CALLBACKS
    
    raise NotImplementedError
    # return torch_model_maker(params), None , collection.TORCH_CALLBACKS
    return None, None , None

def process_params(collection, params, n_folds):
    assert not(n_folds < 2 and params["val_prop"] is None) 
    assert not(n_folds > 1 and params["val_prop"] is not None) 

    param_hash = get_hash(params)

    model_fn, experiment_fn, callback_map = get_experiment_methods(collection, params)   

    return param_hash, model_fn, experiment_fn, callback_map

def run_param_kfold_experiment(collection, params, model_save_path, log_path, experiment_path, n_folds, workers=1, save=False, trial=False):
    param_hash, model_fn, experiment_fn, callback_map = process_params(collection, params, n_folds)    
    train_gens, val_gens, extra_val_gens = collection.get_data(params, n_folds)
    
    print("running experiment: {} {}".format(params["description"], param_hash))

    results = experiment_fn(
        "{}___{}".format(params["description"],param_hash[:8]),
        param_hash,
        model_fn,
        train_gens,
        val_gens,
        params["epochs"],
        params["steps_per_epoch"],
        params["batch_size"],
        extra_val_gens,
        workers,
        save,
        model_save_path,
        log_path,
    )

    if not trial:
        save_experiment(experiment_path, "{}_{}".format(params["description"],param_hash), parameters=params, results=results)

    

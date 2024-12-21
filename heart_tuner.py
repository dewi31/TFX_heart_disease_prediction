import tensorflow as tf
import tensorflow_transform as tft 
from tensorflow.keras import layers
import os  

from kerastuner.engine import base_tuner 
from typing import NamedTuple, Dict, Text, Any 
from tfx.components.trainer.fn_args_utils import FnArgs
from tfx.components import Tuner
import kerastuner as kt

LABEL_KEY = "target"
FEATURE_KEYS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalachh", "exang", "oldpeak", "slope", "ca", "thal"]

def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                            ('fit_kwargs', Dict[Text,Any])])

def input_fn(file_pattern, 
             tf_transform_output,
             num_epochs,
             batch_size=100)->tf.data.Dataset:
    
    transform_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key = transformed_name(LABEL_KEY)
    )
    return dataset

def model_builder(hp):
    num_features = len(FEATURE_KEYS)
    transformed_FEATURE_KEYS = [transformed_name(key) for key in FEATURE_KEYS]
    inputs = [tf.keras.Input(shape=(1,), name=f) for f in transformed_FEATURE_KEYS]
    x = layers.concatenate(inputs)
    
    hp_units = hp.Int('units', min_value=8, max_value=48, step=8)
    x = layers.Dense(hp_units, activation="relu")(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    hp_learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )
    
    model.summary()
    return model

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    tuner = kt.RandomSearch(
        model_builder,
        objective='val_binary_accuracy',
        max_trials=5,
        directory=fn_args.working_dir,
        project_name='heart_tuning'
    )
    
    train_set = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)
    
    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": val_set,
            "epochs": 25
        }
    )

import tensorflow as tf
import tensorflow_transform as tft
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs
from kerastuner.engine.hyperparameters import HyperParameters
import os

LABEL_KEY = "target"
FEATURE_KEYS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalachh", "exang", "oldpeak", "slope", "ca", "thal"]

def transformed_name(key):
    return key + "_xf"

def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')

def input_fn(file_pattern,
             tf_transform_output,
             num_epochs=None,
             batch_size=128) -> tf.data.Dataset:
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()
    return tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY)
    )

def model_builder(hp):
    num_features = len(FEATURE_KEYS)
    transformed_FEATURE_KEYS = [transformed_name(key) for key in FEATURE_KEYS]
    inputs = [tf.keras.Input(shape=(1,), name=f) for f in transformed_FEATURE_KEYS]
    x = layers.concatenate(inputs)

    # Mengambil nilai parameter terbaik dari tuner
    hp_units = hp['values']['units'] 
    hp_learning_rate = hp['values']['learning_rate']

    x = layers.Dense(hp_units, activation="relu")(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)


    model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        metrics=[tf.keras.metrics.BinaryAccuracy()]
    )

    model.summary()
    return model

def _get_serve_tf_examples_fn(model, tf_transform_output):
    
    model.tft_layer = tf_transform_output.transform_features_layer()
    
    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples):
        
        feature_spec = tf_transform_output.raw_feature_spec()
        
        feature_spec.pop(LABEL_KEY)
        
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
        
        transformed_features = model.tft_layer(parsed_features)
        
        # get predictions using the transformed features
        return model(transformed_features)
        
    return serve_tf_examples_fn

def run_fn(fn_args: FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    hp = fn_args.hyperparameters
    model = model_builder(hp)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, num_epochs=10)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, num_epochs=10)

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir = log_dir, update_freq='batch'
    )
    
    es = tf.keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', mode='max', verbose=1, patience=5)
    mc = tf.keras.callbacks.ModelCheckpoint(fn_args.serving_model_dir, monitor='val_binary_accuracy', mode='max', verbose=1, save_best_only=True)
    

    model.fit(
        train_dataset,
        epochs=25,
        validation_data=eval_dataset,
        callbacks=[tensorboard_callback, es, mc]
    )

    signatures = {
        'serving_default':
        _get_serve_tf_examples_fn(model, tf_transform_output).get_concrete_function(
                                    tf.TensorSpec(
                                    shape=[None],
                                    dtype=tf.string,
                                    name='examples'))
    }

    # Save the model
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)

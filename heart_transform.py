import tensorflow_transform as tft

LABEL_KEY = "target"
FEATURE_KEYS = "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalachh", "exang", "oldpeak", "slope", "ca", "thal"
def transformed_name(key):
     return key + "_xf"
def preprocessing_fn(inputs):
    
    outputs = {
        transformed_name(key): tft.scale_to_0_1(inputs[key])
        for key in FEATURE_KEYS
    }
    
    outputs[transformed_name(LABEL_KEY)] = inputs[LABEL_KEY]
    
    return outputs

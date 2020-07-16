"""A custom class for predictions in AI Platform."""

import base64
import numpy as np
import os
import pickle
import tempfile
import tensorflow as tf


class MyPredictor(object):
  """My custom prediction class.

  This implements the interface expected by AI Platform.
  """

  def __init__(self, model, preprocessor):
    """Return a MyPredictor instance."""
    self._model = model
    self._preprocessor = preprocessor

  def predict(self, instances, **kwargs):
    """Predict the output for the corresponding instances."""
    tensors = []
    for instance in instances:
      # Read image and create file
      b64s = instance['image_input']['b64']
      f = tempfile.TemporaryFile()
      binstr = base64.b64decode(b64s)
      f.write(binstr)
      f.close()
      img = tf.keras.preprocessing.image.load_img(f.name)
      tensor = tf.keras.preprocessing.image.img_to_array(img)
      # Preprocess
      img_size, factor = self._preprocessor
      tensor = tensor / factor
      tensor = tensor.reshape((1,) + tensor.shape)
      tensors.append(tensor)

    # Predict
    predict_tensor = np.vstack(tensors)
    return self._model.predict(predict_tensor)

  @classmethod
  def from_path(cls, model_dir):
    """Load model."""
    model_path = os.path.join(model_dir, 'keras_model')
    model = tf.keras.models.load_model(model_path)

    preprocessor_path = os.path.join(model_dir, 'preprocessor.pkl')
    with open(preprocessor_path) as f:
      preprocessor = pickle.load(f)

    return cls(model, preprocessor)

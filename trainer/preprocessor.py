"""A class to preprocess images."""

from tensorflow.keras.preprocessing.image import ImageDataGenerator


class MyImagePreprocessor:
  """Preprocess images for training or inference."""

  def __init__(self, img_size, batch_size):
    """Create a MyImagePreprocessor.

    Args:
      img_size: The size of the resulting images.
      batch_size: The size of the batch (for training purposes)

    """
    self._img_size = img_size
    self._batch_size = batch_size

    self._img_datagen = ImageDataGenerator(rescale=1 / 255.0)

  def generator(self, dirname):
    """Create and return a generator for the preprocessed images.

    Args:
      dirname: The name of the directory with the images.
    """
    self._img_generator = self._img_datagen.flow_from_directory(
        dirname,
        target_size=(self._img_size, self._img_size),
        batch_size=self._batch_size,
        class_mode='binary')

    return self._img_generator

  def img_data_generator(self):
    """Get the ImageDataGenerator."""
    return self._img_datagen

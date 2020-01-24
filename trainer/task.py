"""A simple main file to showcase the template."""

import argparse
import logging.config
import os
import tensorflow as tf

from .model import build_model
from google.cloud import storage
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.preprocessing.image import ImageDataGenerator

"""
This module is an example for a single Python application with some
top level functions. The tests directory includes some unitary tests
for these functions.

This is one of two main files samples included in this
template. Please feel free to remove this, or the other
(sklearn_main.py), or adapt as you need.
"""

LOGGER = logging.getLogger()


def _list_files_by_prefix(bucket_name, prefix):
  storage_client = storage.Client()
  # Note: Client.list_blobs requires at least package version 1.17.0.
  blobs = storage_client.list_blobs(
      bucket_name, prefix=prefix, delimiter=None)

  names = [blob.name for blob in blobs]
  return names


def _download_file(bucket_name, remote_name, dest_name):
  storage_client = storage.Client()

  bucket = storage_client.bucket(bucket_name)
  blob = bucket.blob(remote_name)
  blob.download_to_filename(dest_name)

  print(
      "Blob {} downloaded to {}.".format(
          remote_name, dest_name
      )
  )


def download_prepare_data(bucket_name, prefix):
  """Download and prepare the data for training.

  Args:
    bucket_name: Name of the bucket where the data is stored
    prefix: Prefix to the path of all the files
  """
  names = _list_files_by_prefix(bucket_name, prefix)

  for name in names:
    fn = name.split('/')[-1]
    if fn.endswith('jpg'):
      label = fn.split('.')[0]
      dest_dir = 'data/%s/' % label
      dest_name = dest_dir + fn

      # Check that dest dir exists
      if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

      _download_file(bucket_name, name, dest_name)
    else:
      print("Non jpg found: %s" % fn)


def train_and_evaluate(
    bucket_name,
    prefix,
    download,
    img_size,
    batch_size,
    n_imgs,
    epochs
):
  """Train and evaluate the model."""
  if download:
    download_prepare_data(bucket_name, prefix)
  else:
    print("Not downloading data")

  # FIXME: train a model
  img_datagen = ImageDataGenerator(rescale=1 / 255.0)

  img_generator = img_datagen.flow_from_directory(
      'data',
      target_size=(img_size, img_size),
      batch_size=batch_size,
      class_mode='binary')

  steps = int(n_imgs / batch_size)

  model = build_model(img_size)
  model.compile(
      optimizer=optimizers.Adam(),
      loss=losses.binary_crossentropy,
      metrics=['accuracy']
  )
  model.fit_generator(img_generator, epochs=epochs, steps_per_epoch=steps)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--bucket-name", required=True)
  parser.add_argument("--prefix", required=True)
  parser.add_argument("--epochs", required=True, type=int)
  parser.add_argument("--download", action='store_true')

  args = parser.parse_args()

  bucket_name = args.bucket_name
  prefix = args.prefix
  download = args.download
  epochs = args.epochs

  train_and_evaluate(bucket_name,
                     prefix,
                     download,
                     128,
                     1,
                     5,
                     epochs)

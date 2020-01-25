"""A simple main file to showcase the template."""

import trainer
import argparse
import logging.config
import os
import random
import time
import tensorflow as tf

from .model import build_model
from .utils import upload_local_directory_to_gcs
from google.cloud import storage
from tensorflow.core.framework.summary_pb2 import Summary
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


def download_prepare_data(bucket_name, prefix, train_split):
  """Download and prepare the data for training.

  Args:
    bucket_name: Name of the bucket where the data is stored
    prefix: Prefix to the path of all the files
    train_split: Number between 0 and 1 for the train/test split
  """
  names = _list_files_by_prefix(bucket_name, prefix)

  for name in names:
    fn = name.split('/')[-1]
    if fn.endswith('jpg'):
      label = fn.split('.')[0]

      randnum = random.random()
      if randnum < train_split:
        # Training
        dest_dir = 'data/%s/%s/' % ('train', label)
      else:
        # Test
        dest_dir = 'data/%s/%s/' % ('test', label)

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
    epochs,
    job_dir,
    train_split
):
  """Train and evaluate the model."""
  if download:
    download_prepare_data(bucket_name, prefix, train_split)
  else:
    print("Not downloading data")

  img_datagen = ImageDataGenerator(rescale=1 / 255.0)

  img_generator = img_datagen.flow_from_directory(
      'data/train',
      target_size=(img_size, img_size),
      batch_size=batch_size,
      class_mode='binary')

  test_generator = img_datagen.flow_from_directory(
      'data/test',
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

  model_loss, model_acc = model.evaluate_generator(test_generator)

  print('MODEL LOSS: %.4f' % model_loss)
  print('MODEL ACC: %.4f' % model_acc)

  # Report metrics to Cloud ML Engine for hypertuning
  metric_tag = 'accuracy_dogs_cats'
  summary = Summary(value=[Summary.Value(tag=metric_tag,
                                         simple_value=model_acc)])
  eval_path = os.path.join(job_dir, metric_tag)
  LOGGER.info("Writing metrics to %s" % eval_path)
  summary_writer = tf.summary.FileWriter(eval_path)

  summary_writer.add_summary(summary)
  summary_writer.flush()

  localdir = 'my_model'
  tf.keras.experimental.export_saved_model(model, localdir)
  # TF 2.0 --> model.save(...)

  # gs://bucket_name/prefix1/prefix2/....
  dest_bucket_name = job_dir.split('/')[2]
  timestamp = int(round(time.time() * 1000))
  path_in_bucket = 'saved_models/' + trainer.__version__ + '/' + timestamp + '/'

  # Upload to GCS
  client = storage.Client()
  bucket = client.bucket(dest_bucket_name)
  LOGGER.info("Uploading model to gs://%s/%s" % (dest_bucket_name,
                                                 path_in_bucket))
  upload_local_directory_to_gcs(localdir, bucket, path_in_bucket)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--bucket-name", required=True)
  parser.add_argument("--prefix", required=True)
  parser.add_argument("--epochs", required=True, type=int)
  parser.add_argument("--img-size", default=128, type=int)
  parser.add_argument("--download", action='store_true')
  parser.add_argument("--job-dir", required=False)
  parser.add_argument("--train-split", default=0.9, type=float)

  args = parser.parse_args()

  # Tuneable hyperparameters
  epochs = args.epochs
  img_size = args.img_size

  bucket_name = args.bucket_name
  prefix = args.prefix
  download = args.download
  job_dir = args.job_dir
  train_split = args.train_split

  train_and_evaluate(bucket_name,
                     prefix,
                     download,
                     img_size,
                     10,
                     40,
                     epochs,
                     job_dir,
                     train_split)

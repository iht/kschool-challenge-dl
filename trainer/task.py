"""A simple main file to showcase the template."""

import argparse
import copy
import logging.config
import os
import pickle
import random
import time

from .model import build_model
from .preprocessor import MyImagePreprocessor
from .utils import upload_local_directory_to_gcs, write_summary_to_aiplatform
from google.cloud import storage

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import optimizers
from tensorflow.keras import losses


"""
This module is an example for a single Python application with some
top level functions. The tests directory includes some unitary tests
for these functions.
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

  preprocessor = MyImagePreprocessor(img_size, batch_size)

  img_generator = preprocessor.generator('data/train')
  test_generator = preprocessor.generator('data/test')

  steps = int(n_imgs / batch_size)

  model = build_model(img_size)

  # Report metrics to Tensorboard, to follow the training process
  tensorboard_logdir = os.path.join(job_dir, 'tblogs')
  tb_callback = TensorBoard(tensorboard_logdir)

  model.compile(
      optimizer=optimizers.Adam(),
      loss=losses.binary_crossentropy,
      metrics=['accuracy']
  )

  model.fit_generator(img_generator,
                      epochs=epochs,
                      steps_per_epoch=steps,
                      callbacks=[tb_callback])

  model_loss, model_acc = model.evaluate_generator(test_generator)

  print('MODEL LOSS: %.4f' % model_loss)
  print('MODEL ACC: %.4f' % model_acc)

  # Report metrics to Cloud AI Platform for hypertuning
  metric_tag = 'accuracy_dogs_cats'
  write_summary_to_aiplatform(metric_tag, job_dir, model_acc)

  # Save model
  localdir = 'export'
  localmodel = os.path.join(localdir, 'keras_model')
  model.save(localmodel, save_format='tf')

  # Save preprocessor
  localpreprocfn = os.path.join(localdir, 'preprocess.pkl')
  with open(localpreprocfn, 'wb') as f:
    pickle.dump(copy.deepcopy(preprocessor.preprocess_params()), f)

  # gs://bucket_name/prefix1/prefix2/....
  dest_bucket_name = job_dir.split('/')[2]
  timestamp = str(int(round(time.time() * 1000)))
  path_in_bucket = 'saved_models/%d_%d/' % (epochs, img_size) + timestamp

  # Upload to GCS
  client = storage.Client()
  bucket = client.bucket(dest_bucket_name)
  LOGGER.info("Uploading model to gs://%s/%s" % (
      dest_bucket_name,
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

  # Other hyperparameters
  batch_size = 50
  num_images_generator = 400

  bucket_name = args.bucket_name
  prefix = args.prefix
  download = args.download
  job_dir = args.job_dir
  train_split = args.train_split

  train_and_evaluate(bucket_name,
                     prefix,
                     download,
                     img_size,
                     batch_size,
                     num_images_generator,
                     epochs,
                     job_dir,
                     train_split)

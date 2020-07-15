"""Utils for the models in ML Platform."""

import glob
import logging
import os
import tensorflow as tf

from tensorflow.core.framework.summary_pb2 import Summary

LOGGER = logging.getLogger()


def upload_local_directory_to_gcs(local_path, bucket, gcs_path):
  """Upload a local directory, recursively, to GCS.

  Args:
   local_path: String with the local directory to be uploaded.
   bucket: Destination bucket object created with a GCS client
   gcs_path: Path in the bucket for the destination directory

  """
  assert os.path.isdir(local_path)
  for local_file in glob.glob(local_path + '/**'):
    if not os.path.isfile(local_file):
      upload_local_directory_to_gcs(
          local_file,
          bucket,
          gcs_path + "/" + os.path.basename(local_file))
    else:
      remote_path = os.path.join(gcs_path, local_file[1 + len(local_path):])
      blob = bucket.blob(remote_path)
      blob.upload_from_filename(local_file)


def write_summary_to_aiplatform(metric_tag, destdir, metric):
  """Write a metric as a TF Summary.

  AI Platform will use this metric for hyperparameters tuning.

  Args:
    metric_tag: A string with the tag used for hypertuning.
    destdir: Destination path for the metric to be written.
    metric: Value of the metric to be written.

  """
  summary = Summary(value=[Summary.Value(tag=metric_tag,
                                         simple_value=metric)])
  eval_path = os.path.join(destdir, metric_tag)
  LOGGER.info("Writing metrics to %s" % eval_path)
  summary_writer = tf.summary.FileWriter(eval_path)

  summary_writer.add_summary(summary)
  summary_writer.flush()

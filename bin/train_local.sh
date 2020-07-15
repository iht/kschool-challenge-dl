gcloud ai-platform local train \
  --package-path ./trainer \
  --module-name trainer.task \
  --job-dir=/tmp/models/ \
  -- \
  --epochs 5 \
  --img-size 128 \
  --bucket-name kschool-challenge-vcm \
  --prefix data/dogs_cats/

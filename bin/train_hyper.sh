gcloud ai-platform jobs submit training generator_train_ht_`date +"%s"` \
  --python-version=3.7 \
  --runtime-version=1.15 \
  --scale-tier basic \
  --package-path ./trainer \
  --module-name trainer.task \
  --job-dir=gs://ihr-kschool-training/tmp/ \
  --region us-central1 \
  --config hyper.yaml
  -- \
  --download \
  --bucket-name kschool-challenge-vcm \
  --prefix data/dogs_cats/


  
  

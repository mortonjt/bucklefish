echo 'multinomial'
python multinomial.py \
   --train_biom ../data/88soils/train_88soils_processed.biom \
   --train_metadata ../data/88soils/train_88soils_metadata.txt \
   --test_biom ../data/88soils/test_88soils_processed.biom \
   --test_metadata ../data/88soils/test_88soils_metadata.txt \
   --formula ph \
   --learning_rate 1e-1 \
   --batch_size 5 \
   --beta_scale 3 \
   --gamma_scale 3 \
   --min_sample_count 100 \
   --min_feature_count 10 \
   --save_path ../logs/multinomial_log \
   --checkpoint_interval 60 \
   --epochs_to_train 100

echo 'poisson'
python poisson_sparse.py \
  --train_biom ../data/88soils/train_88soils_processed.biom \
  --train_metadata ../data/88soils/train_88soils_metadata.txt \
  --test_biom ../data/88soils/test_88soils_processed.biom \
  --test_metadata ../data/88soils/test_88soils_metadata.txt \
  --formula ph \
  --learning_rate 1e-1 \
  --num_neg_samples 1024 \
  --batch_size 1024 \
  --beta_scale 3 \
  --gamma_scale 3 \
  --min_sample_count 100 \
  --min_feature_count 10 \
  --save_path ../logs/poisson_sparse_log \
  --checkpoint_interval 60 \
  --epochs_to_train 100

echo 'poisson_parallel'
python poisson_sparse_parallel.py \
  --train_biom ../data/88soils/train_88soils_processed.biom \
  --train_metadata ../data/88soils/train_88soils_metadata.txt \
  --test_biom ../data/88soils/test_88soils_processed.biom \
  --test_metadata ../data/88soils/test_88soils_metadata.txt \
  --formula ph \
  --learning_rate 1e-1 \
  --num_neg_samples 1025 \
  --batch_size 1024 \
  --beta_scale 3 \
  --gamma_scale 3 \
  --min_sample_count 100 \
  --min_feature_count 10 \
  --save_path ../logs/poisson_sparse_parallel_log \
  --checkpoint_interval 60 \
  --epochs_to_train 100

# echo 'poisson ilr'
# python poisson_sparse_ilr.py \
#    --train_biom ../data/88soils/train_88soils_processed.biom \
#    --train_metadata ../data/88soils/train_88soils_metadata.txt \
#    --test_biom ../data/88soils/test_88soils_processed.biom \
#    --test_metadata ../data/88soils/test_88soils_metadata.txt \
#    --tree ../data/88soils/88soils_tree.nwk \
#    --formula ph \
#    --learning_rate 1e-1 \
#    --num_neg_samples 1024 \
#    --batch_size 1024 \
#    --beta_scale 1 \
#    --gamma_scale 1 \
#    --checkpoint_interval 60 \
#    --min_sample_count 100 \
#    --min_feature_count 10 \
#    --save_path ../logs/poisson_sparse_ilr_log \
#    --epochs_to_train 100 \
#    --formula ph \



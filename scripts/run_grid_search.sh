#!/bin/bash

# In this example, we show how to train SimCSE on unsupervised Wikipedia data.
# If you want to train it with multiple GPU cards, see "run_sup_example.sh"
# about how to use PyTorch's distributed data parallel.

export CUDA_VISIBLE_DEVICES=1

for k in 0.1 1 10; do
for t in 0.1 1 10; do
export save_path=new_result/unsup-bert-base-uncased_layer6_k${k}_t${t}
python train.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/prev_best_bert_train_file.csv \
    --output_dir ${save_path} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --logging_steps 25 \
    --pooler_type cls_before_pooler \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --do_mlm \
    --mlm_weight 1.0 \
    --k_weight $k \
    --t_weight $t \
    "$@"
done;
done
export CUDA_VISIBLE_DEVICES=0
python evaluation.py \
    --model_name_or_path new_result/unsup-bert-base-uncased \
    --pooler cls_before_pooler \
    --task_set sts \
    --mode test

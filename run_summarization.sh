#!/bin/bash

CSV_PATH='/home/ec2-user/moymarce/datasets/CNN-DailyMail/info-extraction/target-info/csv/'
DATA='source_oracle_kg'
#MODEL='/home/ec2-user/moymarce/code/transformers/checkpoints/1-source_summary_kg/'
MODEL='facebook/bart-base'
OUT_DIR='./checkpoints/5-${DATA}/'


python examples/pytorch/summarization/run_summarization_BART_Extended.py \
--model_name_or_path $MODEL \
--do_train \
--do_eval \
--do_predict \
--num_train_epochs=5 \
--save_steps=5000 \
--logging_dir ./logs \
--logging_strategy steps \
--logging_steps 500 \
--save_strategy steps \
--save_steps 5000 \
--report_to tensorboard \
--train_file $CSV_PATH${DATA}_train.csv \
--validation_file $CSV_PATH${DATA}_valid.csv \
--test_file $CSV_PATH${DATA}_test.csv \
--output_dir $OUT_DIR \
--per_device_train_batch_size=2 \
--per_device_eval_batch_size=2 \
--overwrite_output_dir \
--predict_with_generate \
--load_best_model_at_end \
--overwrite_output_dir \

# Remove checkpoints
rm -rf $OUT_DIR/checkpoint*

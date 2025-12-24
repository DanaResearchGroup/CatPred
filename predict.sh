#!/bin/bash
set -euo pipefail
TEST_FILE_PREFIX=./demo/batch_ki_input
RECORDS_FILE=${TEST_FILE_PREFIX}.json.gz
CHECKPOINT_DIR=../catpred_pipeline/data/pretrained/production/ki/

python ./scripts/create_pdbrecords.py --data_file ${TEST_FILE_PREFIX}.csv --out_file ${RECORDS_FILE}
python predict.py --test_path ${TEST_FILE_PREFIX}.csv --preds_path ${TEST_FILE_PREFIX}_output.csv --checkpoint_dir $CHECKPOINT_DIR --uncertainty_method mve --smiles_columns SMILES --individual_ensemble_predictions --protein_records_path ${RECORDS_FILE}

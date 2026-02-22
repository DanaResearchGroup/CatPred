#!/bin/bash
set -euo pipefail
TEST_FILE_PREFIX=output/bees_62ab6c1b_kcat/bees_62ab6c1b_kcat_input
OUTPUT_FILE=output/bees_62ab6c1b_kcat/bees_62ab6c1b_kcat_kcat_output.csv
RECORDS_FILE=output/bees_62ab6c1b_kcat/bees_62ab6c1b_kcat_input.json.gz
CHECKPOINT_DIR=/home/omerkfir/Kinetic_repos/catpred_pipeline/data/pretrained/production/kcat

python ./scripts/create_pdbrecords.py --data_file ${TEST_FILE_PREFIX}.csv --out_file ${RECORDS_FILE}
python predict.py --test_path ${TEST_FILE_PREFIX}.csv --preds_path $OUTPUT_FILE --checkpoint_dir $CHECKPOINT_DIR --uncertainty_method mve --smiles_columns SMILES --individual_ensemble_predictions --protein_records_path ${RECORDS_FILE}

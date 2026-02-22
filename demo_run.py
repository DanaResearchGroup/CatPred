"""
Enzyme Kinetics Parameter Prediction Script

This script predicts enzyme kinetics parameters (kcat, Km, or Ki) using a pre-trained model.
It processes input data, generates predictions, and saves the results.

Usage:
    python demo_run.py --parameter <kcat|km|ki> --input_file <path_to_input_csv> --checkpoint_dir <path_to_pretrained_checkpoint_dir> [--use_gpu]

Dependencies:
    pandas, numpy, rdkit, IPython, argparse
"""

import time
import os
import pandas as pd
import numpy as np
from IPython.display import Image, display
from rdkit import Chem
from IPython.display import display, Latex, Math
import argparse
import json
import stat

def create_csv_sh(parameter, input_file_path, checkpoint_dir):
    # #region agent log
    try:
        os.makedirs('.cursor', exist_ok=True)
        with open('.cursor/debug.log', 'a') as log:
            log.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H5", "location": "demo_run.py:25", "message": "create_csv_sh() function entry", "data": {"parameter": parameter, "input_file_path": input_file_path, "checkpoint_dir": checkpoint_dir, "input_file_exists": os.path.exists(input_file_path)}, "timestamp": int(time.time() * 1000)}) + "\n")
    except Exception:
        pass
    # #endregion
    
    df = pd.read_csv(input_file_path)
    smiles_list = df.SMILES
    seq_list = df.sequence
    smiles_list_new = []
    invalid_rows = []

    for i, smi in enumerate(smiles_list):
        # Check for NaN or empty values
        if pd.isna(smi) or (isinstance(smi, str) and smi.strip() == ''):
            invalid_rows.append((i, "Empty or missing SMILES value"))
            continue
        
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                invalid_rows.append((i, "Failed to parse SMILES (returned None)"))
                continue
            smi = Chem.MolToSmiles(mol)
            if parameter == 'kcat' and '.' in smi:
                smi = '.'.join(sorted(smi.split('.')))
            smiles_list_new.append(smi)
        except Exception as e:
            invalid_rows.append((i, str(e)))

    if invalid_rows:
        print(f'\nFound {len(invalid_rows)} invalid SMILES row(s):')
        for row_idx, error_msg in invalid_rows:
            print(f'  Row {row_idx}: {error_msg}')
            if row_idx < len(smiles_list):
                smi_preview = str(smiles_list.iloc[row_idx])[:100]
                print(f'    SMILES preview: {smi_preview}...' if len(str(smiles_list.iloc[row_idx])) > 100 else f'    SMILES: {smiles_list.iloc[row_idx]}')
        print('\nCorrect your input! Exiting..')
        return None, None

    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    invalid_seq_rows = []
    for i, seq in enumerate(seq_list):
        if not set(seq).issubset(valid_aas):
            invalid_seq_rows.append(i)

    if invalid_seq_rows:
        print(f'\nFound {len(invalid_seq_rows)} invalid sequence row(s):')
        for row_idx in invalid_seq_rows:
            print(f'  Row {row_idx}: Contains invalid amino acid characters')
            if row_idx < len(seq_list):
                seq_preview = str(seq_list.iloc[row_idx])[:100]
                print(f'    Sequence preview: {seq_preview}...' if len(str(seq_list.iloc[row_idx])) > 100 else f'    Sequence: {seq_list.iloc[row_idx]}')
        print('\nCorrect your input! Exiting..')
        return None, None

    # Extract base name from input file (remove extension and "input" if present)
    input_basename = os.path.basename(input_file_path)
    if input_basename.endswith('.csv'):
        input_basename = input_basename[:-4]
    if input_basename.endswith('_input'):
        input_basename = input_basename[:-6]
    
    # Create output directory structure
    output_base_dir = 'output'
    output_dir = os.path.join(output_base_dir, input_basename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create file paths within the output directory
    input_file_new_path = os.path.join(output_dir, f'{input_basename}_input.csv')
    output_file_path = os.path.join(output_dir, f'{input_basename}_{parameter}_output.csv')
    
    df['SMILES'] = smiles_list_new
    df.to_csv(input_file_new_path)

    # #region agent log
    try:
        os.makedirs('.cursor', exist_ok=True)
        script_path = os.path.abspath('predict.sh')
        with open('.cursor/debug.log', 'a') as log:
            log.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1,H5", "location": "demo_run.py:58", "message": "Before creating predict.sh", "data": {"script_path": script_path, "cwd": os.getcwd()}, "timestamp": int(time.time() * 1000)}) + "\n")
    except Exception:
        pass
    # #endregion

    # Get paths for predict.sh (relative to project root)
    test_file_prefix = input_file_new_path[:-4]  # Remove .csv extension
    records_file = f'{test_file_prefix}.json.gz'
    
    with open('predict.sh', 'w') as f:
        script_content = f'''#!/bin/bash
set -euo pipefail
TEST_FILE_PREFIX={test_file_prefix}
OUTPUT_FILE={output_file_path}
RECORDS_FILE={records_file}
CHECKPOINT_DIR={checkpoint_dir}

python ./scripts/create_pdbrecords.py --data_file ${{TEST_FILE_PREFIX}}.csv --out_file ${{RECORDS_FILE}}
python predict.py --test_path ${{TEST_FILE_PREFIX}}.csv --preds_path $OUTPUT_FILE --checkpoint_dir $CHECKPOINT_DIR --uncertainty_method mve --smiles_columns SMILES --individual_ensemble_predictions --protein_records_path ${{RECORDS_FILE}}
'''
        f.write(script_content)

    # #region agent log
    try:
        file_stat = os.stat('predict.sh')
        file_mode = oct(file_stat.st_mode)[-3:]
        with open('.cursor/debug.log', 'a') as log:
            log.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "demo_run.py:77", "message": "After creating predict.sh, before chmod", "data": {"file_exists": os.path.exists('predict.sh'), "file_mode": file_mode, "is_executable": os.access('predict.sh', os.X_OK)}, "timestamp": int(time.time() * 1000)}) + "\n")
    except Exception as e:
        pass
    # #endregion

    os.chmod('predict.sh', stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)

    # #region agent log
    try:
        file_stat_after = os.stat('predict.sh')
        file_mode_after = oct(file_stat_after.st_mode)[-3:]
        with open('.cursor/debug.log', 'a') as log:
            log.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "demo_run.py:84", "message": "After chmod predict.sh", "data": {"file_mode": file_mode_after, "is_executable": os.access('predict.sh', os.X_OK)}, "timestamp": int(time.time() * 1000)}) + "\n")
    except Exception as e:
        pass
    # #endregion

    # #region agent log
    try:
        with open('predict.sh', 'r') as f:
            script_content_check = f.read()
            has_leading_whitespace = script_content_check.startswith(' ') or script_content_check.startswith('\t')
        with open('.cursor/debug.log', 'a') as log:
            log.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H2", "location": "demo_run.py:92", "message": "Checking script content", "data": {"has_leading_whitespace": has_leading_whitespace, "first_50_chars": script_content_check[:50], "has_smiles_column": "--smiles_column" in script_content_check, "has_smiles_columns": "--smiles_columns" in script_content_check}, "timestamp": int(time.time() * 1000)}) + "\n")
    except Exception as e:
        pass
    # #endregion

    return output_file_path, output_dir

def get_predictions(parameter, outfile):
    """
    Process prediction results and add additional metrics.

    Args:
        parameter (str): The kinetics parameter that was predicted.
        outfile (str): Path to the output CSV file from the prediction.

    Returns:
        pandas.DataFrame: Processed predictions with additional metrics.
    """
    df = pd.read_csv(outfile)
    pred_col, pred_logcol, pred_sd_totcol, pred_sd_aleacol, pred_sd_epicol = [], [], [], [], []

    unit = 'mM'
    if parameter == 'kcat':
        target_col = 'log10kcat_max'
        unit = 's^(-1)'
    elif parameter == 'km':
        target_col = 'log10km_mean'
    else:
        target_col = 'log10ki_mean'

    unc_col = f'{target_col}_mve_uncal_var'

    for _, row in df.iterrows():
        model_cols = [col for col in row.index if col.startswith(target_col) and 'model_' in col]

        unc = row[unc_col]
        prediction = row[target_col]
        prediction_linear = np.power(10, prediction)

        model_outs = np.array([row[col] for col in model_cols])
        epi_unc = np.var(model_outs)
        alea_unc = unc - epi_unc
        epi_unc = np.sqrt(epi_unc)
        alea_unc = np.sqrt(alea_unc)
        unc = np.sqrt(unc)

        pred_col.append(prediction_linear)
        pred_logcol.append(prediction)
        pred_sd_totcol.append(unc)
        pred_sd_aleacol.append(alea_unc)
        pred_sd_epicol.append(epi_unc)

    df[f'Prediction_({unit})'] = pred_col
    df['Prediction_log10'] = pred_logcol
    df['SD_total'] = pred_sd_totcol
    df['SD_aleatoric'] = pred_sd_aleacol
    df['SD_epistemic'] = pred_sd_epicol

    return df

def main(args):
    # #region agent log
    try:
        os.makedirs('.cursor', exist_ok=True)
        with open('.cursor/debug.log', 'a') as log:
            log.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H5", "location": "demo_run.py:120", "message": "main() function entry", "data": {"cwd": os.getcwd(), "parameter": args.parameter, "input_file": args.input_file, "checkpoint_dir": args.checkpoint_dir}, "timestamp": int(time.time() * 1000)}) + "\n")
    except Exception:
        pass  # Don't fail if logging fails
    # #endregion
    
    print(os.getcwd())

    # #region agent log
    try:
        with open('.cursor/debug.log', 'a') as log:
            log.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H5", "location": "demo_run.py:130", "message": "Before calling create_csv_sh", "data": {}, "timestamp": int(time.time() * 1000)}) + "\n")
    except:
        pass
    # #endregion

    result = create_csv_sh(args.parameter, args.input_file, args.checkpoint_dir)
    
    # #region agent log
    try:
        with open('.cursor/debug.log', 'a') as log:
            log.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H5", "location": "demo_run.py:137", "message": "After calling create_csv_sh", "data": {"result": result}, "timestamp": int(time.time() * 1000)}) + "\n")
    except:
        pass
    # #endregion
    
    if result is None or result[0] is None:
        return
    
    outfile, output_dir = result

    print('Predicting.. This will take a while..')

    # #region agent log
    try:
        script_exists = os.path.exists('predict.sh')
        script_executable = os.access('predict.sh', os.X_OK) if script_exists else False
        with open('.cursor/debug.log', 'a') as log:
            log.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1,H5", "location": "demo_run.py:145", "message": "Before executing predict.sh", "data": {"script_exists": script_exists, "script_executable": script_executable, "cwd": os.getcwd(), "script_abs_path": os.path.abspath('predict.sh') if script_exists else None}, "timestamp": int(time.time() * 1000)}) + "\n")
    except Exception:
        pass
    # #endregion

    if args.use_gpu:
        exit_code = os.system("export PROTEIN_EMBED_USE_CPU=0;./predict.sh")
    else:
        exit_code = os.system("export PROTEIN_EMBED_USE_CPU=1;./predict.sh")

    # #region agent log
    try:
        with open('.cursor/debug.log', 'a') as log:
            log.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H1", "location": "demo_run.py:151", "message": "After executing predict.sh", "data": {"exit_code": exit_code, "exit_code_hex": hex(exit_code)}, "timestamp": int(time.time() * 1000)}) + "\n")
    except Exception:
        pass
    # #endregion

    output_final = get_predictions(args.parameter, outfile)
    filename = outfile.split('/')[-1]
    # Save final results in the same output directory
    final_output_path = os.path.join(output_dir, filename)
    output_final.to_csv(final_output_path, index=False)
    print(f'Output saved to {output_dir}/{filename}')

if __name__ == "__main__":
    # #region agent log
    try:
        import os
        os.makedirs('.cursor', exist_ok=True)
        with open('.cursor/debug.log', 'a') as log:
            import json
            import time
            log.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H5", "location": "demo_run.py:236", "message": "Script entry point", "data": {"cwd": os.getcwd()}, "timestamp": int(time.time() * 1000)}) + "\n")
    except Exception:
        # Silently ignore logging errors (e.g., read-only filesystem)
        pass
    # #endregion
    
    parser = argparse.ArgumentParser(description="Predict enzyme kinetics parameters.")
    parser.add_argument("--parameter", type=str, choices=["kcat", "km", "ki"], required=True,
                        help="Kinetics parameter to predict (kcat, km, or ki)")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to the input CSV file")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU for prediction (default is CPU)")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Path to the model checkpoint directory")

    # #region agent log
    try:
        with open('.cursor/debug.log', 'a') as log:
            log.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H5", "location": "demo_run.py:252", "message": "Before parse_args", "data": {}, "timestamp": int(time.time() * 1000)}) + "\n")
    except:
        pass
    # #endregion

    args = parser.parse_args()
    args.parameter = args.parameter.lower()

    # #region agent log
    try:
        with open('.cursor/debug.log', 'a') as log:
            log.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H5", "location": "demo_run.py:260", "message": "After parse_args, before main", "data": {"parameter": args.parameter, "input_file": args.input_file, "checkpoint_dir": args.checkpoint_dir}, "timestamp": int(time.time() * 1000)}) + "\n")
    except:
        pass
    # #endregion

    try:
        main(args)
    except Exception as e:
        # #region agent log
        try:
            import traceback
            with open('.cursor/debug.log', 'a') as log:
                log.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": "H5", "location": "demo_run.py:268", "message": "Exception in main()", "data": {"error": str(e), "traceback": traceback.format_exc()}, "timestamp": int(time.time() * 1000)}) + "\n")
        except Exception:
            # Silently ignore logging errors (e.g., read-only filesystem)
            pass
        # #endregion
        raise


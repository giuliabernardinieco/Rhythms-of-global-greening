#!/bin/bash
#PBS -l select=1:ncpus=4:mem=100gb
#PBS -l walltime=24:00:00
#PBS -o ./logs -e ./logs
#PBS -J 128-130

cd ${PBS_O_WORKDIR}

module load miniforge/3
eval "$(~/miniforge3/bin/conda shell.bash hook)"
conda activate venv-wav

# python3 power-analysis-timeseries.py
PYTHON_SCRIPT="power-analysis-timeseries-complete-scales_windows.py"

# Input file and output file
INPUT_FILE="${PBS_O_WORKDIR}/points_csv/points_${PBS_ARRAY_INDEX}.csv"
OUTPUT_FILE1="${PBS_O_WORKDIR}/results/output_sumpower_w1_${PBS_ARRAY_INDEX}.csv"
OUTPUT_FILE2="${PBS_O_WORKDIR}/results/output_sumpower_w2_${PBS_ARRAY_INDEX}.csv"

# Execute the python script
python ${PYTHON_SCRIPT} -i ${INPUT_FILE} -o ${OUTPUT_FILE1} -w ${OUTPUT_FILE2}

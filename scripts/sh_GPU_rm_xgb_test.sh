
#!/bin/bash
#SBATCH --account=f202418352cpcaa1g
#SBATCH --job-name=rm_test1
#SBATCH --mail-user=vadibekyan@astro.up.pt
#SBATCH --mail-type=END,FAIL
#SBATCH --output=/projects/F202418352CPCAA1/logs/rm_test1_%j.out
#SBATCH --error=/projects/F202418352CPCAA1/logs/rm_test1_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=32

set -euo pipefail

module purge
module --ignore_cache load foss/2024a
module --ignore_cache load CUDA/12.6.0
module --ignore_cache load TensorFlow/2.18.1-foss-2024a-CUDA-12.6.0
module --ignore_cache load SciPy-bundle/2024.05-gfbf-2024a
module --ignore_cache load scikit-learn/1.6.1-gfbf-2024a


source /projects/F202418352CPCAA1/venvs/tf218_parquet/bin/activate

export TMPDIR=/projects/F202418352CPCAA1/tmp
export MPLCONFIGDIR=/projects/F202418352CPCAA1/tmp/matplotlib
export XDG_CACHE_HOME=/projects/F202418352CPCAA1/cache

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p /projects/F202418352CPCAA1/logs
mkdir -p "$TMPDIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME/astropy"
mkdir -p /projects/F202418352CPCAA1/results/learning_curve_full

cd /projects/F202418352CPCAA1

python scripts/leave_one_observation_out.py \
  --input-csv model_tables/rm_df_full.csv \
  --output-dir results/xgb_run_01 \
  --features vrad_subtract_mean fwhm_fractional_mean bis_span_fractional_mean contrast_fractional_mean \
  --target true_vrad \
  --model xgboost \
  --model-params '{"n_estimators": 500, "max_depth": 5, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9, "objective": "reg:squarederror", "random_state": 42}'

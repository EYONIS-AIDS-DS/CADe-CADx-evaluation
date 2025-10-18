#!/bin/bash

#SBATCH --job-name=median.ai_ds.lcs_evaluate.process
#SBATCH --partition=median
#SBATCH --time=20-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=60
#SBATCH --mem=300G

readonly CONTAINER=`readlink -f $1`; shift

source init_sbatch.bash

# exec_app $LOCAL_CONTAINER lcseval.ingest
# exec_app $LOCAL_CONTAINER lcseval.ingest.lcs 60
# exec_app $LOCAL_CONTAINER lcseval.ingest.test_definition 60
# exec_app $LOCAL_CONTAINER lcseval.correct
exec_app $LOCAL_CONTAINER lcseval.merge 60
# exec_app $LOCAL_CONTAINER lcseval.pair 60
# exec_app $LOCAL_CONTAINER lcseval.extract.pairing_series
# exec_app $LOCAL_CONTAINER lcseval.extract.pairing_lesions

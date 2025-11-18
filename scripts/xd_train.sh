# bash ./scripts/xd_train.sh
TRAINER=HPT
CFG=xd
SHOTS=0
GPU=1

S_DATASET=coco
OUTPUT_DIR=./results
DATA=./
DIRGPT=${DATA}/gpt_data

DIR=${OUTPUT_DIR}/output_img/${S_DATASET}/${TRAINER}/${CFG}/default
CUDA_VISIBLE_DEVICES=${GPU} python train.py \
--root ${DATA} \
--seed ${SEED} \
--trainer ${TRAINER} \
--dataset-config-file configs/datasets/xd/${S_DATASET}.yaml \
--config-file configs/trainers/${TRAINER}/${CFG}.yaml \
--output-dir ${DIR} \
DATASET.GPT_DIR ${DIRGPT} \
DATASET.NUM_SHOTS ${SHOTS}

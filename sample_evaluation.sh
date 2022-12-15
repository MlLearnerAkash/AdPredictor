#!/bin/bash
datapath=/home/mlserver/patchcore_filling_30s/dataset/mvtec
loadpath=/home/mlserver/patchcore_filling_30s/results/MVTecAD_Results/

modelfolder=ful_mask_182166_1 #7 #S40_13 #IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_152
# modelfolder=IM224_Ensemble_L2-3_P001_D1024-384_PS-3_AN-1
savefolder=evaluated_results'/'$modelfolder

datasets=('bottle' )
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

PYTHONPATH=src python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 --save_segmentation_images  $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 182 --imagesize 182 "${dataset_flags[@]}" mvtec $datapath


#python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 -p  /home/genie/patchcore-inspection/results/MVTecAD_Results/IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0_9/models/mvtec_bottle\

torchrun --nproc_per_node 1 --nnodes 1 --rdzv_id 18635 --rdzv_backend c10d \
 --rdzv_endpoint localhost:29506 inference.py \
 --config "configs/LVSM_spatialgen_decoder_only.yaml" \
 inference_out_dir = ./experiments/spatialgen_evaluation/spiral_cams-512-2
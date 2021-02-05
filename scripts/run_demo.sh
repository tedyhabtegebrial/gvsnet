CUDA_VSIBLE_DEVICES=0 python ../demo.py \
    --dataset=carla_samples \
    --mode=demo \
    --movement_type=circle \
    --data_path=./datasets/carla_samples \
    --output_path=./output/carla_samples_style_1 \
    --pre_trained_gvsnet=./pre_trained_models/carla/gvsnet_model.pt \
    --style_path=./data/sample_styles/carla_1.png \
    # --gpu_id=-1 \


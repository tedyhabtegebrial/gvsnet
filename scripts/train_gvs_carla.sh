python -m torch.distributed.launch --nproc_per_node=4 --nnodes=1 \
            ../train_gvs_model.py \
            --dataset=carla \
            --stereo_baseline=0.54 \
            --batch_size=2 \
            --num_epochs=20 \
            --mode=train \
            --port=8008 \
            --data_path=/path/to/carla \
            --logging_path=/path/tp/logging_path \
            --pre_trained_sun=/path/to/pretrained/sun_model.pt \
            --image_log_interval=2000 \

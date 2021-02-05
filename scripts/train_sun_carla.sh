python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 \
            ../train_sun_model.py \
            --ngpu=1 \
            --dataset=carla \
            --stereo_baseline=0.54 \
            --batch_size=2 \
            --num_epochs=30 \
            --mode=train \
            --port=6071 \
            --data_path=/data/teddy/carla/ \
            --logging_path=/data/teddy/temp/gvsnet_expts/carla/sun \
            --image_log_interval=2000 \


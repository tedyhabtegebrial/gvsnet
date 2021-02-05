python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 \
            ../train_sun_model.py \
            --ngpu=2 \
            --dataset=carla \
            --stereo_baseline=0.54 \
            --batch_size=3 \
            --num_epochs=30 \
            --mode=train \
            --port=6071 \
            --data_path=/netscratch/teddy/carla/raw256x256 \
            --logging_path=/netscratch/teddy/gvsnet_expts


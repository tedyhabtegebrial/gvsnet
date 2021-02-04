python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 train_sun_model.py --ngpu=1 \
            --dataset=carla \
            --batch_size=1 \
            --num_epochs=30 \
            --mode=train \
            --data_path=/netscratch/teddy/carla/raw256x256 \
            --logging_path=/netscratch/teddy/gvsnet_expts


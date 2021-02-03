python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 ../train_sun_model.py --ngpu=1
            --dataset=carla \
            --batch_size=2 \
            --num_epochs=30 \
            --mode=train \
            --data_path=/data/teddy/carla \
            --logging_path=/data/teddy/gvsnet_expts/carla \


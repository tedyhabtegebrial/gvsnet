python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 \
            ../train_gvs_model.py \
            --dataset=carla \
            --stereo_baseline=0.54 \
            --batch_size=2 \
            --num_epochs=20 \
            --mode=train \
            --port=8008 \
            --data_path=/data/teddy/carla/ \
            --logging_path=/data/teddy/temp/gvsnet_expts/carla/gvs \
            --pre_trained_sun=/data/teddy/temp/gvsnet_expts/carla/sun/models/model_epoch_0.pt \
            --image_log_interval=2000 \

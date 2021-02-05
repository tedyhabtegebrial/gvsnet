srun -K --ntasks=2 --gpus-per-task=1 --cpus-per-gpu=6 --mem-per-cpu=6G -p V100-16GB -N 1 \
        --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
        --container-image=/netscratch/enroot/dlcc_pytorch_20.07.sqsh  \
        --container-workdir=`pwd` --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
        python ../train_gvs_model.py \
            --dataset=carla \
            --stereo_baseline=0.54 \
            --batch_size=2 \
            --num_epochs=20 \
            --mode=train \
            --port=6071 \
            --slurm \
            --data_path=/netscratch/teddy/carla/raw256x256 \
            --logging_path=/netscratch/teddy/gvsnet_expts/carla/ltn_adn \
            --pre_trained_sun=/netscratch/teddy/gvsnet_expts/carla/sun/models/model_epoch_0.pt \
            --image_log_interval=2000 \

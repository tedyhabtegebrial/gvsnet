<h2 align=center> Generative View Synthesis </h2>
<div align=center>
This repository contains code accompanying the paper <br>
<i>Generative View Synthesis: From Single-view Semantics to Novel-view Images.</i> <span> <a href="https://tedyhabtegebrial.github.io/">Tewodros Habtegebrial</a></span> , 
<span> <a href="https://varunjampani.github.io/">Varun Jampani</a></span> , 
<span> <a href="http://alumni.soe.ucsc.edu/~orazio/">Orazio Gallo</a></span> , 
<span> <a href="https://av.dfki.de/members/stricker/">Didier Stricker</a></span>  
<br> Presented at NeurIPS 2020. <br> The project page can be found <a href="https://gvsnet.github.io/">here</a>
</div>

<div align=center width=750px class="row">

<div class="column">
    <img src="/docs/assets/GVSNet.png">
  </div>
</div>



#### Dependencies
This code was develped with ```python3.6```
```
scikit-image
torch # tested with versions 1.7.1, but it might work with older versions as well
tqdm
```
#### Instructions
##### Runnig the demo code
###### Pre-trained models
Download a pre-trained model for the CARLA dataset from [here](https://drive.google.com/file/d/1xTRwuo1nGl0MVeNBFJSbwsCGTZvppGY3/view?usp=sharing). Extract it to the folder ```pre_trained_models/carla```

###### Download sample test dataset
Download 34 sample scenes (from the CARLA dataset) for demo purpose [link](https://drive.google.com/file/d/1lStDu9RI4JmU2IR4g0nBB1ZPwY5KIq_Z/view?usp=sharing). Extract the dataset to the folder ```datasets/carla_samples```

###### Run the demo code
```
CUDA_VSIBLE_DEVICES=0 python demo.py \
    --dataset=carla \
    --mode=demo \
    --movement_type=circle \
    --data_path=./datasets/carla_samples \
    --output_path=./output/carla_samples \
    --pre_trained_gvsnet=./pre_trained_models/carla/gvsnet_model.pt \
    --style_path=./data/sample_styles/carla_1.png \

```
###### Controlling style of the generated views
A style image can be pased with the following flag ``` --style_path ```. If not given the color image of input view is used as a style image.

## Training the GVSNet model
### Datasets
Fow downloading the datasets used in our experiments please read instructions here [datasets](/docs/datasets.md)

### Traing the Semantic Uplifting Model
Recommended batch sizes and number of epochs
  * ``` sun model ``` batch_size=12 and above, num_epochs=30
  * ``` ltn+adn ``` batch_size=16 and above, num_epochs=20 
```
cd scripts
./train_sun_carla

```
### Traing the Layered Translation and Appearance Decoder Networks
```
cd scripts
./train_gvs_carla.sh
```

Acknowledgments: This repo builds upon the [SPADE](https://github.com/NVlabs/SPADE) repository from NVIDIA
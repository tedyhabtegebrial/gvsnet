### Generative View Synthesis: From Single-view Semantics to Novel-view Images
Generative View Synthesis: From Single-view Semantics to Novel-view Images.<br>
[Tewodros Habtegebrial](https://tedyhabtegebrial.github.io/),  [Varun Jampani](https://varunjampani.github.io/), [Orazio Gallo](http://alumni.soe.ucsc.edu/~orazio/),  and [Didier Stricker](https://av.dfki.de/members/stricker/).<br>
In NeurIPS 2020

#### Dependencies
```
python3.6
torch # tested with versions 1.7.1, but it should work with older versions as well
apex
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
CUDA_VSIBLE_DEVICES=0 python3 demo.py \
    --mode=demo \
    --batch_size=1 \
    --dataset=carla \
    --movement_type=circle \
    --data_path=./datasets/carla_samples \
    --output_path=./output/carla_samples \
    --style_path=./data/sample_styles/road_2.jpg \
```
###### Controlling style of the generated views
A style image can be pased with the following flag ``` --style_path ```. If not given the color image of input view is used as a style image.
# Exploiting Pre-trained Feature Networks for Generative Adversarial Networks in Audio-domain Loop Generation

## Overview 

Authors: [Yen-Tung Yeh](https://arthurddd.github.io/), [Bo-Yu Chen](https://paulyuchen.com/), [Yi-Hsuan Yang](http://mac.citi.sinica.edu.tw/~yang/)

* This repository contains code for ISMIR2022 paper **Exploiting Pre-trained Feature Networks for Generative Adversarial Networks in Audio-domain Loop Generation**. 

* We provide pre-trained models to generate drum loops and synth loops.

[Paper]() | [Demo](https://arthurddd.github.io/PjLoopGAN/)

## Environment
```
$ conda env create -f environment.yml
```
## Download Pre-trained Checkpoints
* drum fusion 

```
gdown --id 15h07E__BxvaHqBiLwQLe2BzbRU8uPnHF
```
* drum vgg 

```
gdown --id 1cXGXA2b8nLKEClGSPW6ttiwwWErL1Fx4
``` 

* synth fusion 

```
gdown --id 1phrhU-QxQc33adIigzl_6GycmpHvwmo1
```

* synth vgg 

```
gdown --id 1I4RFYFTRzL6V0V8ndFqNHnJIDlyFzf-p
```

## Quick Start 

Generate audio with our pre-trained model. 

### setup 

1. Download the pre-trained checkpoint above.
2. Put your checkpoint into the checkpoint/```loops_type```/```config```
	
	```loops_type``` can ```drum``` or ```synth```, ```config``` can be ```vgg``` or ```fusion```.
	
	For example, If you download the **drum fusion checkpoint**, then put it into the **checkpoint/drum/fusion** folder.  


### generate audio 
* Generate one bar drum loop (```vgg```)

```
bash quick_start/generate_drum_vgg.sh
```

* Generate one bar drum loop (```fusion```)

```
bash quick_start/generate_drum_fusion.sh
```

* Generate one bar synth loop (```vgg```)

```
bash quick_start/generate_synth_vgg.sh
```

* Generate one bar synth loop (```fusion```)

```
bash quick_start/generate_synth_fusion.sh
```
## Dataset

We collected drum loops and synth loops from [Looperman](https://www.looperman.com/). Unfortunately, we can not provide our dataset due to license issues. 

## Training 

In the following, we describe how to train the model from scratch.

### pre-processing 
We mainly follow the same pre-preprocessing method with [LoopTest](https://github.com/allenhung1025/LoopTest) repository.

To preprocess data, modify some settings such as the data path in the code and run codes in the [preprocess](./preprocess) directory with the following orders. 


```
python trim_2_seconds.py 
python extract_mel.py
python make_dataset.py
python compute_mean_std.py
```

### Train the model 

Check [train.sh](./scripts/train.sh) and modify some arguments for training.

```
python3 train.py \
    --size 64 --batch 64 --sample_dir [sample dir]  \
    --checkpoint_dir [model ckpt dir]  --loops_type [kinds of loops] \
    --fusion [fushion or not] \
    --pre_trained_model_type [types of pre-trained feature network] \ 
    --pre_trained_model_path [pre-trained feature network ckpt path]\
    [mel-spectrogram from the previous pre-processing step]
```

* ```sample_dir``` is the directory to store the generated Mel-spectrograms during training. 
* ```checkpoint_dir``` is the directory to store model checkpoints.
* ``` loops_type``` means which kinds of loops you want to generate. [```"synth"```, ```"drums"```]
* ```fusion``` indicates whether using fusion or not. [```"on"```, ```"off"```] 
* ```pre_trained_model_type``` stands for different pre-trained feature networks. [```"vgg"```, ```"autotagging"```, ```"loops_genre"```]
* ```pre_trained_model_path``` is the data path of pre-trained checkpoints.

Note: If ```fusion``` is ```"on"```, then the ```pre_trained_model_type``` can only be ```"loops_genre"``` or ```"autotagging"```. This is because with the ```fusion``` method, ```"vgg"``` is needed for default.  


Last, run the following command.

```
bash scripts/train.sh
```

## Inference 
In the following section, we describe how to generate audio.

Check [generate_audio.sh](./scripts/generate_audio.sh) and modify some arguments

```
python3 inference.py \
    --ckpt [generator_checkpoint] \
    --pics [generate how many audio] \
    --data_path [mean and std for generating] \
    --store_path [where to store audio] \
    --vocoder_folder "melgan/drum_vocoder"
```
* ```ckpt``` is the path of your checkpoint for generating audio.
* ```pics``` means how many mel-spectrograms you want to generate.
* ```data_path``` stores the mean and std for our dataset. [```"data/drum"```, ```"data/synth"```]
* ```store_path``` is the directory to store your own audio.
* ```vocoder_folder``` contains the MelGAN information for generating, we provide our drum vocoder and synth vocoder. [```"melgan/drum_vocoder"```, ```"melgan/synth_vocoder"```]

Last, run the following command.

```
bash scripts/generate_audio.sh
```
## Vocoder 

We use [MelGAN](https://github.com/descriptinc/melgan-neurips) as the vocoder. We trained the vocoder with the looperman dataset and we provide two checkpoints in the [melgan](./melgan) directory, one is for ```drum loop``` and the other is for ```synth loop```. 

## References 

The code comes heavily from the code below


* [Official MelGAN repo][melgan] 
* [LoopTest][looptest]
* [Official Projected GAN repo][pggan]

[melgan]: https://github.com/descriptinc/melgan-neurips
[looptest]: https://github.com/allenhung1025/LoopTest
[pggan]: https://github.com/autonomousvision/projected_gan


## Citation
If you find this repo useful, please kindly cite the following information.

```

```

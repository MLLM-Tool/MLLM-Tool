# Tool_LMM:A Large Multi-Modal Model for Tool Learning
[Chenyu Wang], [Weixin Luo], [Lin Ma], [Shenghua Gao].


**Tool_LMM, School of Information Science and Technology, ShanghaiTech University**

-----

<a href='https://next-gpt.github.io/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
<a href='#'><img src='https://img.shields.io/badge/Demo-Page-purple'></a> 
<a href='https://arxiv.org/pdf/2309.05519'><img src='https://img.shields.io/badge/Paper-PDF-orange'></a> 

This repository hosts the code, data and model weight of **Tool_LMM**, the first end-to-end MM-LLM that perceives input and generates output in arbitrary combinations (any-to-any) of text, image, video, and audio and beyond.


-----------

## 🎉 News 

- [x] [2024.01.16] 🚀🚀 Release the code of Tool_LMM in version `7b_tiva_v0`.
- [√] [2024.01.16] 🔨🧩 Release.
- [x] [2024.01.16] 📢📢 Release the T2M instruction dataset.
- [x] [2024.01.16] 👏👏 Release the checkpoint of NExT-GPT in version [7b_tiva_v0](https://huggingface.co/ChocoWu/nextgpt_7b_tiva_v0) .
- [x] [2023.10.15] 🔨🚀 Update of NExT-GPT in version [7b_tiva_v0](https://huggingface.co/ChocoWu/nextgpt_7b_tiva_v0) .

## 👉 TODO 
- [ ] Collect more data and release v2 dataset.
- [ ] Update Tool_LMM in more types&sizes of LLMs.
- [ ] Empower Tool_LMM with retrieving open-set tools.
- [ ] ...


-----------

<span id='introduction'/>

## Brief Introduction 

NExt-GPT is built on top of existing pre-trained LLM, multimodal encoder and SoTA diffusion models, with sufficient end-to-end instruction tuning.

<p align="center" width="100%">
<a target="_blank"><img src="figures/framework.png" alt="Video-LLaMA" style="width: 90%; min-width: 200px; display: block; margin: auto;"></a>
</p>

- **Multimodal Encoding Stage.** Leveraging established encoders to encode inputs in various modalities, where these representations are projected into language-like representations comprehensible to the LLM through a projection layer.
- **LLM Understanding and Reasoning Stage.** Harnessing an existing open-sourced LLM as the core to process input information for semantic understanding and reasoning. The LLM not only directly generates text tokens but also produces unique “modality signal” tokens that serve as instructions to dictate the decoding layers whether & what modal content to output correspondingly.
- **Multimodal Generation Stage.** Receiving the multimodal signals with specific instructions from LLM (if any), the Transformer-based output projection layers map the signal token representations into the ones that are understandable to following multimodal decoders.

For more technical details, kindly refer to the [paper](https://arxiv.org/pdf/2309.05519.pdf). 

-----------

<span id='Usage'/>

## Getting Started


<span id='all_catelogue'/>

### Table of Contents:
* <a href='#Code Structure'>1. Code Structure</a>
* <a href='#Environment Preparation'>2. Environment Preparation </a>
* <a href='#Training on Your Own'>3. Training/Adapting Tool_LMM on Your Own</a>
  * <a href='#Prepare Pre-trained Checkpoint'>3.1. Preparing Pre-trained Checkpoint</a>
  * <a href='#Prepare Dataset'>3.2. Preparing Dataset </a>
  * <a href='#Train Tool_LMM'>3.3. Training Tool_LMM</a>
* <a href='#Run Tool_LMM System'>4. Running Tool_LMM System</a>
  * <a href='#Prepare checkpoints'>4.1. Preparing checkpoints</a>
  * <a href='#Deploy Demo System'>4.2. Deploying Demo System</a>

****



<span id='Code Structure'/>

### 1. Code Structure 

```
├── data
│   ├── IT_data_ins                           # instruction data
│   │   └── T+X-T_data                    # text+[image/audio/video] to text instruction data
├── code
│   ├── config
│   │   ├──__init__.py
│   │   ├── base.yaml                     # the model configuration 
│   │   └── openllama_peft.yaml                  # instruction-tuning configuration
│   ├── dsconfig
│   │   └──  openllama_peft_stage_1.json                  # deepspeed configuration for instruction-tuning training
│   ├── dataset
│   │   ├──__init__ .py
│   │   ├──_sampler.py
│   │   ├──_utils.py
│   │   ├── catalog.py                    # the catalog information of the dataset
│   │   ├── T+X-T_instruction_dataset.py  # process and load text+x-to-text instruction dataset
│   │   └── concat_dataset.py             # process and load multiple dataset
│   ├── model                     
│   │   ├── ImageBind                     # the code from ImageBind Model
│   │   ├──__init__ .py 
│   │   ├── openllama.py       # the main model file
│   │   ├── agent.py
│   │   └── modeling_llama.py
│   ├── scripts
│   │   └── train.sh                      # training Tool_LMM script
│   ├── header.py
│   ├── process_embeddings.py             # precompute the captions embeddings
│   ├── train.py                          # training
│   └── inference.py                      # inference
├── pretrained_checkpoint                   # frozen params of pretrained modules
│   ├── imagebind_ckpt
│   │   ├──huge                       # version
│   │   │   └──imagebind_huge.pth
│   ├── llm_ckpt
│   │   ├── vicuna_7b
│   │   │   ├── config.json
│   │   │   ├── pytorch_model-00001-of-00002.bin
│   │   │   ├── tokenizer.model
│   │   │   └── ...
│   │   └── ...
├── LICENCE.md
├── README.md
└── requirements.txt
```

<span id='Environment Preparation'/>

### 2. Environment Preparation  <a href='#all_catelogue'>[Back to Top]</a>
Please first clone the repo and install the required environment, which can be done by running the following commands:
```
conda env create -n toollmm python=3.8

conda activate toollmm

# CUDA 11.7
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117

git clone https://github.com/NExT-GPT/NExT-GPT.git
cd toollmm

pip install -r requirements.txt *

conda install -c conda-forge cartopy
conda install -c conda-forge pycocotools
```

<span id='Training on Your Own'/>

### 3. Training/Adapting Your Own Tool-LMM 

####


<span id='Prepare Pre-trained Checkpoint'/>

#### 3.1. Preparing Pre-trained Checkpoint  <a href='#all_catelogue'>[Back to Top]</a>
Please follow the instructions to prepare the ImageBind and Large Language Models(LLM) checkpoints.

- `ImageBind`
The pre-trained checkpoint can be downloaded from [here](https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth) with version `huge`. Afterward, put the `imagebind_huge.pth` file at [[./ckpt/pretrained_ckpt/imagebind_ckpt/huge]](ckpt/pretrained_ckpt/imagebind_ckpt/). 
- `Large Language Models`:
first prepare the LLaMA by following the instructions [[here]](ckpt/pretrained_ckpt/prepare_vicuna.md). Then put the pre-trained model at [[./ckpt/pretrained_ckpt/llm_ckpt/]](ckpt/pretrained_ckptllm_ckpt/). 

|**Base Language Model**|**Maximum Sequence Length**|**Huggingface Delta Weights Address**|
|:-------------:|:-------------:|:-------------:|
|Vicuna-7B |512|[lmsys/vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)|
|Vicuna-13B|512|[lmsys/vicuna-13b-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5)|
|Llama-7B |512|[huggyllama/llama-7b](https://huggingface.co/huggyllama/llama-7b)|
|Llama-13B |512|[huggyllama/llama-13b](https://huggingface.co/huggyllama/llama-13b)|
|Llama2-7B |512|[meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)|
|Llama2-13B|512|[meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)|
|Llama2-Chat-7B|512|[meta-llama/Llama2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)|
|Llama2-Chat-13B|512|[meta-llama/Llama-2-13b-chat-hf](https://huggingface.co/meta-llama/Llama-2-13b-chat-hf)|


<span id='Prepare Dataset'/>

#### 3.2. Preparing Dataset  <a href='#all_catelogue'>[Back to Top]</a>
Please download the following datasets used for model training:

A) T-X pairs data
  - `CC3M` of ***text-image*** pairs, please follow this instruction [[here]](./data/T-X_pair_data/cc3m/prepare.md). Then put the data at [[./data/T-X_pair_data/cc3m]](./data/T-X_pair_data/cc3m).
  - `WebVid` of ***text-video*** pairs, see the [[instruction]](./data/T-X_pair_data/webvid/prepare.md). The file should be saved at [[./data/T-X_pair_data/webvid]](./data/T-X_pair_data/webvid).
  - `AudioCap` of ***text-audio*** pairs, see the [[instruction]](./data/T-X_pair_data/audiocap/prepare.md). Save the data in [[./data/T-X_pair_data/audiocap]](./data/T-X_pair_data/audiocap).

B) Instruction data
  - T+X-T
    - `LLaVA` of the ***visual instruction data***, download it from [here](https://github.com/haotian-liu/LLaVA/blob/main/docs/Data.md), and then put it at [[./data/IT_data/T+X-T_data/llava]](./data/IT_data/T+X-T_data/llava/).
    - `Alpaca` of the ***textual instruction data***, download it from [here](https://github.com/tatsu-lab/stanford_alpaca), and then put it at [[./data/IT_data/T+X-T_data/alpaca/]](data/IT_data/T+X-T_data/alpaca/).
    - `VideoChat`, download the ***video instruction data*** [here](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data), and then put it at [[./data/IT_data/T+X-T_data/videochat/]](data/IT_data/T+X-T_data/videochat/).
    
    Side note：After downloading dataset, please run `preprocess_dataset.py` to preprocess the dataset into a unified format.


<span id='Train Tool_LMM'/>

#### 3.3. Training Tool_LMM  <a href='#all_catelogue'>[Back to Top]</a>

First of all, please refer to the base configuration file [[./code/config/base.yaml]](./code/config/base.yaml) for the basic system setting of overall modules.

Then, the training of Tool_LMM starts with this script:
```angular2html
cd ./code
bash scripts/train.sh
```
Specifying the command:
```angular2html
deepspeed --include localhost:0 --master_addr 127.0.0.1 --master_port 28459 train.py \
    --model nextgpt \
    --stage 1\
    --save_path  ../ckpt/delta_ckpt/nextgpt/7b_tiva_v0/\
    --log_path ../ckpt/delta_ckpt/nextgpt/7b_tiva_v0/log/
```
where the key arguments are:
- `--include`: `localhost:0` indicating the GPT cuda number `0` of deepspeed.
- `--stage`: training stage.
- `--save_path`: the directory which saves the trained delta weights. This directory will be automatically created.
- `--log_path`: the directory which saves the log file.





The whole Tool_LMM training involves 3 steps:

- **Step-1**: Encoding-side LLM-centric Multimodal Alignment. This stage trains the ***input projection layer*** while freezing the ImageBind, LLM, output projection layer.
  
  Just run the above `train.sh` script by setting: `--stage 1`
  
  Also refer to the running config file [[./code/config/stage_1.yaml]](./code/config/stage_1.yaml) and deepspeed config file [[./code/dsconfig/stage_1.yaml]](./code/dsconfig/stage_1.yaml) for more step-wise configurations.

  Note that the dataset used for training in this step is included `dataset_name_list` and the dataset name must precisely match the definition in [[./code/dataset/catalog.py]](./code/dataset/catalog.py)  


- **Step-2**: Decoding-side Instruction-following Alignment. This stage trains the ***output projection layers*** while freezing the ImageBind, LLM, input projection layers.

  Just run the above `train.sh` script by setting: `--stage 2`

  Also refer to the running config file [[./code/config/stage_2.yaml]](./code/config/stage_2.yaml) and deepspeed config file [[./code/dsconfig/stage_2.yaml]](./code/dsconfig/stage_2.yaml) for more step-wise configurations.




- **Step-3**: Instruction Tuning. This stage instruction-tune 1) the ***LLM*** via LoRA, 2) ***input projection layer*** and 3) ***output projection layer*** on the instruction dataset.

  Just run the above `train.sh` script by setting: `--stage 3`

  Also refer to the running config file [[./code/config/stage_3.yaml]](./code/config/stage_3.yaml) and deepspeed config file [[./code/dsconfig/stage_3.yaml]](./code/dsconfig/stage_3.yaml) for more step-wise configurations.



<span id='Run NExT-GPT System'/>

## 4. Training your own TOOLLMM system<a href='#all_catelogue'>[Back to Top]</a>

<span id='Prepare checkpoints'/>

#### 4.1. Preparing Checkpoints

First, loading the pre-trained NExT-GPT system.
- **Step-1**: load `Frozen parameters`. Please refer to <a href='#Prepare Pre-trained Checkpoint'>3.1 Preparing Pre-trained Checkpoint</a>.

- **Step-2**: load `Tunable parameters`. Please put the NExT-GPT system at [[./ckpt/delta_ckpt/nextgpt/7b_tiva_v0]](./ckpt/delta_ckpt/nextgpt/7b_tiva_v0). You may either 1) use the params trained yourselves, or 2) download our checkpoints from [Huggingface](https://huggingface.co/ChocoWu/nextgpt_7b_tiva_v0). 

<span id='Deploy Demo System'/>

#### 4.2. Deploying Gradio Demo
Upon completion of the checkpoint loading, you can run the demo locally via:
```angular2html
cd ./code
bash scripts/app.sh
```
Specifying the key arguments as:
- `--nextgpt_ckpt_path`: the path of pre-trained NExT-GPT params.

---------

## Contact

For any questions or feedback, feel free to contact [Chenyu Wang](wangchy8@shanghaitech.edu.cn).

## Citation

If you find NextGPT useful in your research or applications, please kindly cite:
```
@articles{wang2024,
  title={NExT-GPT: Any-to-Any Multimodal LLM},
  author={Shengqiong Wu and Hao Fei and Leigang Qu and Wei Ji and Tat-Seng Chua},
  journal = {CoRR},
  volume = {abs/2309.05519},
  year={2023}
}
```




## Acknowledgements
You may refer to related work that serves as foundations for our framework and code repository, 
[Vicuna](https://github.com/lm-sys/FastChat), 
[ImageBind](https://github.com/facebookresearch/ImageBind), 
[Stable Diffusion](https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/text2img), 
[AudioLDM](https://github.com/haoheliu/AudioLDM), and
[Zeroscope](https://huggingface.co/cerspense/zeroscope_v2_576w).
We also partially draw inspirations from 
[PandaGPT](https://github.com/yxuansu/PandaGPT), 
[VPGTrans](https://vpgtrans.github.io/), 
[GILL](https://github.com/kohjingyu/gill/), 
[CoDi](https://codi-gen.github.io/),
[Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA),
and [MiniGPT-4](https://github.com/Vision-CAIR/MiniGPT-4).
Thanks for their wonderful works.



## License Notices
This repository is under [BSD 3-Clause License](LICENSE.txt).
NExT-GPT is a research project intended for non-commercial use only. 
One must NOT use the code of NExT-GPT for any illegal, harmful, violent, racist, or sexual purposes. 
One is strictly prohibited from engaging in any activity that will potentially violate these guidelines.
Any potential commercial use of this code should be approved by the authors.

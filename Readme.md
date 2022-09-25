# Speech2Vec Reality Check

> Code for "Homophone Reveals the Truth: A Reality Check for Speech2Vec"
> 
> Paper: [Latest Version (9.24)](https://safe-liar.oss-cn-beijing.aliyuncs.com/Homophone-v9.23.pdf) 

## Requirements

- Free GPU RAM >= 3GB
- Free System RAM >= 30GB
- Pytorch Version >= 1.8.2

## Data

Download the `dataset.zip` (2.4GB) from [Google Drive](https://drive.google.com/drive/folders/1KSWCSttpPOHVaJXJxuqv7U-GCa1n2wJ4?usp=sharing).
Unzip it to the project root. Its structure is shown below:

```
dataset/
├── info
│   ├── 500h_word2wav_keys.pkl
│   ├── 500h_word_counter.pkl
│   ├── 500h_word_split.pkl
│   └── eval
│       ├── all_words_5846.pkl
│       ├── files
│       │   ├── EN-MC-30.txt
│       │   ├── EN-MEN-TR-3k.txt
│       │   ├── EN-MTurk-287.txt
│       │   ├── EN-MTurk-771.txt
│       │   ├── EN-RG-65.txt
│       │   ├── EN-RW-STANFORD.txt
│       │   ├── EN-SIMLEX-999.txt
│       │   ├── EN-SimVerb-3500.txt
│       │   ├── EN-VERB-143.txt
│       │   ├── EN-WS-353-ALL.txt
│       │   ├── EN-WS-353-REL.txt
│       │   ├── EN-WS-353-SIM.txt
│       │   └── EN-YP-130.txt
│       └── homophone.txt
├── split_mfcc_dict.pkl
└── split_mfcc_mean_std.pkl
```

We also provided some speech sentence segment exmaples in `SentenceSegmentExamples.zip`.

This [instruction](jupyters/01_Instruction.ipynb) describes how we generated those files.

## Training

Just run `python 1_train.py`

The full training process (500-epoch) takes 8.4 days on an AMD 3900XT + RTX3090 machine.

----

*use [wandb](https://wandb.ai) to view the training process:*

1. Create  `.wb_config.json`  file in the project root, using the following content:
   
   ```
   {
     "WB_KEY": "Your wandb auth key"
   }
   ```

2. add `--dryrun=False` to the training command, for example:   `python 1_train.py --dryrun=False`

## CheckPoints & Embeddings

The checkpoints and embeddings of every epoch are in `Full500EpochModelsEmbedings.zip` (2.9GB)

The `Rand Init` model corresponds to: `epoch-01_ws0.10_men0.08_loss-1.000000.pkl`

The `500-Epoch` model corresponds to: `epoch499_ws0.15_men0.08_loss0.247943.pkl`

## Statement

I would like to see anyone come forward with counterarguments to my report, and **show me the code** to prove Speech2Vec's validity. I'll be very happy to see that I was wrong.

Feel free to contact me:

Email:  hcs@ruc.edu.cn 

WeChat: 
<img src="https://cdn.huacishu.com/img/202209240747994.jpeg" title="" alt="" width="212">

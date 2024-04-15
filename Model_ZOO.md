
## ATTENTION

important yml params: \
dataset_path: root dir of the dataset \
word_embedding_root: the root dir of the word embedding, for example: the glove path is: word_embedding_root+'/glove/glove.6B.300d.txt' \

The dataset should be: `utzappos`, `mit-states`

The log files will be saved to `/logs/Method/Dataset/Time/` 


The dataset dir tree, follow it to set the `dataset_path` and `word_embedding_root`:

```
$ word_embedding_root $
│
├── $ dataset_path $
│   ├── images
│   ├── compositional-split-natural 
│   └── smetadata_compositional-split-manual.t7
│
├── ......other dataset_path, such as mit-states
│
├── glove
│   ├── glove.6B.100d.txt
│   ├── glove.6B.200d.txt
│   ├── glove.6B.300d.txt
│   └── glove_vocab.txt
│
└──
```

### Word Embedding 

Modify these codes at `models/word_embedding.py`. \ 
Now only support glove.


### CoT
Need download the [Glove](https://drive.google.com/drive/folders/1BE2X70eNMIMkGYwhe01HA4c5jixUQdWd?usp=sharing) to the parent directory of the dataset root. 
See more by following the official repo [CoT](https://github.com/HanjaeKim98/CoT)

``` sh
# train

CUDA_VISIBLE_DEVICES=2 python train.py --cfg configs/CoT.yml
```

### CANet
``` sh
# train

CUDA_VISIBLE_DEVICES=2 python train.py --dataset utzappos --dataset_path /data/sjh/dataset/ut-zappos --word_embedding_root /data/sjh/dataset/ --cfg configs/CANet.yml
```

### SCEN
``` sh
# train

CUDA_VISIBLE_DEVICES=2 python train.py --dataset utzappos --dataset_path /data/sjh/dataset/ut-zappos --word_embedding_root /data/sjh/dataset/ --cfg configs/SCEN.yml
```

### SCEN
``` sh
# train

CUDA_VISIBLE_DEVICES=2 python train.py --dataset utzappos --dataset_path /data/sjh/dataset/ut-zappos --word_embedding_root /data/sjh/dataset/ --cfg configs/SCEN.yml
```

### IVR

IVR does not support mit-states dataset, because it goes wrong at `IVRDataset.same_A_diff_B` method. 

``` sh
# train

CUDA_VISIBLE_DEVICES=2 python train.py --dataset utzappos --dataset_path /data/sjh/dataset/ut-zappos --word_embedding_root /data/sjh/dataset/ --cfg configs/ivr.yml
```

### CompCos
``` sh
# train

CUDA_VISIBLE_DEVICES=2 python train.py --dataset utzappos --dataset_path /data/sjh/dataset/ut-zappos --word_embedding_root /data/sjh/dataset/ --cfg configs/CompCos.yml
```



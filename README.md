This repository contains an updated version of pruning techniques demonstrated in [[1]](#1).  
GLUE benchmark task was updated to ```transformers``` v.4.0.2 from the last compatiable v.2.0.0 so far. Datasets are no longer needed to be downloaded manually; layers removal held the same way.


# Installation
**Transformers and datasets**
```
pip install git+https://github.com/huggingface/transformers
pip install datasets
```
**Get updated run_glue.py**
```
wget https://raw.githubusercontent.com/upunaprosk/transformers/master/examples/run_glue.py
```
# Training

Parameters include all possible ```TrainingArguments``` [[2]](#2) that can be found [here](https://github.com/huggingface/transformers/blob/c4deb7b3ae64f6a4bd0e86cdbd3985de4d24b46e/src/transformers/training_args.py#L95-L419) 
and additional ones (layers removal):   
--remove_layers (e.g. you can remove top three layers by mentioning --remove_layers 9,10,11);   
--freeze_layers (specify layer numbers to freeze during finetuning e.g. 0,1,2 to freeze first three layers);  
--freeze_embeddings;   
--model_type (bert/xlnet/xlm/roberta/distilbert/albert/xlmroberta). 

Run example (CoLA, w&b logger*, top 2 layers _removed_)
```
python run_glue.py \
  --report_to wandb \
  --model_name_or_path bert-base-cased \
  --task_name CoLA \
  --learning_rate 2e-5 \
  --do_train \
  --do_eval \
  --logging_steps 50 \
  --max_seq_length 128 \
  --evaluation_strategy steps \
  --per_gpu_train_batch_size 32 \
  --output_dir /tmp/CoLA \
  --num_train_epochs 3.0 \
  --remove_layers 10,11 \
  --run_name demo\
  --model_type bert
```  
*To initialize w&b run: 
```
import wandb
wandb.login()
```
## References
<a id="1">[1]</a> 
Sajjad, H., Dalvi, F., Durrani, N., & Nakov, P.. (2020). On the Effect of Dropping Layers of Pre-trained Transformer Models.  

<a id="2">[2]</a> 
Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., Cistac, P., Rault, T., Louf, R., Funtowicz, M., Davison, J., Shleifer, S., Platen, P., Ma, C., Jernite, Y., Plu, J., Xu, C., Scao, T., Gugger, S., Drame, M., Lhoest, Q., & Rush, A.. (2019). HuggingFace's Transformers: State-of-the-art Natural Language Processing.

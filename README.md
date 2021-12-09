# torch-ddp
This is a project of data parallel that running on NLP tasks.

You can e-mail Jiamao at `964723423@qq.com` , if you have any questions.

## Requirements

Our code works with the following environment.
* `python>=3.7`
* `pytorch>=1.3`
* `transformers`
* `spacy`


## Tips
you should contain `zh_core_web_sm` belonged spacy when you running the project.

## Run 
0:`python pre_myfile.py`
1:`python generate.py`
2:`python data__process.py`
3:`python python -m torch.distributed.launch --nproc_per_node=1 --use_env train.py.py`





# Machine Translation Example

This repository contains a series of scripts for the whole pipeline

# Running Example

* create a data folder
`make de-en-data`
* copy training and dev corpus inside and then `cd de-en-data/`
* use pipeline to get vocabulary and tokenized training and dev data.
`../tool/pipeline.py --source_input=bitext.de --target_input=bitext.en --source=de --target=en --source_vocab=30000 --target_vocab=30000 --dev_source=dev.de --dev_target=dev.en`
* with source_input defining source corpus, target_input defining target corpus. source_vocab and target_vocab defining the vocabulary size.
* After getting the needed files. Go to parent folder to change Configuration.py, Specify the corpus location and vocabulary location.
* Running can be done with specific GPU. `python __main__.py --gpu=gpu1`
* If early stop is set, the training will stop before the iteration number and calculate the BLEU of dev set.

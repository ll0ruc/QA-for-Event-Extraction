# QA-for-Event-Extraction
A question answering method for Event Extraction on your own Chinese dataset (Also in English dataset).

## Prerequisites

1. Install the packages.
   ```
   pip install pytorch_pretrained_bert
   pip install numpy torch
   ```
   Download BERT-base-chinese model from https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz 
   
   Download bert-base-chinese-vocab from https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt.
   
   Then put these file in 
   ```
   ./Bert-base-chinese
      --bert_config.json
      --pytorch_model.bin
      --vocab.txt
    ```
   
2. Prepare **Your own dataset**.

**`sample.json`**
```
[
  {
    "words": "在冲突中，一名年轻的激进示威者用尖刀刺伤警员。",
    "golden-event-mentions": [
      {
        "event_type": "冲突：冲击",
        "trigger": {
          "text": "刺伤",
          "start": 18,
          "end": 20
        },
        "arguments": [
          {
            "text": "警员",
            "start": 20,
            "end": 22,
            "entity_type": "人物",
            "role": "受害者"
          },
          {
            "text": "激进示威者",
            "start": 10,
            "end": 15,
            "entity_type": "人物",
            "role": "攻击者"
          }
        ]
      }
    ]
  },
]
```
put the data into
```
./data/
   --train.json
   --dev.json
   --test.json
```
3. Add your trigger and argument category in the file ./const.py

4. Design question templates based on your event trigger and argument in ./question.csv

for example
```
   事件类型,事件角色,设计的问题
   冲突：冲击,攻击者,谁发起了这次攻击事件？
```

## Usage

Run:

```bash
sudo python qa-trigger.py
``` 
- Then you can get the "trigger_model.pt" in main directory.
- You can get Precision/Recall/F1 on event trigger extraction.

```bash
sudo python qa-argument.py
``` 
- Then you can get the "argument_model.pt" in main directory.
- You can get Precision/Recall/F1 on event argument extraction.


### Reference

* xinyadu's eeqa repository [[github]](https://github.com/xinyadu/eeqa)
* Nlpcl-lab's bert_event_extraction repository [[github]](https://github.com/nlpcl-lab/bert-event-extraction)

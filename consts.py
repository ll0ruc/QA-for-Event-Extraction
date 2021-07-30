NONE = 'O'
PAD = "[PAD]"
UNK = "[UNK]"
CLS = '[CLS]'
SEP = '[SEP]'
BIO_B = 'B'
BIO_I = 'I'

#论元问题构造模板
question_file = "./question.csv"
#BERT的路径
Bert_path = './bert-base-chinese'

#触发词的类别
Trigger = ["冲突：冲击",
            "冲突：死亡",
             ...,
            '交涉：质疑']
#论元的类别
Role = ["人物","时间", "地点",...,]
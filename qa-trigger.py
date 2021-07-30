import os
import random
import json
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils import data
from consts import PAD, CLS, SEP, UNK, NONE, Trigger, Bert_path

def build_vocab(labels, BIO_tagging=True):
    all_labels = [NONE]
    for label in labels:
        if BIO_tagging:
            all_labels.append('B-{}'.format(label))
            all_labels.append('I-{}'.format(label))
        else:
            all_labels.append(label)
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}

    return all_labels, label2idx, idx2label
    
def calc_metric(y_true, y_pred):
    """
    :param y_true: [(tuple), ...]
    :param y_pred: [(tuple), ...]
    :return:
    """
    num_proposed = len(y_pred)
    num_gold = len(y_true)

    y_true_set = set(y_true)
    num_correct = 0
    for item in y_pred:
        if item in y_true_set:
            num_correct += 1

    print('proposed: {}\tcorrect: {}\tgold: {}'.format(num_proposed, num_correct, num_gold))

    if num_proposed != 0:
        precision = num_correct / num_proposed
    else:
        precision = 1.0

    if num_gold != 0:
        recall = num_correct / num_gold
    else:
        recall = 1.0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1

def find_triggers(labels):
    """
    :param labels: ['B-Conflict:Attack', 'I-Conflict:Attack', 'O', 'B-Life:Marry']
    :return: [(0, 2, 'Conflict:Attack'), (3, 4, 'Life:Marry')]
    """
    result = []
    labels = [label.split('-') for label in labels]

    for i in range(len(labels)):
        if labels[i][0] == 'B':
            result.append([i, i + 1, labels[i][1]])
    for item in result:
        j = item[1]
        while j < len(labels):
            if labels[j][0] == 'I':
                j = j + 1
                item[1] = j
            else:
                break

    return [tuple(item) for item in result]    

candidate_queries = [
['句', '子', '的', '触', '发', '词', '是', '哪', '个', '?'], # 0 what is the trigger in the event?
['触', '发', '词', '?'], # 1 what happened in the event?
]
    
tokenizer = BertTokenizer.from_pretrained(Bert_path, do_lower_case=False)
all_triggers, trigger2idx, idx2trigger =  build_vocab(Trigger ,BIO_tagging=True)


class InitDataset(data.Dataset):
    def __init__(self, fpath=None):
        self.sent, self.trigger, self.event_type, self.answer = [], [], [], []
        with open(fpath, 'r', encoding="utf-8") as f:
            data = json.load(f)
            for sen_i, item in enumerate(data):
                words = item['words']
                if len(words) < 5 or len(words) > 480:
                    continue
                if len(item['golden-event-mentions']) == 0:
                    continue
                for event_mention in item['golden-event-mentions']:
                    event_type = event_mention['event_type']
                    st, ed = event_mention['trigger']['start'], event_mention['trigger']['end']
                    answer = [NONE] * len(words)
                    for i in range(st, ed):
                        if i == st:
                            answer[i] = 'B-{}'.format(event_type)
                        else:
                            answer[i] = 'I-{}'.format(event_type)
                                
                    self.sent.append(words)
                    self.event_type.append(event_type)
                    self.answer.append(answer)


    def __len__(self):
        return len(self.sent)
    
    def __getitem__(self, idx):
        question = candidate_queries[0]
        event_type, answer = self.event_type[idx], self.answer[idx]
        words = self.sent[idx]
        wordslen = len(words)
        question = tokenizer.tokenize(question)
        question_offset = len(question)
        tokens = []
        for w in words:
            sub_token = tokenizer.tokenize(w)
            if len(sub_token) == 0:
                sub_token = [UNK]
            tokens.append(sub_token[0])
        assert len(tokens) == len(words)
        sentence = [CLS] + question + [SEP] + tokens + [SEP]
        words_index = []
        for i in range(question_offset+2, len(sentence)-1):
            words_index.append(i)
        assert len(words_index) == len(words)
        tokens_x = tokenizer.convert_tokens_to_ids(sentence)
        seqlen = len(tokens_x)
        trigger_label = [trigger2idx[a] for a in answer]
        segment_id = [0] * (question_offset+2) + [1] * (len(tokens)+1)
        mask = [1] * seqlen
        assert len(tokens_x) == len(segment_id) == len(mask)

        return tokens_x, seqlen, segment_id, mask, trigger_label, words, words_index

def pad(batch):
    tokens_x_2d, seqlen_1d, segment_2d, mask_2d, trigger_label_2d, words_2d, words_index = list(map(list, zip(*batch)))
    maxlen = np.array(seqlen_1d).max()

    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (maxlen - len(tokens_x_2d[i]))
        segment_2d[i] = segment_2d[i] + [0] * (maxlen - len(segment_2d[i]))
        mask_2d[i] = mask_2d[i] + [0] * (maxlen - len(mask_2d[i]))
        words_index[i] = words_index[i] + [0] * (maxlen - len(words_index[i]))
        trigger_label_2d[i] = trigger_label_2d[i] + [trigger2idx[NONE]] * (maxlen - len(trigger_label_2d[i]))

    return tokens_x_2d, segment_2d, mask_2d, trigger_label_2d, words_2d, words_index

class Net(nn.Module):
    def __init__(self, trigger_size=None, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.bert = BertModel.from_pretrained(Bert_path)
        hidden_size = 768
        self.fc_trigger = nn.Sequential(
            nn.Linear(hidden_size, trigger_size)
        )

    def forward(self, tokens_x_2d, mask_2d, segment_2d, words_index):
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        segment_2d = torch.LongTensor(segment_2d).to(self.device)
        mask_2d = torch.LongTensor(mask_2d).to(self.device)
        words_index = torch.LongTensor(words_index).to(self.device)
        
        batch_size, SEQ_LEN = tokens_x_2d.shape[:]
        encoded_layers, _ = self.bert(input_ids=tokens_x_2d,token_type_ids=segment_2d,attention_mask=mask_2d)
        enc = encoded_layers[-1]
        x = enc
        for i in range(batch_size):
            x[i] = torch.index_select(x[i], 0, words_index[i])
        trigger_logits = self.fc_trigger(x)
        trigger_hat_2d = trigger_logits.argmax(-1)
        
        return trigger_logits, trigger_hat_2d

def train(model, iterator, optimizer, criterion):
    model.train()
    # for i, batch in enumerate(iterator):
    tokens_x_2d, segment_2d, mask_2d, trigger_label_2d, words_2d, words_index = iterator
    optimizer.zero_grad()
    trigger_logits, trigger_hat_2d = model(tokens_x_2d=tokens_x_2d, segment_2d=segment_2d, mask_2d=mask_2d,
                                      words_index=words_index)
                                      
    trigger_label_2d = torch.LongTensor(trigger_label_2d).to(model.device)
    trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])
    trigger_loss = criterion(trigger_logits, trigger_label_2d.view(-1))
    loss = trigger_loss
    loss.backward()
    optimizer.step()
    if step % 100 == 0:  # monitoring
        print("step: {}, loss: {}".format(step, loss.item()))
        
def eval(model, iterator):
    model.eval()

    words_all, triggers_all, triggers_hat_all = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            tokens_x_2d, segment_2d, mask_2d, trigger_label_2d, words_2d, words_index = batch
            _, trigger_hat_2d = model(tokens_x_2d=tokens_x_2d, segment_2d=segment_2d, mask_2d=mask_2d,
                                      words_index=words_index)
            
            words_all.extend(words_2d)
            triggers_all.extend(trigger_label_2d)
            triggers_hat_all.extend(trigger_hat_2d.cpu().numpy().tolist())

    triggers_true, triggers_pred = [], []
    with open('temp', 'w') as fout:
        for i, (words, triggers, triggers_hat) in enumerate(zip(words_all, triggers_all, triggers_hat_all)):
            triggers_hat = triggers_hat[:len(words)]
            triggers_hat = [idx2trigger[hat] for hat in triggers_hat]
            triggers = [idx2trigger[hat] for hat in triggers]

            # [(ith sentence, t_start, t_end, t_type_str)]
            triggers_true.extend([(i, *item) for item in find_triggers(triggers)])
            triggers_pred.extend([(i, *item) for item in find_triggers(triggers_hat)])

            for w, t, t_h in zip(words, triggers, triggers_hat):
                fout.write('{}\t{}\t{}\n'.format(w, t, t_h))

    # get results (classification)
    print('[trigger classification]')
    trigger_p, trigger_r, trigger_f1 = calc_metric(triggers_true, triggers_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p, trigger_r, trigger_f1))
    
    return trigger_f1
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.00003)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--eval_per_epoch", type=int, default=1)
    parser.add_argument("--early_stop", type=int, default= 1* 4)
    parser.add_argument("--trainset", type=str, default="data/train.json")
    parser.add_argument("--devset", type=str, default="data/dev.json")
    parser.add_argument("--testset", type=str, default="data/test.json")

    hp = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # random.seed(hp.seed)
    # np.random.seed(hp.seed)
    # torch.manual_seed(hp.seed)
    # torch.cuda.manual_seed_all(hp.seed)
    model = Net(trigger_size=len(all_triggers), device=device)
    if device == 'cuda':
        model = model.to(device)
    train_dataset = InitDataset(fpath=hp.trainset)
    dev_dataset = InitDataset(fpath=hp.devset)
    test_dataset = InitDataset(fpath=hp.testset)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=2,
                                 collate_fn=pad)
    dev_iter = data.DataLoader(dataset=dev_dataset,
                               batch_size=hp.batch_size,
                               shuffle=False,
                               num_workers=2,
                               collate_fn=pad)
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=hp.batch_size,
                                shuffle=False,
                                num_workers=2,
                                collate_fn=pad)


    optimizer = optim.Adam(model.parameters(), lr=hp.lr, weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss()

    eval_step = len(train_iter) // hp.eval_per_epoch

    best_scores = 0.0
    stop = 0
    break_indic = 0
    for epoch in range(1, hp.n_epochs + 1):
        if break_indic == 1:
            break
        print("=========start epoch #{} ...=======".format(epoch))
        for step, batch in enumerate(train_iter):
            train(model, batch, optimizer, criterion)
            if (step + 1) % eval_step == 0:
                print("=========eval dev ===========")
                print("epoch: {}, step: {} / {}".format(epoch, step + 1, len(train_iter)))
                dev_f1 = eval(model, dev_iter)
                print("========eval test =========")
                print("epoch: {}, step: {} / {}".format(epoch, step + 1, len(train_iter)))
                test_f1 = eval(model, test_iter)

                stop += 1
                if stop > hp.early_stop:
                    print("early_stop Opportunity has been run out")
                    print(
                        "the best dev scores f1 = {} in epoch: {}, step: {} / {}".format(best_result[2], best_result[0],
                                                                                         best_result[1],
                                                                                         len(train_iter)))
                    print("the test scores f1 = {}".format(best_result[-1]))
                    break_indic = 1
                    break
                if dev_f1 > best_scores:
                    print("the newest dev scores in epoch: {}, step: {} / {}".format(epoch, step + 1, len(train_iter)))
                    # metric_test, _ = eval(model, test_iter, fname + '_test',write=False)
                    best_scores = dev_f1
                    best_result = [epoch, step + 1, best_scores, test_f1]
                    stop = 0
                    torch.save(model, "trigger_model.pt")
import os
import random
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.optim as optim
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
from torch.utils import data
from consts import CLS, SEP, UNK, NONE, Role, question_file, BIO_B, BIO_I, Bert_path


def build_vocab():
    all_labels = [NONE, BIO_B, BIO_I]
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}
    return all_labels, label2idx, idx2label

def find_argument(labels):
    """
    :param labels: ['B', 'I', 'O', 'B']
    :return: [(0, 2), (3, 4)]
    """
    result = []
    for i in range(len(labels)):
        if labels[i] == BIO_B:
            result.append([i, i + 1])

    for item in result:
        j = item[1]
        while j < len(labels):
            if labels[j] == BIO_I:
                j = j + 1
                item[1] = j
            else:
                break

    return [tuple(item) for item in result]

def predict_argument(hat, wordslen, key):
    hat = hat[:wordslen]
    argu_hat = [idx2role[h] for h in hat]
    final_pred = {}
    final_pred[key] = []
    predictions = find_argument(argu_hat)
    for idx, pred in enumerate(predictions):
        if pred[1] - pred[0] >9:
            continue
        final_pred[key].append((pred[0], pred[1]))
    return final_pred


def read_query_templates(question_file):
    """Load query templates"""
    query_templates = dict()
    with open(question_file, "r", encoding='utf-8') as f:
        for line in f:
            event_type, role, query = line.strip().split(",")
            if event_type not in query_templates:
                query_templates[event_type] = dict()
            if role not in query_templates[event_type]:
                query_templates[event_type][role] = query[:-1] + " ? 触 发 词 是 [trigger]"

    return query_templates

tokenizer = BertTokenizer.from_pretrained(Bert_path, do_lower_case=False)
all_roles, role2idx, idx2role = build_vocab()
query_templates = read_query_templates(question_file=question_file)

class InitDataset(data.Dataset):
    def __init__(self, fpath=None):
        self.sent, self.trigger, self.event_type, self.answer = [], [], [], []
        self.question, self.argument = [], []
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
                    trigger_span = (st, ed)
                    trigger_token = " ".join(words[st:ed])
                    for role_type in query_templates[event_type]:
                        query = query_templates[event_type][role_type]
                        query = query.replace("[trigger]", trigger_token)
                        event_type_argument_type = "_".join([event_type, role_type])
                        answer = [NONE] * len(words)
                        argument_glod = {}
                        argument_glod[(sen_i, event_type_argument_type)] = []
                        for argument in event_mention['arguments']:
                            gold_argument_type = argument['role']
                            if gold_argument_type == role_type:
                                for i in range(argument['start'], argument['end']):
                                    if i == argument['start']:
                                        answer[i] = BIO_B
                                    else:
                                        answer[i] = BIO_I
                                argument_glod[(sen_i, event_type_argument_type)].append(
                                    (argument['start'], argument['end']))
                        self.sent.append(words)
                        self.trigger.append(trigger_span)
                        self.event_type.append(event_type)
                        self.question.append(query)
                        self.answer.append(answer)
                        self.argument.append(argument_glod)


    def __len__(self):
        return len(self.sent)

    def __getitem__(self, idx):
        words, question = self.sent[idx], self.question[idx]
        arguments, triggers= self.argument[idx], self.trigger[idx]
        event_type, answer = self.event_type[idx], self.answer[idx]
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
        for i in range(question_offset + 2, len(sentence) - 1):
            words_index.append(i)
        assert len(words_index) == len(words)
        tokens_x = tokenizer.convert_tokens_to_ids(sentence)
        seqlen = len(tokens_x)
        argument_label = [role2idx[a] for a in answer]
        segment_id = [0] * (question_offset + 2) + [1] * (len(tokens) + 1)
        mask = [1] * seqlen
        assert len(tokens_x) == len(segment_id) == len(mask)

        return tokens_x, seqlen, segment_id, mask, argument_label, arguments, words_index, wordslen


def pad(batch):
    tokens_x_2d, seqlen_1d, segment_2d, mask_2d, argument_label_2d, arguments, words_index, wordslen = list(map(list, zip(*batch)))
    maxlen = np.array(seqlen_1d).max()

    for i in range(len(tokens_x_2d)):
        tokens_x_2d[i] = tokens_x_2d[i] + [0] * (maxlen - len(tokens_x_2d[i]))
        segment_2d[i] = segment_2d[i] + [0] * (maxlen - len(segment_2d[i]))
        mask_2d[i] = mask_2d[i] + [0] * (maxlen - len(mask_2d[i]))
        words_index[i] = words_index[i] + [0] * (maxlen - len(words_index[i]))
        argument_label_2d[i] = argument_label_2d[i] + [role2idx[NONE]] * (maxlen - len(argument_label_2d[i]))

    return tokens_x_2d, segment_2d, mask_2d, argument_label_2d, arguments, words_index, wordslen


class Net(nn.Module):
    def __init__(self, argument_size=None, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.bert = BertModel.from_pretrained(Bert_path)
        hidden_size = 768
        self.fc_argument = nn.Sequential(
            nn.Linear(hidden_size, argument_size)
        )

    def forward(self, tokens_x_2d, mask_2d, segment_2d, words_index):
        tokens_x_2d = torch.LongTensor(tokens_x_2d).to(self.device)
        segment_2d = torch.LongTensor(segment_2d).to(self.device)
        mask_2d = torch.LongTensor(mask_2d).to(self.device)
        words_index = torch.LongTensor(words_index).to(self.device)
        batch_size, SEQ_LEN = tokens_x_2d.shape[:]
        encoded_layers, _ = self.bert(input_ids=tokens_x_2d, token_type_ids=segment_2d, attention_mask=mask_2d)
        enc = encoded_layers[-1]
        x = enc
        for i in range(batch_size):
            x[i] = torch.index_select(x[i], 0, words_index[i])
        argument_logits = self.fc_argument(x)
        argument_hat_2d = argument_logits.argmax(-1)

        return argument_logits, argument_hat_2d


def train(model, iterator, optimizer, criterion):
    model.train()
    # for i, batch in enumerate(iterator):
    tokens_x_2d, segment_2d, mask_2d, argument_label_2d, arguments, words_index, wordslen = iterator
    optimizer.zero_grad()
    argument_logits, argument_hat_2d = model(tokens_x_2d=tokens_x_2d, segment_2d=segment_2d, mask_2d=mask_2d,
                                           words_index=words_index)

    argument_label_2d = torch.LongTensor(argument_label_2d).to(model.device)
    argument_logits = argument_logits.view(-1, argument_logits.shape[-1])
    argument_loss = criterion(argument_logits, argument_label_2d.view(-1))
    loss = argument_loss
    loss.backward()
    optimizer.step()
    if step % 100 == 0:  # monitoring
        print("step: {}, loss: {}".format(step, loss.item()))


def eval(model, iterator):
    model.eval()

    arguments_all, predict_all = [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            tokens_x_2d, segment_2d, mask_2d, argument_label_2d, arguments, words_index, wordslen = batch
            _, argument_hat_2d = model(tokens_x_2d=tokens_x_2d, segment_2d=segment_2d, mask_2d=mask_2d,
                                      words_index=words_index)

            arg_hat = argument_hat_2d.cpu().numpy().tolist()
            predicts = []
            for j in range(len(arguments)):
                ar_key = list(arguments[j].keys())[0]
                pred = predict_argument(arg_hat[j], wordslen[j], ar_key)
                predicts.append(pred)
            arguments_all.extend(arguments)
            predict_all.extend(predicts)

    # get results (classification)
    gold_arg_n, pred_arg_n, pred_in_gold_n, gold_in_pred_n = 0, 0, 0, 0
    for (id, argu) in enumerate(arguments_all):
        ar_key = list(argu.keys())[0]
        pred_arg = predict_all[id][ar_key]
        gold_arg = argu[ar_key]
        # pred_arg_n
        for argument in pred_arg:
            pred_arg_n += 1
        # gold_arg_n
        for argument in gold_arg:
            gold_arg_n += 1
        # pred_in_gold_n
        for argument in pred_arg:
            if argument in gold_arg:
                pred_in_gold_n += 1
        # gold_in_pred_n
        for argument in gold_arg:
            if argument in pred_arg:
                gold_in_pred_n += 1

    prec_c, recall_c, f1_c = 0, 0, 0
    if pred_arg_n != 0:
        prec_c = 100.0 * pred_in_gold_n / pred_arg_n
    else:
        prec_c = 0
    if gold_arg_n != 0:
        recall_c = 100.0 * gold_in_pred_n / gold_arg_n
    else:
        recall_c = 0
    if prec_c or recall_c:
        f1_c = 2 * prec_c * recall_c / (prec_c + recall_c)
    else:
        f1_c = 0
    print('[argument classification]')
    print('proposed: {}\tcorrect: {}\tgold: {}'.format(pred_arg_n, pred_in_gold_n, gold_arg_n))
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(prec_c, recall_c, f1_c))

    return f1_c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.00003)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--eval_per_epoch", type=int, default=3)
    parser.add_argument("--early_stop", type=int, default=3 * 4)
    parser.add_argument("--trainset", type=str, default="data/train.json")
    parser.add_argument("--devset", type=str, default="data/dev.json")
    parser.add_argument("--testset", type=str, default="data/test.json")

    hp = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Net(argument_size=len(all_roles), device=device)
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
                    torch.save(model, "argument_model.pt")
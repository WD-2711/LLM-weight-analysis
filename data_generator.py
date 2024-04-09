import os
import json

class Dataset:
    def __init__(self, questions=[], answers=[]):
        self.questions = questions
        self.answers = answers

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.answers[idx]
    
    def add_item(self, q_item, a_item):
        self.questions.append(q_item)
        self.answers.append(a_item)
    
    def combine(self, new_dataset):
        self.questions += new_dataset.questions
        self.answers += new_dataset.answers
    
    def max_qlen(self):
        max_len = -1
        for i in self.questions:
            if max_len < len(i):
                max_len = len(i)
        return max_len

def find_json(dir):
    result = []
    for _, _, files in os.walk(dir):
        for file in files:
            if file.endswith('.json'):
                result.append(file)
    return result

def process_bbh(d):
    dataset_bbh = Dataset([], [])
    jsons = find_json(d)
    for j in jsons:
        j = os.path.join(d, j)
        f = open(j, 'r')
        data = json.load(f)
        f.close()
        suffix = data['canary']
        samples = data['examples']
        for s in samples:
            question = "Q: " + s['input'] + '.\n' + suffix + '\n'
            answer = s['target']
            if len(question) > 300:
                continue
            dataset_bbh.add_item(question, answer)
    return dataset_bbh

def process_boolq(d):
    dataset_boolq = Dataset([], [])
    jsons = find_json(d)
    for j in jsons:
        j = os.path.join(d, j)
        f = open(j, 'r')
        for line in f:
            data = json.loads(line)
            question = "Q: " + data['passage'] + '\n' + data['question'] + "?\nPlease output 'True' or 'False'.\n"
            answer = "True" if data['answer'] else "False"
            if len(question) > 300:
                continue
            dataset_boolq.add_item(question, answer)
        f.close()
    return dataset_boolq

def make_datasets(dirs):
    for d in dirs:
        if d == "./data/bbh":
            dataset_bbh = process_bbh(d)
        elif d == "./data/boolq":
            dataset_boolq = process_boolq(d)
    print("[+] dataset_boolq: {:4d} | dataset_bbh: {:4d}".format(len(dataset_boolq), len(dataset_bbh)))
    dataset_bbh.combine(dataset_boolq)
    return dataset_bbh

if __name__ == "__main__":
    dirs = ["./data/bbh", "./data/boolq"]
    dataset = make_datasets(dirs)


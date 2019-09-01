#coding:utf-8
import numpy as np
from sklearn.metrics import roc_auc_score
RESULT_TXT = 'results.txt'

def load_predict_scores(result_file):
    fp = open(result_file)
    author_result = {}
    for line in fp:
        author, paper, label, score = line.strip().split()
        label = int(label)
        score = float(score)
        if author not in author_result:
            author_result[author] = ([], [])
        author_result[author][0].append(label)
        author_result[author][1].append(score)
    return author_result

def main():
    author_result = load_predict_scores(RESULT_TXT)
    res = []
    for author, (labels, results) in author_result.items():
        try:
            score = roc_auc_score(np.array(labels), np.array(results))
            print('{} {}'.format(author, score))
            res.append(score)
        except ValueError:
            print('error {}'.format(author))
    print(np.average(res))

if __name__ == '__main__':
    main()


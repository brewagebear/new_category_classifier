import re
import csv
import pandas as pd
import os
from NaiveBayesClassifier import NaiveBayesClassifier

global test_list

def txt_to_csv(self):
    for dirpath, dirnames, filenames in os.walk(self.word_dir):
        for filename in filenames:
            if filename.split('_')[0] in self.category.keys():
                with open(self.text_path + '/' + filename.split('_')[0] + '/' + filename, 'r', encoding='utf-8') as r:
                    with open(self.csv_path + '/result.csv', 'a', encoding='utf-8', newline='') as w:
                        writer = csv.writer(w, delimiter=',')
                        clear_text = cleanText(r.read())
                        writer.writerow([self.category[filename.split('_')[0]], clear_text])
                        self.word_list.append({"category": filename.split('_')[0], "contents": clear_text})
    print(self.word_list)


def cleanText(read_data):
    # 텍스트에 포함되어 있는 특수 문자 제거
    text = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', read_data).replace('\n', '').replace('\t', '')
    return text

def get_test_file():
    path_dir = '/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/'
    file_list = os.listdir(path_dir)  # path 에 존재하는 파일 목록 가져오기
    file_list.sort()  # 파일 이름 순서대로 정렬

    for i in file_list:
        f = open(path_dir+i)
        test_list.append(f.read())

if __name__ == "__main__":
    model = NaiveBayesClassifier()
    df = pd.read_csv('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/dev/out/result.csv',
                     delimiter=',',
                     header=None,
                     names=['LABLE', 'CONTENT'],
                     encoding='utf-8')
    model.train(df)
    test_list = []
    get_test_file()

    for i in test_list:
        model.category_probability(i)
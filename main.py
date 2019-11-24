import re
import csv
import pandas as pd
import os
from NaiveBayesClassifier import NaiveBayesClassifier

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


if __name__ == "__main__":
    model = NaiveBayesClassifier()
    df = pd.read_csv('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/dev/out/result.csv',
                     delimiter=',',
                     header=None,
                     names=['LABLE', 'CONTENT'],
                     encoding='utf-8')
    model.train(df)
    test_list = []

    test1 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/1.txt', 'r').read()
    test_list.append(test1)
    test2 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/2.txt', 'r').read()
    test_list.append(test2)
    test3 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/3.txt', 'r').read()
    test_list.append(test3)
    test4 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/4.txt', 'r').read()
    test_list.append(test4)
    test5 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/5.txt', 'r').read()
    test_list.append(test5)
    test6 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/6.txt', 'r').read()
    test_list.append(test6)
    test7 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/7.txt', 'r').read()
    test_list.append(test7)
    test8 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/8.txt', 'r').read()
    test_list.append(test8)
    test9 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/9.txt', 'r').read()
    test_list.append(test9)
    test10 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/10.txt', 'r').read()
    test_list.append(test10)
    test11 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/11.txt', 'r').read()
    test_list.append(test11)
    test12 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/12.txt', 'r').read()
    test_list.append(test12)
    test13 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/13.txt', 'r').read()
    test_list.append(test13)
    test14 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/14.txt', 'r').read()
    test_list.append(test14)
    test15 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/15.txt', 'r').read()
    test_list.append(test15)
    test16 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/16.txt', 'r').read()
    test_list.append(test16)
    test17 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/17.txt', 'r').read()
    test_list.append(test17)
    test18 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/18.txt', 'r').read()
    test_list.append(test18)
    test19 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/19.txt', 'r').read()
    test_list.append(test19)
    test20 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/20.txt', 'r').read()
    test_list.append(test20)
    test21 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/21.txt', 'r').read()
    test_list.append(test21)
    test22 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/22.txt', 'r').read()
    test_list.append(test22)
    test23 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/23.txt', 'r').read()
    test_list.append(test23)
    test24 = open('/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/test/24.txt', 'r').read()
    test_list.append(test24)

    for i in test_list:
        model.category_probability(i)
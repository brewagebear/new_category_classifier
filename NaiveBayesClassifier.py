from collections import defaultdict
from pandas import read_table
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import math
import os.path

class NaiveBayesClassifier(object):
    def __init__(self, k=0.5):
        self.k = k
        self.word_probs = []
        self.category = {"interest": 0, "jobs": 1, "moneysupply": 2, "trade": 3}
        self.word_dir = os.path.abspath('.')
        self.text_path = '/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/dev'
        self.csv_path = '/Users/sinsuung/Workspace/Python/unstructured_data_final_project/corpus/dev/out'

    def count_word(self, training_set):
        # 1. interest 2. jobs 3. moneysupply 4.trade
        counts = defaultdict(lambda : [0, 0, 0, 0])
        training_set_arr = training_set.values

        for category, message in training_set_arr:
            for word in self.remove_stop_words(message):
                counts[word][int(category)] += 1
                print(counts[word])
        return counts

    def remove_stop_words(self, doc):
        tagged_list = []

        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(doc)

        for w in word_tokens:
            if w not in stop_words:
                tagged_list.append(w.lower())
        tagged_list = pos_tag(tagged_list)

        result = [t[0] for t in tagged_list if t[1] == "NN"]
        return set(result)

    def train(self, training_set):
        # 카테고리별 문서 세기
        num_interest = len(training_set[training_set['LABLE'] == 0])
        num_jobs = len(training_set[training_set['LABLE'] == 1])
        num_money_supply = len(training_set[training_set['LABLE'] == 2])
        num_trade = len(training_set[training_set['LABLE'] == 3])

        # 학습 데이터 적용하여 모델 만들기
        word_counts = self.count_word(training_set)
        print(word_counts)
        self.word_probs =  self.word_probabilities(word_counts, num_interest, num_jobs, num_money_supply, num_trade)
        print(self.word_probs)


    def word_probabilities(self, counts, total_interest_new, total_jobs_news, total_money_supply_news,
                           total_trade_news):
        k = self.k
        return [(w,
                 (interest + k) / (total_interest_new + 2 * k),
                 (jobs + k) / (total_jobs_news + 2 * k),
                 (money_supply + k) / (total_money_supply_news + 2 * k),
                 (trade + k) / (total_trade_news + 2 * k))
                for w, (interest, jobs, money_supply, trade) in counts.items()]

    def category_probability(self, message):
        word_probs = self.word_probs
        message_words = self.remove_stop_words(message)
        log_prob_if_interest = log_prob_if_jobs = log_prob_if_moneysupply = log_prob_if_trade = 0.0

        for word, prob_if_interest, prob_if_jobs, prob_if_moneysupply, prob_if_trade in word_probs:
            if word in message_words:
                log_prob_if_interest += math.log(prob_if_interest)
                log_prob_if_jobs     += math.log(prob_if_jobs)
                log_prob_if_moneysupply += math.log(prob_if_moneysupply)
                log_prob_if_trade    += math.log(prob_if_trade)
            else:
                log_prob_if_interest += math.log(1.0 - prob_if_interest)
                log_prob_if_jobs += math.log(1.0 - prob_if_jobs)
                log_prob_if_moneysupply += math.log(1.0 - prob_if_moneysupply)
                log_prob_if_trade += math.log(1.0 - prob_if_trade)

        prob_if_interest = math.exp(log_prob_if_interest)
        prob_if_jobs = math.exp(log_prob_if_jobs)
        prob_if_moneysupply = math.exp(log_prob_if_moneysupply)
        prob_if_trade = math.exp(log_prob_if_trade)

        max_Proba = 0
        index_Proba = ''
        if max_Proba < (prob_if_interest / (prob_if_interest + prob_if_trade + prob_if_moneysupply + prob_if_jobs)):
            max_Proba = prob_if_interest / (prob_if_interest + prob_if_trade + prob_if_moneysupply + prob_if_jobs)
            index_Proba = 'interest'
        if max_Proba < (prob_if_jobs / (prob_if_interest + prob_if_trade + prob_if_moneysupply + prob_if_jobs)):
            max_Proba = prob_if_jobs / (prob_if_interest + prob_if_trade + prob_if_moneysupply + prob_if_jobs)
            index_Proba = 'jobs'
        if max_Proba < (prob_if_moneysupply / (prob_if_interest + prob_if_trade + prob_if_moneysupply + prob_if_jobs)):
            max_Proba = prob_if_moneysupply / (prob_if_interest + prob_if_trade + prob_if_moneysupply + prob_if_jobs)
            index_Proba = 'moneysupply'
        if max_Proba < (prob_if_trade / (prob_if_interest + prob_if_trade + prob_if_moneysupply + prob_if_jobs)):
            max_Proba = prob_if_trade / (prob_if_interest + prob_if_trade + prob_if_moneysupply + prob_if_jobs)
            index_Proba = 'trade'

        print(max_Proba,"//",index_Proba)
        # return prob_if_spam  / (prob_if_spam + prob_if_not_spam)
        return max_Proba, index_Proba  # 제일 높은 확률과 카테고리
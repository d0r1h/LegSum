"""WordFrequency [Tf_IDf]"""


import spacy
import numpy as np
import pandas as pd
from rouge import Rouge
from heapq import nlargest
from string import punctuation 
from spacy.lang.en.stop_words import STOP_WORDS

data_path = "../LegSuM/Data/BillSum Processed/catest_processed.csv"
data = pd.read_csv(data_path)

stopwords = list(STOP_WORDS)
nlp = spacy.load('en_core_web_sm')
rouge = Rouge()

punctuation = punctuation + '\n'

def GetWordFrequency(doc):
  word_frequnecy_ = {}
  for word in doc:
    if word.text.lower() not in stopwords:
      if word.text.lower() not in punctuation:
        if word.text not in word_frequnecy_.keys():
          word_frequnecy_[word.text] = 1
        else:
          word_frequnecy_[word.text] += 1
  return word_frequnecy_


"""**For Whole Set**"""

text = data['clean_text']
GoldSummaries =  data['summary']
SysSummaries = []

def summaries_case(case):

  doc = nlp(case)
  tokens = [token.text for token in doc]
  word_frequnecy = GetWordFrequency(doc)
  max_frequency = max(word_frequnecy.values())

  for word in word_frequnecy.keys():
      word_frequnecy[word] = word_frequnecy[word] / max_frequency

  sentence_tokens = [i for i in doc.sents]

  def GetSentenceScore(sentence_tokens):
    sentence_score_ = {}
    for sent in sentence_tokens:
      for word in sent:
        if word.text.lower() in word_frequnecy.keys():
          if sent not in sentence_score_.keys():
            sentence_score_[sent] = word_frequnecy[word.text.lower()]
          else:
            sentence_score_[sent] += word_frequnecy[word.text.lower()]
    return sentence_score_

  sentence_score = GetSentenceScore(sentence_tokens)
  sentence_length = int(len(sentence_tokens)*0.1)

  summary_list  = nlargest(sentence_length, sentence_score, key=sentence_score.get)
  Summary = ' '.join([word.text for word in summary_list])
  return Summary

for i, case in enumerate(data['clean_text']):
  try:
    summary = summaries_case(case)
    SysSummaries.append(summary)
    print(i)
  except Exception as e:
    SysSummaries.append(np.NaN)
    print(e, 'for' ,i)


DFSUMResuts = pd.DataFrame(zip(SysSummaries, GoldSummaries), columns = ['SysSummaries', 'GoldSummaries'])
TfIdfRouge = rouge.get_scores(DFSUMResuts['SysSummaries'], DFSUMResuts['GoldSummaries'], avg=True)
TfIdfRouge = pd.DataFrame(TfIdfRouge).set_index([["recall", "precision", "f-measure"]])

path = "./"
TfIdfRouge.to_csv(path + 'TfIdfRouge.csv', index=True, header=True)
"""Luhn's algorithm.ipynb
"""

import sys
import nltk
import pandas as pd
from rouge import Rouge
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.luhn import LuhnSummarizer

score_path = "./"
path = "../LegSuM/Data/BillSum Processed/catest_processed.csv"
data = pd.read_csv(path)

nltk.download("punkt")
rouge = Rouge()

sys.setrecursionlimit(5000)

def summarize(text, sumarizer, SENTENCES_COUNT):
    sentences_ = []
    doc = text
    doc_ = PlaintextParser(doc, Tokenizer("en")).document
    for sentence in sumarizer(doc_, SENTENCES_COUNT):
        sentences_.append(str(sentence))

    summm_ = " ".join(sentences_)
    return summm_

data["LuhnSummary"] = data["clean_text"].map(
                                  lambda x: summarize(x, LuhnSummarizer(), 5)
                                    )
def RougeScore(ModelScore, ModelSummary):

    standard_summary = data["summary"]
    ModelScore_ = rouge.get_scores(ModelSummary, standard_summary, avg=True)
    ModelDF = pd.DataFrame(ModelScore_).set_index(
        [["recall", "precision", "f-measure"]]
    )
    ModelDF.to_csv(score_path + ModelScore + '.csv', index=True, header=True)
    return ModelDF

LuhnRouge = RougeScore("LuhnRouge", data["LuhnSummary"])
LuhnRouge.to_csv(score_path +'LuhnRouge.csv', index=True, header=True)

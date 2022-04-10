"""
Extractive Algorithms 

KL
Lsa
LexRank
TextRank
SumBasic
"""

import sys
import nltk
import pandas as pd
from rouge import Rouge
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.kl import KLSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

score_path = "./"
path = "../LegSum/Data/BillSum Processed/catest_processed.csv"
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

data["LexRankSummary"] = data["clean_text"].map(
    lambda x: summarize(x, LexRankSummarizer(), 5)
)
data["KLSummary"] = data["clean_text"].map(
    lambda x: summarize(x, KLSummarizer(), 5)
)
data["TextRankSummary"] = data["clean_text"].map(
    lambda x: summarize(x, TextRankSummarizer(), 5)
)
data["SumBasicSummary"] = data["clean_text"].map(
    lambda x: summarize(x, SumBasicSummarizer(), 5)
)
data["LsaSummary"] = data["clean_text"].map(
    lambda x: summarize(x, LsaSummarizer(), 5)
)


def RougeScore(ModelScore, ModelSummary):

    standard_summary = data["summary"]
    ModelScore_ = rouge.get_scores(ModelSummary, standard_summary, avg=True)
    ModelDF = pd.DataFrame(ModelScore_).set_index(
        [["recall", "precision", "f-measure"]]
    )
    ModelDF.to_csv(ModelScore, index=True, header=True)
    return ModelDF

LexRouge = RougeScore("LexRouge", data["LexRankSummary"])
KLRouge = RougeScore("KLRouge", data["KLSummary"])
TextRankRouge = RougeScore("TextRankRouge", data["TextRankSummary"])
SumBasicRouge = RougeScore("SumBasicRouge", data["SumBasicSummary"])
LsaRouge = RougeScore("LsaRouge", data["LsaSummary"])

LexRouge.to_csv(score_path +'LexRouge.csv', index=True, header=True)
KLRouge.to_csv(score_path +'KLRouge.csv', index=True, header=True)
TextRankRouge.to_csv(score_path +'TextRankRouge.csv', index=True, header=True)
SumBasicRouge.to_csv(score_path +'SumBasicRouge.csv', index=True, header=True)
LsaRouge.to_csv(score_path +'LsaRouge.csv', index=True, header=True)
"""T5_Small [billsum]"""

import sys
import torch
import pandas as pd
from rouge import Rouge
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

dataset = load_dataset("billsum")
test_cases =  dataset['ca_test']

rouge = Rouge()
device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "Frederick0291/t5-small-finetuned-billsum"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)


"""### Test DataSet"""

CasesText = test_cases['text']
GoldSummary = test_cases['summary']


SystemSummary = []
for i, case in enumerate(CasesText):

  batch = tokenizer(case, return_tensors="pt").to(device)
  summary = model.generate(**batch, max_length=3000, min_length=500)
  Summary = tokenizer.batch_decode(summary, skip_special_tokens=True)

  SystemSummary.append(Summary)
  print(i)


SystemSummaryFinal = []
for i in SystemSummary:
  SystemSummaryFinal.append((i[0]))

Summaries = pd.DataFrame(list(zip(GoldSummary, SystemSummaryFinal)), columns =['GoldSummary', 'SystemSummary'])


path = "./"
Summaries.to_csv(path + "T5_Small.csv", header=True, index=False)


system_summary = Summaries['SystemSummary']
standard_summary = Summaries['GoldSummary']

sys.setrecursionlimit(5000)

score = rouge.get_scores(system_summary, standard_summary, avg=True)
T5small = pd.DataFrame(score).set_index([['recall','precision','f-measure']])
T5small.to_csv(path + "T5smallRouge.csv", header=True)


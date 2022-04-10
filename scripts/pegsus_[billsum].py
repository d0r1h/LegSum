"""
Pegsus [billsum]
"""

import torch
import pandas as pd
from rouge import Rouge
from datasets import load_dataset
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

dataset = load_dataset("billsum")
test =  dataset['ca_test']

CasesText = test['text']
GoldSummary = test['summary']

rouge = Rouge()

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "google/pegasus-billsum"
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)
pegasus_model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)

"""**Inference On TestSet**"""

SystemSummary = []
for i, case in enumerate(CasesText):

  strtolist = []
  strtolist.append(case)

  batch = pegasus_tokenizer(strtolist, truncation=True, padding="longest", return_tensors="pt").to(device)
  summary = pegasus_model.generate(**batch)
  summary_final = pegasus_tokenizer.batch_decode(summary, skip_special_tokens=False)
  SystemSummary.append(summary_final)

  print(i)
  strtolist.clear()

SystemSummaryFinal = []

for i in SystemSummary:
  SystemSummaryFinal.append((i[0]))

Summaries = pd.DataFrame(list(zip(GoldSummary, SystemSummaryFinal)), columns =['GoldSummary', 'SystemSummary'])
Summaries.to_csv("PegsusSummaries.csv", header=True, index=False)


"""### **Pegsus Model Score on CA_Test**"""

PegsusSummaries = pd.read_csv('PegsusSummaries.csv')

system_summary = PegsusSummaries['SystemSummary']
standard_summary = PegsusSummaries['GoldSummary']

score = rouge.get_scores(system_summary, standard_summary, avg=True)
PegsusRouge = pd.DataFrame(score).set_index([['recall','precision','f-measure']])
PegsusRouge.to_csv("PegsusRouge.csv", header=True)
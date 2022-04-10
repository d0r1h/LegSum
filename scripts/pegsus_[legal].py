"""
Pegsus [legal]
"""

import torch
import pandas as pd
from rouge import Rouge
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

dataset = load_dataset("billsum")
test_cases =  dataset['ca_test']

rouge = Rouge()
device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "nsi319/legal-pegasus"

tokenizer = AutoTokenizer.from_pretrained(model_name)  
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)



"""**Test DataSet**"""

CasesText = test_cases['text']
GoldSummary = test_cases['summary']

SystemSummary = []

for i, case in enumerate(CasesText):
    
    batch = tokenizer.encode(case, truncation=True, padding="longest", return_tensors="pt").to(device)
    
    summary = model.generate(batch, min_length=1024, num_beams=9)
    Summary = tokenizer.batch_decode(summary, skip_special_tokens=True)
    
    SystemSummary.append(Summary)
    print(i)


SystemSummaryFinal = []
for i in SystemSummary:
  SystemSummaryFinal.append((i[0]))

Summaries = pd.DataFrame(list(zip(GoldSummary, SystemSummaryFinal)), columns =['GoldSummary', 'SystemSummary'])

path = "./"

Summaries.to_csv(path + "LegalPegsus.csv", header=True, index=False)

LegalPegsusSummaries = pd.read_csv(path + "LegalPegsus.csv")

system_summary = LegalPegsusSummaries['SystemSummary']
standard_summary = LegalPegsusSummaries['GoldSummary']

score = rouge.get_scores(system_summary, standard_summary, avg=True)

LegalPegsusRouge = pd.DataFrame(score).set_index([['recall','precision','f-measure']])
LegalPegsusRouge.to_csv("LegalPegsusRouge.csv", header=True)
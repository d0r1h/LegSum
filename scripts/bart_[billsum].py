"""
BART [Billsum]
"""
import torch
import pandas as pd
from rouge import Rouge
from transformers import pipeline
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

torch.manual_seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_name = "murali-admin/bart-billsum-1"

dataset = load_dataset("billsum")
test_cases =  dataset['ca_test'] # California bills

rouge = Rouge()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)


"""Test DataSet"""
CasesText = test_cases['text']
GoldSummary = test_cases['summary']

SystemSummary = []
for i, case in enumerate(CasesText):
    
    batch = tokenizer(case, truncation=True, padding="longest", return_tensors="pt").to(device)  
    summary = model.generate(**batch, max_length=3000, min_length=500)
    Summary = tokenizer.batch_decode(summary, skip_special_tokens=True)

    SystemSummary.append(Summary)
    print(i)

SystemSummaryFinal = []
for i in SystemSummary:
  SystemSummaryFinal.append((i[0]))

Summaries = pd.DataFrame(list(zip(GoldSummary, SystemSummaryFinal)), columns =['GoldSummary', 'SystemSummary'])

path = "./"
Summaries.to_csv(path + "BartBillSum.csv", header=True, index=False)

"""
Inference on test set into batches
"""

BartSum = pd.read_csv(path + "BartBillSum.csv")
system_summaryUp = [''.join(i.split('.')[:2]) for i in BartSum['SystemSummary']]

BartSum['system_summaryUp'] = system_summaryUp

system_summary = BartSum['SystemSummary']
standard_summary = BartSum['GoldSummary']
system_summaryUp = BartSum['system_summaryUp']

score = rouge.get_scores(system_summaryUp, standard_summary, avg=True)
BartRouge = pd.DataFrame(score).set_index([['recall','precision','f-measure']])
BartRouge.to_csv(path + "BartRouge.csv", header=True)
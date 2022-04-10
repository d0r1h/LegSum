"""
BigBird_Pegasus [bigpatent]
"""

import torch
import pandas as pd
from rouge import Rouge
from transformers import pipeline
from datasets import load_dataset
from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

torch.manual_seed(0)

dataset = load_dataset("billsum")
test_cases =  dataset['ca_test']

rouge = Rouge()
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint = "google/bigbird-pegasus-large-bigpatent"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = BigBirdPegasusForConditionalGeneration.from_pretrained(checkpoint).to(device)


"""Test DataSet"""

CasesText = test_cases['text']
GoldSummary = test_cases['summary']

SystemSummary = []
for i, case in enumerate(CasesText):
    
    input_ids = tokenizer(case, return_tensors='pt').to("cuda")
    sequences = model.generate(**input_ids)

    summary = tokenizer.batch_decode(sequences,
                                 skip_special_tokens=True)
    SystemSummary.append(summary)
    print(i)

SystemSummaryFinal = []
for i in SystemSummary:
  SystemSummaryFinal.append((i[0]))

Summaries = pd.DataFrame(list(zip(GoldSummary, SystemSummaryFinal)), columns =['GoldSummary', 'SystemSummary'])

path = "./"

Summaries.to_csv(path + "bigbird.csv", header=True, index=False)

"""**BART Model Score on Test**"""

Bigbird = pd.read_csv(path + "bigbird.csv")
system_summary = Bigbird['SystemSummary']
standard_summary = Bigbird['GoldSummary']

score = rouge.get_scores(system_summary, standard_summary, avg=True)
BigbirdRouge = pd.DataFrame(score).set_index([['recall','precision','f-measure']])
BigbirdRouge.to_csv(path + "BigbirdRouge.csv", header=True)
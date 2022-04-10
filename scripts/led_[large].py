"""LED [large]
"""

import torch
import pandas as pd
from rouge import Rouge
from transformers import pipeline
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import LEDForConditionalGeneration, LEDTokenizer

torch.manual_seed(0)

dataset = load_dataset("billsum")
test_cases =  dataset['ca_test']

device = "cuda" if torch.cuda.is_available() else "cpu"

rouge = Rouge()
tokenizer = LEDTokenizer.from_pretrained("allenai/led-large-16384-arxiv")
model = LEDForConditionalGeneration.from_pretrained("allenai/led-large-16384-arxiv", return_dict_in_generate=True).to(device)

"""Test DataSet"""

CasesText = test_cases['text']
GoldSummary = test_cases['summary']

SystemSummary = []

for i, case in enumerate(CasesText):
    
    input_ids = tokenizer(case, return_tensors="pt").input_ids.to("cuda")
    global_attention_mask = torch.zeros_like(input_ids)
    global_attention_mask[:, 0] = 1

    sequences = model.generate(input_ids, global_attention_mask=global_attention_mask).sequences
    Summary = tokenizer.batch_decode(sequences, skip_special_tokens=True)

    SystemSummary.append(Summary)
    print(i)

SystemSummaryFinal = []
for i in SystemSummary:
  SystemSummaryFinal.append((i[0]))

Summaries = pd.DataFrame(list(zip(GoldSummary, SystemSummaryFinal)), columns =['GoldSummary', 'SystemSummary'])

path = "./"

Summaries.to_csv(path + "led-large.csv", header=True, index=False)

"""**BART Model Score on Test**"""

LEDlarge = pd.read_csv(path + "led-large.csv")

system_summary = LEDlarge['SystemSummary']
standard_summary = LEDlarge['GoldSummary']

score = rouge.get_scores(system_summary, standard_summary, avg=True)

LEDlargeRouge = pd.DataFrame(score).set_index([['recall','precision','f-measure']])

LEDlargeRouge.to_csv(path + "LEDlargeRouge.csv", header=True)


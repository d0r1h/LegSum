
LegSum: Legal Document Summarization




## Results 

We ran both experiments (Extractive and Abstracive) on BillSum Dataset (ca_test) and results with pre-trained models are presented below.

| Algorithm / model | Rouge-1 | Rouge-2 | Rouge-L|
| ---- | ---- | ---- | ----|
| KL			      |	 24.44 | 9.74	| 21.98 |
| LSA 	              |	 30.85 | 12.45	| 27.64 |
| SumBasics	      |	 31.01 | 12.61	| 27.83 |
| Bert  		      |	 33.29 | 15.17	| 29.67 |
| Tf-Idf 		      |	 33.97 | 15.98	| 29.92 |
| LexRank 	      |  36.83 | 18.98  | 32.95 |  
| TextRank 	      |  36.57 | 19.10  | 32.35 |
| Luhn’s Algorithm  |  37.48 | 19.93  | 33.35 |
| BART		      |	 26.02  | 11.87  | 22.02 |
| Pegasus(small)   |  28.61  | 12.19  | 25.88 |
| T5(small)             | 32.99   | 15.52  | 30.21 |
| BillPegasus         | 34.25   | 16.63  | 30.22 |



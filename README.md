<p align="center">
    <br>
    <img src="https://github.com/d0r1h/LegSum/blob/main/assets/LegSum.png" width="300"/>
    <br>
<p>

<p align="center">
    <a href="https://huggingface.co/spaces/d0r1h/LegSum">
    <img alt="app" src="https://img.shields.io/website?down_color=red&down_message=offline&up_color=yello&up_message=onine&url=https%3A%2F%2Fhuggingface.co%2Fspaces%2Fd0r1h%2FLegSum">
    </a>
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fd0r1h%2FLegSum&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
    <a href="https://twitter.com/intent/tweet?text=Legal Document Summarization text:&url=https%3A%2F%2Fgithub.com%2Fd0r1h%2FSAR%2F">
    <img alt="tweet" src="https://img.shields.io/twitter/url?url=https%3A%2F%2Fgithub.com%2Fd0r1h%2FSAR%2F">
    </a>
  </p>  
  
  
<h4 align="center">
    <p> Legal Document Summarization from classical approaches to State-of-the-art methods</p>
</h4>

<h3 align="center">
    <img src="https://github.com/d0r1h/LegSum/blob/main/assets/legsum_app.png", width="550"></a>
</h3>

This repository accompanying the code for my master's thesis <b>LegSum: Legal Document Summarization</b>


### Notebook:

#### Abstractive Methods

| Notebook | Colab | Model checkpoint |
| ------ | ------ |  ------ | 
| T5 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegSum/blob/main/scripts/notebooks/T5_Small_%5Bbillsum%5D.ipynb) | [Frederick0291/t5-small-finetuned-billsum](https://huggingface.co/Frederick0291/t5-small-finetuned-billsum) |
| BART billsum | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d0r1h/LegSum/blob/main/scripts/notebooks/BART_%5BBillsum%5D.ipynb) | [murali-admin/bart-billsum-1](https://huggingface.co/murali-admin/bart-billsum-1) |
| BART xsum | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LegSum/blob/main/scripts/notebooks/BART_%5Bdistilbart_xsum%5D.ipynb) | [sshleifer/distilbart-xsum-12-6](https://huggingface.co/sshleifer/distilbart-xsum-12-6) |
| Pegasus Legal | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d0r1h/LegSum/blob/main/scripts/notebooks/Pegsus_%5Blegal%5D.ipynb) | [nsi319/legal-pegasus](https://huggingface.co/nsi319/legal-pegasus) |
| Pegasus billsum | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d0r1h/LegSum/blob/main/scripts/notebooks/Pegsus_%5Bbillsum%5D.ipynb) | [google/pegasus-billsum](https://huggingface.co/google/pegasus-billsum) |
| BigBird | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d0r1h/LegSum/blob/main/scripts/notebooks/BigBird_Pegasus_%5Bbigpatent%5D.ipynb) | [google/bigbird-pegasus-large-bigpatent](https://huggingface.co/google/bigbird-pegasus-large-bigpatent) |
| LED | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d0r1h/LegSum/blob/main/scripts/notebooks/LED_%5Blarge%5D.ipynb) | [allenai/led-large-16384-arxiv](https://huggingface.co/allenai/led-large-16384-arxiv) |

#### Extractive Methods

| Notebook | Colab |
| ------ | ------ | 
| Extractive | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d0r1h/LegSum/blob/main/scripts/notebooks/Extractive.ipynb) | 
| Kmeans Bertsum | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d0r1h/LegSum/blob/main/scripts/notebooks/Kmeans_Bertsum.ipynb) | 
| Luhn's algorithm | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d0r1h/LegSum/blob/main/scripts/notebooks/Luhn's_algorithm.ipynb) | 
| TF-IDF | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d0r1h/LegSum/blob/main/scripts/notebooks/WordFrequency_%5BTf_IDf%5D.ipynb) | 

### DataSet:

1. **BillSum**
    
    - [Official github repository](https://github.com/FiscalNote/BillSum) [🤗 Dataset loader](https://huggingface.co/datasets/billsum)
    - Processed and clean version of data can be found [here](https://github.com/d0r1h/LegSum/tree/main/Data)

### Results: 

Following results are on BillSum Dataset (ca_test) with pre-trained models and extractive methods


| Algorithm / model | Rouge-1 | Rouge-2 | Rouge-L|
| ---- | ---- | ---- | ----|
**Extractive**
| KL			      |	 24.44 | 9.74	| 21.98 |
| LSA 	              |	 30.85 | 12.45	| 27.64 |
| SumBasics	      |	 31.01 | 12.61	| 27.83 |
| Bert  		      |	 33.29 | 15.17	| 29.67 |
| Tf-Idf 		      |	 33.97 | 15.98	| 29.92 |
| LexRank 	      |  36.83 | 18.98  | 32.95 |  
| TextRank 	      |  36.57 | 19.10  | 32.35 |
| Luhn’s Algorithm  |  37.48 | 19.93  | 33.35 |
**Abstractive**
| BART		      |	 26.02  | 11.87  | 22.02 |
| Pegasus(small)   |  28.61  | 12.19  | 25.88 |
| T5(small)             | 32.99   | 15.52  | 30.21 |
| BillPegasus         | 34.25   | 16.63  | 30.22 |



## Demo

[Space Link 🤗](https://huggingface.co/spaces/d0r1h/LegSum)

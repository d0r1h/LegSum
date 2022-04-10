"""KmeanSum Bertsum.ipynb
"""

import os
import numpy as np
import pandas as pd
from rouge import Rouge
from nltk.cluster import KMeansClusterer
from scipy.spatial import distance_matrix
from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances_argmin_min

pd.options.mode.chained_assignment = None

import nltk
nltk.download('punkt')

rouge = Rouge()
embedder = SentenceTransformer('distiluse-base-multilingual-cased')

path = "../LegSuM/Data/BillSum Processed/catest_processed.csv"
data = pd.read_csv(path)


"""**For Test DataSet**"""

SystemSummary = []
GoldSummary = data['summary']
CaseText = data['clean_text']

def SummariseCase(case, cluster):

  sentences = nltk.sent_tokenize(case)
  sentences = [sentence.strip() for sentence in sentences]
  tempdata_ = pd.DataFrame(sentences, columns=['sentence'])
  tempdata_['embeddings'] = tempdata_['sentence'].apply(get_sentence_embeddings)

  NUM_CLUSTERS=cluster
  iterations=25

  X = np.array(tempdata_['embeddings'].tolist())
  kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,repeats=iterations,avoid_empty_clusters=True)
  assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

  tempdata_['cluster'] = pd.Series(assigned_clusters, index=tempdata_.index)
  tempdata_['centroid'] = tempdata_['cluster'].apply(lambda x: kclusterer.means()[x]) 

  tempdata_['distance_from_centroid'] = tempdata_.apply(distance_from_centroid, axis=1)
  summary = ' '.join(tempdata_.sort_values('distance_from_centroid',ascending = True). \
                   groupby('cluster').head(1). \
                   sort_index()['sentence'].tolist())
  
  return summary

data[['clean_text', 'summary']].sample(5)

for i, cases in enumerate(data['clean_text']):

  try:
    summary_ = SummariseCase(cases, 10)
    SystemSummary.append(summary_)
    print(i)
    
  except Exception as e:
    SystemSummary.append(np.NaN)
    print(e, 'for' ,i)


Summaries = pd.DataFrame(zip(GoldSummary, SystemSummary), columns = ['GoldSummary', 'SystemSummary'])
Summaries.sample(3)

Summaries.dropna(inplace=True)
Summaries.reset_index(inplace=True, drop=True)

def RougeScore():

    standard_summary = Summaries["GoldSummary"]
    ModelSummary =  Summaries["SystemSummary"]
    
    ModelScore_ = rouge.get_scores(ModelSummary, standard_summary, avg=True)
    ModelDF = pd.DataFrame(ModelScore_).set_index(
        [["recall", "precision", "f-measure"]]
    )
    return ModelDF

KmeansRouge = RougeScore()

path = "./"
KmeansRouge.to_csv(path + "KmeansRouge.csv", index=True, header=True)
# NGS
Dynamic node category-sensitive hypergraph inferring and homo-heterogeneous neighbor feature and spatial dependency learning for prediction of drug-related microbes

# Operating environment
pytroch == 2.1.0  
matplotlib == 3.8.0  
numpy == 1.26.0  
python == 3.9.18  

# File Introduction
data_process.py : Processing drug and microbial similarities and associations, forming embeddings, adjacency matrices, etc.  
early_stoppoing.py : In order to save better parameters for the model  
parameters.py : Hyperparameters of the model  
tools4roc_pr.py : Evaluate the model  
train.py : train the model  
Model517.py ,Transformer.py ,model_all.py : Define the model  
ST1.slsx : Top 20 candidates for every drug  
# data
  drug_drug_interaction_adj.txt:Interactions between drugs
  drugsimilarity.zip：Similarities between drugs
  microbe_microbe_similarity:Similarities between microbes
  net1.mat:Adjacency matrix of drug and microbe heterogeneous graph
# run step
1.data_process.py  
2.train.py  
Before running train.py, you need to create two folders, best_parameter and result, to store the model's parameters and training results

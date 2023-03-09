# PPI_GBERT
In order to replicate the results mentioned in paper, please follow the following steps:
1. The input to the Graph-BERT model are two files, which you can download using the following links for each dataset:
2. 
  Human PPI: https://drive.google.com/drive/folders/1KX9ybM_Mh2RXvqCmN7vJ2X_ywg12DJtq?usp=share_link
  
  C. elegan: https://drive.google.com/drive/folders/1vNPpyGYDFHjHd2ylB77SmAwnh0tG18Vw?usp=share_link
  
  Drosophila: https://drive.google.com/drive/folders/1_4oZZVvBFiub0FPOyQxVEmtZVM3kHqkQ?usp=share_link
  
  E. coli: https://drive.google.com/drive/folders/1wK6S_bZIDWaxW9IIU_GukyWD_ateqL2P?usp=share_link
  
2. Run the command "python script_1_preprocess.py" to compute node WL code, intimacy based subgraph batch, node hop distance.

3. Run the command "python script_2_pre_train.py" for pre-training the Graph-BERT.

4. Please run the command "python script_3_fine_tuning.py" as the entry point to run the model on node classification.

5. script_4_evaluation_plots.py is used for plots drawing and results evaluation purposes.


For more details about the Graph-BERT and the environment required to run the model, you can refer to this link:
  Link: https://github.com/jwzhanggy/Graph-Bert

# IMPROVE tool

A feature based random forest model to priotize neoepitopes. 
This repository is bulid for tool usage. 
Look at all scripts made for the paper at: 
https://github.com/SRHgroup/IMPROVE_paper


# Improve has two steps: 
1. Feature_calculations.py
2. Preidct_immunogenicity.py

## 1. Feature calculations 

Feature_calculations.py calculates all features used for the simple model besides: 

NetMHCExp and Expression: 
  These two features can be obtained by this tool: 

Forginess score: 
  Can be obtained by https://github.com/andrewrech/antigen.garnish
  

## IMPROVE dependency and installation
Save all programs to a folder 

## IMPROVE Simple model

## PRIME 
https://github.com/GfellerLab/PRIME 

### Prime simple depends on:
https://github.com/GfellerLab/MixMHCpred

## NetMHCpan 
https://services.healthtech.dtu.dk/services/NetMHCpan-4.1/

## NetMHCstaboan 
https://services.healthtech.dtu.dk/services/NetMHCstabpan-1.0/

## Self similarity with kernel self similarity 
https://github.com/SRHgroup/self_similarity

Save all tools to a folder used for the feature calculation "--ProgramDir"

## IMPROVE TME excluded

PrioScore: Priority score from Mupexi: https://services.healthtech.dtu.dk/services/MuPeXI-1.1/

CelPrev: Obtained from PyClone: https://github.com/Roth-Lab/pyclone

## IMPROVE TME included (RNA sequencing data is needed)

CYT: Geometric mean of GZMA and PRF1 

MCPmean: Mean score of the 10 cell populations estimated from MCP-Counter 


  
## 2. Predict immunogenicity 
You can predit with three diffrent versions:
  1. Simple: All features from simple model and addition of NetMHCExp, Expression and Foreginess score 
  2. TME excluded: All features from simple model with addition of priority score and cellular prevelance
  3. TME included: All features from TME excluded including RNA feature:
    CYT (geometirc mean of GZMA and PRF1) and MCP-mean gained from MCP-counter. 
  
# Usage

Feature calculation tool
--------------------------
python3 bin/feature_calculations.py --file data/test_file_for_feature_calculation_small.tsv --dataset "testing" --ProgramDir "/Users/annieborch/Documents/programs/" --outfile "data/calculated_features_test.tsv" --TmpDir "/Users/annieborch/Documents/programs/"

Predict tools: 
----------------------
python3 bin/Predict_immunogenicity.py --file data/calculated_features_test.tsv --model TME_excluded  --outfile "output_prediction_tme_excluded.tsv" 

python3 bin/Predict_immunogenicity.py --file data/calculated_features_test.tsv --model TME_included  --outfile "output_prediction_tme_included.tsv" 

python3 bin/Predict_immunogenicity.py --file data/calculated_features_test.tsv --model Simple  --outfile "output_prediction_simple.tsv" 
  
  


#!/bin/bash

# Run feature calculation and prediction 

python3 bin/feature_calculations.py --file data/test_file_for_feature_calculation.tsv --dataset "testing" --ProgramDir "/Users/annieborch/Documents/programs/" --outfile "data/calculated_features_test.tsv" --TmpDir "/Users/annieborch/Documents/programs/"  && python3 bin/Predict_immunogenicity.py --file data/calculated_features_test.tsv --model Simple  --outfile "output_prediction_simple.tsv" 
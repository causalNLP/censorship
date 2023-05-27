#!/bin/bash

# Replace <google_sheet_link> with the actual link to the Google Sheet
curl -o dataset_train.csv "https://drive.google.com/file/d/1q90A7B5iibF5-Hr1SzYFMqkzZjNcd6Ux/view?usp=sharing/export?format=csv"
curl -o dataset_test.csv "https://drive.google.com/file/d/1VlsdUPNrmyVWc8r76Jvj_At_dQF3rKMP/view?usp=sharing/export?format=csv"
curl -o dataset_val.csv "https://drive.google.com/file/d/1D2uqA35mA_TzstMCU2I42QllE5w0nZ1S/view?usp=sharing/export?format=csv"
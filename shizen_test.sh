module load python
python OpenNMT-py/translate.py -model corpora/subs_custom/stop/raw/shizen_model_step_100000.pt -src corpora/shizen/shizen_src.txt -output corpora/subs_custom/stop/raw/shizen_pred.txt -replace_unk -verbose

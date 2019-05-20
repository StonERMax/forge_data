
# echo ###############################################
# python -W ignore davis_synth_create.py --num 80

# echo ###############################################
# echo ###############################################

# python -W ignore segtrack_synth_create.py --num 40


# echo ###############################################
# echo ###############################################
# python -W ignore youtube_synth_create.py --num 40


cd ~/dataset/VIRAT_dataset
python virat_ano.py

cd /home/islama6a/local/forge_data
python -W ignore virat_synth_create.py

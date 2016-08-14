echo $1
./gru_trainer.sh >> $1
echo -e "\n" >> $1
./gru_tester.sh >> $1
echo -e "\n" >> $1
cat src/settings_char.py >> $1

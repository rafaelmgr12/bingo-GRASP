mkdir data_output
start=`date +%s`
python3 ANN_FIT_2.py
end=`date +%s`

runtime=$((end-start))
echo ${runtime} >> saida.dat


done




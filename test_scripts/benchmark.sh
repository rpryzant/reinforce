
runs=3

echo "random baseline...300 x $runs"
for ((i=1; i<=$runs; i++));
do
    echo "\t run $i..."
    python main.py -p randomBaseline -b 300 -csv > "random-$i.csv"
done
echo "average score (baseline):"
python test_scripts/analyze.py random-1.csv random-2.csv random-3.csv 

	
echo "q replay...300 x $runs"
for ((i=1; i<=$runs; i++));
do
    echo "\t run $i..."
    python main.py -p linearReplayQ -b 300 -csv > "linear-$i.csv"
done
echo "average score (q replay):"
python test_scripts/analyze.py linear-1.csv linear-2.csv linear-3.csv 


echo "sarsa Lambda...300 x $runs"
for ((i=1; i<=$runs; i++));
do
    echo "\t run $i..."
    python main.py -p sarsaLambda -b 300 -csv > "sarsaLambda-$i.csv"
done
echo "average score (sarsa lambda):"
python test_scripts/analyze.py sarsaLambda-1.csv sarsaLambda-2.csv sarsaLambda-3.csv


###### UNCOMMENT FOR MORE TESTS #########

#echo "sarsa...300 x $runs"
#for ((i=1; i<=$runs; i++));
#do
#    echo "\t run $i..."
#    python main.py -p sarsa -b 300 -csv > "sarsaLambda-$i.csv"
#done
#echo "average score (sarsa):"
#python test_scripts/analyze.py sarsaLambda-1.csv sarsaLambda-2.csv sarsaLambda-3.csv


#echo "q learning...300 x $runs"
#for ((i=1; i<=$runs; i++));
#do
#    echo "\t run $i..."
#    python main.py -p linearQ -b 300 -csv > "sarsaLambda-$i.csv"
#done
#echo "average score (q learning):"
#python test_scripts/analyze.py sarsaLambda-1.csv sarsaLambda-2.csv sarsaLambda-3.csv


#echo "nn...300 x $runs"
#for ((i=1; i<=$runs; i++));
#do
#    echo "\t run $i..."
#    python main.py -p nn -b 300 -csv > "sarsaLambda-$i.csv"
#done
#echo "average score (nn):"
#python test_scripts/analyze.py sarsaLambda-1.csv sarsaLambda-2.csv sarsaLambda-3.csv


	

rm *.csv
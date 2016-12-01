
player="$1"
batches="$2"
runs="$3"


for ((i=1; i<=$runs; i++))
do
    echo "$player-$i"
    python ../main.py -p $player -b $batches -csv > "$player-$i.csv"
done
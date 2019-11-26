echo "speaker,segs,time" > stats.csv
SEGS_CUM=0
TIME_CUM=0
for D in segments/*/; do 
    SEGS=$(ls -l "$D" | wc -l)
    TIME=$(soxi -D "$D"/*.wav | paste -sd+ - | bc -l)
    SEGS_CUM=$(echo "$SEGS_CUM" + "$SEGS" | bc -l)
    TIME_CUM=$(echo "$TIME_CUM" + "$TIME" | bc -l)
    AVG=$(echo "scale=4; $SEGS / $TIME" | bc -l)
    echo "$D","$SEGS","$TIME" >> stats.csv
    echo "$D" segs: "$SEGS" time: "$TIME" avg: "$AVG"
done
AVG_CUM=$(echo "scale=4; $SEGS_CUM / $TIME_CUM" | bc -l)
echo Cumulative segs: "$SEGS_CUM" time: "$TIME_CUM" avg: "$AVG_CUM"

BASE_PATH=/home/hutch_research/data/2019_ml_teaching
INSTRUCTORS=(
    100
    198
    203
    226
    388
    504
    660
    784
    925
)
for INST in "${INSTRUCTORS[@]}"; do
    ln -s $BASE_PATH/$INST $INST
done

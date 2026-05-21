#!/bin/bash
INPUT=$1
MLP=$2
DIR_DATA=$3
# DIR_DATA=/home/jp3h/Documents/mlp/tests/gp/simulated
# INPUT=$(find "$DIR_DATA" -name "*input_simulated-*.tsv" | head -n1 | tail -n1)
# # DIR_DATA=/home/jp3h/Documents/mlp/tests/gp/empirical
# # INPUT=$(find "$DIR_DATA" -name "*.tsv" | grep -v "LINEAR" | grep -v "MLP" | sort | head -n6 | tail -n1)
# MLP=/home/jp3h/Documents/mlp/target/release/mlp
# mkdir ${DIR_DATA}/mlp_misc_output
if [[ $(dirname $INPUT) == "." ]]
then
    echo "Please use the full path of the input file ($INPUT)."
    exit 1
fi
# # Fixed parameters
# N_EPOCHS=1000
# N_BURNIN_EPOCHS=100
# F_PATIENT_EPOCHS=0.01
# F_VALIDATION=0.1
# N_BATCHES=1
# N_HIDDEN_LAYERS=1
# N_HIDDEN_NODES=700
# ACTIVATION="Linear"
# LEARNING_RATE=0.00001
# OPTIMISER="Adam"
# DROPOUT_RATES=0.00
N_REPS=5
N_FOLDS=10
# Setup the output directory
ID=$(basename ${INPUT%.tsv*})
mkdir ${DIR_DATA}/mlp_misc_output/${ID}
cd ${DIR_DATA}/mlp_misc_output/${ID}
# Setup the output file
OUTPUT=$(echo $INPUT | sed 's/input_simulated/output_simulated/g' | sed "s/.tsv/-MLP_E${N_EPOCHS}_B${N_BURNIN_EPOCHS}_F${F_PATIENT_EPOCHS}_V${F_VALIDATION}_B${N_BATCHES}_H${N_HIDDEN_LAYERS}.tsv/g")
if [[ $(echo $(basename $OUTPUT) | grep "^output" | wc -l) -eq 0 ]]
then
    OUTPUT="$(dirname $OUTPUT)/output-$(basename $OUTPUT)"
fi
echo -e "datasets\treps\tfolds\tnt\tnv\tmodels\tcorr\tr2" > $OUTPUT
N=$(echo $(cut -f1 $INPUT | wc -l) - 1 | bc)
M=$(echo "scale=0; $N / $N_FOLDS" | bc)
P=$(head -n1 $INPUT | cut -f2- | awk '{print NF}')
echo "$INPUT -->  $OUTPUT (N=$N; M=$M; P=$P)"
# Run replicated k-fold cross-validation
for REP in $(seq 1 $N_REPS)
do
    # REP=1
    IDX_SHUFFLED=($(shuf --random-source=<(yes $REP) -e $(seq 2 $(echo $N + 1 | bc))))
    # echo ${IDX_SHUFFLED[@]}
    for FOLD in $(seq 1 $N_FOLDS)
    do
        # FOLD=1
        IDX_INI=$(echo "(($FOLD - 1) * $M) + 1" | bc)
        IDX_FIN=$(echo "$FOLD * $M" | bc)
        IDX_TRAINING=()
        IDX_VALIDATION=()
        for i in $(seq 0 $N)
        do
            if [[ ($i -ge $IDX_INI) && ($i -le $IDX_FIN) ]]
            then
                # echo "$i; ${IDX_SHUFFLED[i]}"
                IDX_VALIDATION+=("${IDX_SHUFFLED[i]}")
            else
                IDX_TRAINING+=("${IDX_SHUFFLED[i]}")
            fi
        done
        # echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        # echo "IDX_TRAINING: ${IDX_TRAINING[@]}"
        # echo "IDX_VALIDATION: ${IDX_VALIDATION[@]}"
        TRAINING_IDX=${IDX_TRAINING[@]}
        VALIDATION_IDX=${IDX_VALIDATION[@]}
        head -n1 $INPUT > TRAINING_SET.tmp
        head -n1 $INPUT > VALIDATION_SET.tmp
        awk -v idx="$TRAINING_IDX" 'BEGIN {FS="\t"; OFS="\t"} { split(idx, a, " "); for (i in a) b[a[i]] } NR in b' $INPUT >> TRAINING_SET.tmp
        awk -v idx="$VALIDATION_IDX" 'BEGIN {FS="\t"; OFS="\t"} { split(idx, a, " "); for (i in a) b[a[i]] } NR in b' $INPUT >> VALIDATION_SET.tmp
        # Selecting between ReLU and Linear activation functions (selectiong the latter renders the model simply linear)
        time ${MLP} \
            -f TRAINING_SET.tmp \
            -o OUTPUT.tmp.json \
            -v \
            --hyperparameter-optimisation \
            --range-hidden-layers="1,1,1" \
            --range-hidden-layer-nodes="700,700,700" \
            --range-dropout-rates="0.0,0.0,0.01" \
            --range-learning-rates="1e-5,1e-5,1e-5" \
            --range-n-epochs="1000,1000,1000" \
            --range-n-burnin-epochs="100,100,100" \
            --range-f-patient-epochs="0.01,0.01,0.01" \
            --range-f-validation="0.1,0.1,0.1" \
            --range-n-batches="1,1,1" \
            --selection-costs="MSE" \
            --selection-optimisers="Adam,GradientDescent" \
            --selection-activations="ReLU,Linear" \
            --selection-weights-initialisations="He,Cauchy" \
            --skip-marginals
        # rm *.svg
        # time ${MLP} \
        #     -f TRAINING_SET.tmp \
        #     -o OUTPUT.tmp.json \
        #     -v \
        #     --n-epochs=${N_EPOCHS} \
        #     --n-burnin-epochs=${N_BURNIN_EPOCHS} \
        #     --f-patient-epochs=${F_PATIENT_EPOCHS} \
        #     --f-validation=${F_VALIDATION} \
        #     --n-batches=${N_BATCHES} \
        #     --n-hidden-layers=${N_HIDDEN_LAYERS} \
        #     --n-hidden-nodes=${N_HIDDEN_NODES} \
        #     --activation=${ACTIVATION} \
        #     --learning-rate=${LEARNING_RATE} \
        #     --optimiser=${OPTIMISER} \
        #     --dropout-rates=${DROPOUT_RATES} \
        #     --skip-marginals
        time ${MLP} \
            -f VALIDATION_SET.tmp \
            -m OUTPUT.tmp.json \
            -v \
            --predict-only
        cut -f1 VALIDATION_SET.tmp > true.tmp
        cut -f1 OUTPUT.tmp-predictions.tsv > pred.tmp
        paste -d'\t' true.tmp pred.tmp > true_vs_pred.tmp
        # head true_vs_pred.tmp
        NT=$(tail -n+2 TRAINING_SET.tmp | wc -l)
        NV=$(tail -n+2 VALIDATION_SET.tmp | wc -l)
        U_TRUE=$(tail -n+2 true_vs_pred.tmp | awk '{sum+=$1; count++} END {printf("%.21f\n", sum/count)}')
        U_PRED=$(tail -n+2 true_vs_pred.tmp | awk '{sum+=$2; count++} END {printf("%.21f\n", sum/count)}')
        S_TRUE=$(tail -n+2 true_vs_pred.tmp | awk -v U_TRUE="$U_TRUE" '{sum+=(($1-U_TRUE)^2); count++} END {printf("%.21f\n", sqrt(sum/(count-1)))}')
        S_PRED=$(tail -n+2 true_vs_pred.tmp | awk -v U_PRED="$U_PRED" '{sum+=(($2-U_PRED)^2); count++} END {printf("%.21f\n", sqrt(sum/(count-1)))}')
        V_TRUE_PRED=$(tail -n+2 true_vs_pred.tmp | awk -v U_TRUE="$U_TRUE" -v U_PRED="$U_PRED" '{sum+=(($1-U_TRUE)*($2-U_PRED)); count++} END {printf("%.21f\n", sum/(count-1))}')
        MSE=$(tail -n+2 true_vs_pred.tmp | awk '{sum+=(($1 - $2)^2); count++} END {printf("%.21f\n", sum/(count - 1))}')
        CORR="$(echo "scale=12; $V_TRUE_PRED / (($S_TRUE * $S_PRED) + 0.00000000001)" | bc | sed 's/[.]/0./g')"
        R2="$(echo "scale=12; 1.00 - ($MSE / (($S_TRUE^2) + 0.00000000001))" | bc | sed 's/[.]/0./g')"
        echo "U_TRUE: $U_TRUE"
        echo "U_PRED: $U_PRED"
        echo "S_TRUE: $S_TRUE"
        echo "S_PRED: $S_PRED"
        echo "V_TRUE_PRED: $V_TRUE_PRED"
        echo "MSE: $MSE"
        echo "CORR: $CORR"
        echo "R2: $R2"
        # Update the output file
        echo -e "$(basename $INPUT)\t$REP\t$FOLD\t$NT\t$NV\tmlp\t$CORR\t$R2" >> $OUTPUT
        # Clean-up
        rm *.tmp* *.svg
    done
done

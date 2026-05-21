#!/bin/bash
if [[ $1 == "-h" || $1 == "--help" ]]; then
    echo "Usage: sh mlp.sh MLP_PATH ANALYSIS_TYPE FNAME_INPUT [GP ARGS...]"
    echo "MLP: path to the mlp executable (https://github.com/jeffersonfparil/mlp)."
    echo "ANALYSIS_TYPE: 'trials' or 'gp'."
    echo "FNAME_INPUT: path to the input file for the analysis:"
    echo -e "\t- For trials analysis (i.e. to extract the marginal effects of each genotype), this should be a tab-separated file with a header row and columns for year, site, treatment, entry, replication, and response variable."
    echo -e "\t- For genomic prediction analysis (i.e. repeated k-fold cross-validation), this should be a tab-separated file with a header row and columns for the response variable followed by the features."
    echo "GP ARGS: additional arguments for genomic prediction analysis. These are: "
    echo -e "\t- N_REPS: number of replications of k-fold cross-validation"
    echo -e "\t- N_FOLDS: number of folds for k-fold cross-validation"
    echo -e "\t- BASE_SEED: base seed for reshuffling genotypes per replication of k-fold cross-validation (the seed for each replication will be BASE_SEED + REP)"
    # TODO ADD EXAMPLES see simulate.sh for examples...
    exit 0
fi
MLP=$1
ANALYSIS_TYPE=$2
if [[ -z $ANALYSIS_TYPE ]]; then echo "Error: Missing argument for analysis type (trials or gp)."; exit 1; fi
if [[ $ANALYSIS_TYPE != "trials" && $ANALYSIS_TYPE != "gp" ]]; then echo "Error: Invalid analysis type. Expected 'trials' or 'gp', got '${ANALYSIS_TYPE}'."; exit 1; fi
FNAME_INPUT=$3
if [[ -z $MLP ]]; then echo "Error: Missing argument for MLP path."; exit 1; fi
if [[ ! -f $MLP ]]; then echo "Error: MLP executable not found at the specified path: '${MLP}'."; exit 1; fi
if [[ -z $FNAME_INPUT ]]; then echo "Error: Missing argument for input file name."; exit 1; fi
if [[ ! -f $FNAME_INPUT ]]; then echo "Error: Input file not found at the specified path: '${FNAME_INPUT}'."; exit 1; fi
if [[ $ANALYSIS_TYPE == "trials" ]]; then
    echo "################################################################"
    echo "### Running multi-layer perceptron model for trials analysis ###"
    echo "################################################################"
    N_EPOCHS=500
    F_PATIENT_EPOCHS=0.01
    N_BATCHES=1
    N_HIDDEN_LAYERS=1
    N_HIDDEN_NODES=64
    MARGINALS_ORDER=1
    BNAME_INPUT=$(basename $FNAME_INPUT)
    BNAME_OUTPUT=$(echo $BNAME_INPUT | sed 's/input_simulated/output_simulated/g' | sed "s/.tsv$/-MLP_E${N_EPOCHS}_F${F_PATIENT_EPOCHS}_B${N_BATCHES}_H${N_HIDDEN_LAYERS}_M${MARGINALS_ORDER}.json/g")
    if [[ $(echo $BNAME_OUTPUT | grep "output_simulated") == "" ]]; then
        BNAME_OUTPUT="output_empirical-${BNAME_OUTPUT}"
    fi
    FNAME_OUTPUT_JSON=$(pwd)/$BNAME_OUTPUT
    FNAME_OUTPUT_MARGINALS=${FNAME_OUTPUT_JSON%.*}-marginals.tsv
    TMP_OUTDIR="$(pwd)/tmp_dir-${BNAME_OUTPUT%.*}"
    echo "INPUT: $FNAME_INPUT"
    echo "OUTPUT: $FNAME_OUTPUT_MARGINALS"
    echo "TMP_OUTDIR: $TMP_OUTDIR"
    time ${MLP} \
        -f ${FNAME_INPUT} \
        -o ${FNAME_OUTPUT_JSON} \
        -v \
        --n-epochs=${N_EPOCHS} \
        --f-patient-epochs=${F_PATIENT_EPOCHS} \
        --n-batches=${N_BATCHES} \
        --n-hidden-layers=${N_HIDDEN_LAYERS} \
        --n-hidden-nodes=${N_HIDDEN_NODES} \
        --marginals-order=${MARGINALS_ORDER}
    mkdir $TMP_OUTDIR
    mv $FNAME_OUTPUT_JSON $TMP_OUTDIR
    mv *.svg $TMP_OUTDIR
    mv *.png $TMP_OUTDIR
else
    echo "#########################################################################################################"
    echo "### Running multi-layer perceptron model for genomic prediction with repeated k-fold cross-validation ###"
    echo "#########################################################################################################"
    N_REPS=$4
    N_FOLDS=$5
    BASE_SEED=$6
    if [[ -z $BASE_SEED ]]; then echo "Error: Missing argument for base seed for random number generation (BASE_SEED)."; exit 1; fi
    if [[ -z $N_REPS ]]; then echo "Error: Missing argument for number of replications of k-fold cross-validation (N_REPS)."; exit 1; fi
    if [[ -z $N_FOLDS ]]; then echo "Error: Missing argument for number of folds for k-fold cross-validation (N_FOLDS)."; exit 1; fi
    N=$(echo $(cut -f1 $FNAME_INPUT | wc -l) - 1 | bc)
    M=$(echo "scale=0; $N / $N_FOLDS" | bc)
    P=$(head -n1 $FNAME_INPUT | cut -f2- | awk '{print NF}')
    while [[ $M -lt 10 ]]; do
        N_FOLDS=$(echo "$N_FOLDS - 1" | bc)
        M=$(echo "scale=0; $N / $N_FOLDS" | bc)
    done
    if [[ $N_FOLDS -lt 2 ]]; then
        echo "Error: Not enough folds for k-fold cross-validation. Please reduce the number of folds or increase the number of observations."
        exit 1
    else
        echo "Using $N_FOLDS folds for k-fold cross-validation (M=$M observations per fold)."
    fi
    N_EPOCHS=1000
    N_BURNIN_EPOCHS=100
    F_PATIENT_EPOCHS=0.01
    F_VALIDATION=0.01
    N_BATCHES=1
    N_HIDDEN_LAYERS=1
    N_HIDDEN_NODES=700
    DROPOUT_RATE=0.0
    LEARNING_RATE=1e-5
    COST="MSE"
    OPTIMISERS="Adam,GradientDescent"
    ACTIVATIONS="ReLU,Linear"
    WEIGHTS_INITIALISATIONS="He,Cauchy"
    BNAME_INPUT=$(basename $FNAME_INPUT)
    BNAME_OUTPUT=$(echo $BNAME_INPUT | sed 's/input_simulated/output_simulated/g' | sed "s/.tsv$/-MLP_GP_CV.json/g")
    if [[ $(echo $BNAME_OUTPUT | grep "output_simulated") == "" ]]; then
        BNAME_OUTPUT="output_empirical-${BNAME_OUTPUT}"
    fi
    FNAME_OUTPUT_JSON=$(pwd)/$BNAME_OUTPUT
    FNAME_OUTPUT_CV=${FNAME_OUTPUT_JSON%.*}-cv.tsv
    TMP_OUTDIR="$(pwd)/tmp_dir-${BNAME_OUTPUT%.*}"
    mkdir $TMP_OUTDIR
    echo "INPUT: $FNAME_INPUT"
    echo "OUTPUT: $FNAME_OUTPUT_CV"
    echo "TMP_OUTDIR: $TMP_OUTDIR"
    echo "N_SAMPLES: $N"
    echo "N_FEATURES: $P"
    echo "N_REPS: $N_REPS"
    echo "N_FOLDS: $N_FOLDS"
    echo "N_SAMPLES_PER_FOLD: $M"
    # Run replicated k-fold cross-validation
    echo -e "datasets\treps\tfolds\tnt\tnv\tmodels\tcorr\tr2" > ${FNAME_OUTPUT_CV}.tmp
    for REP in $(seq 1 $N_REPS); do
        # REP=1
        SEED=$(echo "$BASE_SEED + $REP" | bc)
        IDX_SHUFFLED=($(shuf --random-source=<(yes $SEED) -e $(seq 2 $(echo $N + 1 | bc))))
        # echo "IDX_SHUFFLED: ${IDX_SHUFFLED[@]}"
        for FOLD in $(seq 1 $N_FOLDS); do
            # FOLD=1
            IDX_INI=$(echo "(($FOLD - 1) * $M) + 1" | bc)
            IDX_FIN=$(echo "$FOLD * $M" | bc)
            IDX_TRAINING=()
            IDX_VALIDATION=()
            for i in $(seq 0 $N); do
                if [[ ($i -ge $IDX_INI) && ($i -le $IDX_FIN) ]]; then
                    # echo "$i; ${IDX_SHUFFLED[i]}"
                    IDX_VALIDATION+=("${IDX_SHUFFLED[i]}")
                else
                    IDX_TRAINING+=("${IDX_SHUFFLED[i]}")
                fi
            done
            # echo "IDX_TRAINING: ${IDX_TRAINING[@]}"
            # echo "IDX_VALIDATION: ${IDX_VALIDATION[@]}"
            TRAINING_IDX=${IDX_TRAINING[@]}
            VALIDATION_IDX=${IDX_VALIDATION[@]}
            head -n1 $FNAME_INPUT > ${TMP_OUTDIR}/TRAINING_SET.tmp
            head -n1 $FNAME_INPUT > ${TMP_OUTDIR}/VALIDATION_SET.tmp
            awk -v idx="$TRAINING_IDX" 'BEGIN {FS="\t"; OFS="\t"} { split(idx, a, " "); for (i in a) b[a[i]] } NR in b' $FNAME_INPUT >> ${TMP_OUTDIR}/TRAINING_SET.tmp
            awk -v idx="$VALIDATION_IDX" 'BEGIN {FS="\t"; OFS="\t"} { split(idx, a, " "); for (i in a) b[a[i]] } NR in b' $FNAME_INPUT >> ${TMP_OUTDIR}/VALIDATION_SET.tmp
            # Fit with optimisation for optimiser, activation function, and weights initialisation hyperparameters
            time ${MLP} \
                -f ${TMP_OUTDIR}/TRAINING_SET.tmp \
                -o ${TMP_OUTDIR}/OUTPUT.tmp.json \
                -v \
                --hyperparameter-optimisation \
                --range-hidden-layers="${N_HIDDEN_LAYERS},${N_HIDDEN_LAYERS},${N_HIDDEN_LAYERS}" \
                --range-hidden-layer-nodes="${N_HIDDEN_NODES},${N_HIDDEN_NODES},${N_HIDDEN_NODES}" \
                --range-dropout-rates="${DROPOUT_RATE},${DROPOUT_RATE},0.01" \
                --range-learning-rates="${LEARNING_RATE},${LEARNING_RATE},${LEARNING_RATE}" \
                --range-n-epochs="${N_EPOCHS},${N_EPOCHS},${N_EPOCHS}" \
                --range-n-burnin-epochs="${N_BURNIN_EPOCHS},${N_BURNIN_EPOCHS},${N_BURNIN_EPOCHS}" \
                --range-f-patient-epochs="${F_PATIENT_EPOCHS},${F_PATIENT_EPOCHS},${F_PATIENT_EPOCHS}" \
                --range-f-validation="${F_VALIDATION},${F_VALIDATION},${F_VALIDATION}" \
                --range-n-batches="${N_BATCHES},${N_BATCHES},${N_BATCHES}" \
                --selection-costs="${COST}" \
                --selection-optimisers="${OPTIMISERS}" \
                --selection-activations="${ACTIVATIONS}" \
                --selection-weights-initialisations="${WEIGHTS_INITIALISATIONS}" \
                --skip-marginals > ${FNAME_OUTPUT_CV}.log
            # Clean-up just a bit
            for f in $(grep "Find the loss curve saved as: " ${FNAME_OUTPUT_CV}.log | cut -d ':' -f2  | cut -d ' ' -f2); do rm $f; done
            for f in $(grep "Find the observed vs predicted scatterplot saved as: " ${FNAME_OUTPUT_CV}.log | cut -d ':' -f2  | cut -d ' ' -f2); do rm $f; done
            # Predict
            time ${MLP} \
                -f ${TMP_OUTDIR}/VALIDATION_SET.tmp \
                -m ${TMP_OUTDIR}/OUTPUT.tmp.json \
                -v \
                --predict-only
            cut -f1 ${TMP_OUTDIR}/VALIDATION_SET.tmp > ${TMP_OUTDIR}/true.tmp
            cut -f1 ${TMP_OUTDIR}/OUTPUT-predictions.tmp > ${TMP_OUTDIR}/pred.tmp
            paste -d'\t' ${TMP_OUTDIR}/true.tmp ${TMP_OUTDIR}/pred.tmp > ${TMP_OUTDIR}/true_vs_pred.tmp
            NT=$(tail -n+2 ${TMP_OUTDIR}/TRAINING_SET.tmp | wc -l)
            NV=$(tail -n+2 ${TMP_OUTDIR}/VALIDATION_SET.tmp | wc -l)
            U_TRUE=$(tail -n+2 ${TMP_OUTDIR}/true_vs_pred.tmp | awk '{sum+=$1; count++} END {printf("%.21f\n", sum/count)}')
            U_PRED=$(tail -n+2 ${TMP_OUTDIR}/true_vs_pred.tmp | awk '{sum+=$2; count++} END {printf("%.21f\n", sum/count)}')
            S_TRUE=$(tail -n+2 ${TMP_OUTDIR}/true_vs_pred.tmp | awk -v U_TRUE="$U_TRUE" '{sum+=(($1-U_TRUE)^2); count++} END {printf("%.21f\n", sqrt(sum/(count-1)))}')
            S_PRED=$(tail -n+2 ${TMP_OUTDIR}/true_vs_pred.tmp | awk -v U_PRED="$U_PRED" '{sum+=(($2-U_PRED)^2); count++} END {printf("%.21f\n", sqrt(sum/(count-1)))}')
            V_TRUE_PRED=$(tail -n+2 ${TMP_OUTDIR}/true_vs_pred.tmp | awk -v U_TRUE="$U_TRUE" -v U_PRED="$U_PRED" '{sum+=(($1-U_TRUE)*($2-U_PRED)); count++} END {printf("%.21f\n", sum/(count-1))}')
            MSE=$(tail -n+2 ${TMP_OUTDIR}/true_vs_pred.tmp | awk '{sum+=(($1 - $2)^2); count++} END {printf("%.21f\n", sum/(count - 1))}')
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
            echo -e "$(basename $FNAME_INPUT)\t$REP\t$FOLD\t$NT\t$NV\tmlp\t$CORR\t$R2" >> ${FNAME_OUTPUT_CV}.tmp
        done
    done
    mv ${FNAME_OUTPUT_CV}.tmp $FNAME_OUTPUT_CV
    rm ${FNAME_OUTPUT_CV}.log
    rm -R $TMP_OUTDIR
fi

# ### Test
# ### Trials
# MLP=${HOME}/Documents/mlp/target/release/mlp
# ANALYSIS_TYPE=trials
# FNAME_INPUT=${HOME}/Documents/mlp/tests/datasets/agridat/australia.soybean-yield.tsv
# sh ./mlp.sh $MLP $ANALYSIS_TYPE $FNAME_INPUT
# #### GP
# MLP=${HOME}/Documents/mlp/target/release/mlp
# ANALYSIS_TYPE=gp
# DATA_TYPE=CONTINUOUS
# N=100
# P=50
# HIDDEN_LAYERS=2
# sh ./simulate.sh $MLP $ANALYSIS_TYPE $DATA_TYPE $N $P $HIDDEN_LAYERS

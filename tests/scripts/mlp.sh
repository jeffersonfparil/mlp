#!/bin/bash
if [[ $1 == "-h" || $1 == "--help" ]]; then
    echo "Usage: sh mlp.sh MLP_PATH ANALYSIS_TYPE FNAME_INPUT DIRNAME_OUTPUT [GP ARGS...]"
    echo "1. MLP: path to the mlp executable (https://github.com/jeffersonfparil/mlp)."
    echo "2. ANALYSIS_TYPE: 'trials' or 'gp'."
    echo "3. FNAME_INPUT: path to the input file for the analysis:"
    echo "4. DIRNAME_OUTPUT: directory where output files will be saved."
    echo -e "\t- For trials analysis (i.e. to extract the marginal effects of each genotype), this should be a tab-separated file with a header row and columns for year, site, treatment, entry, replication, and response variable."
    echo -e "\t- For genomic prediction analysis (i.e. repeated k-fold cross-validation), this should be a tab-separated file with a header row and columns for the response variable followed by the features."
    echo "GP ARGS: additional arguments for genomic prediction analysis. These are: "
    echo -e "\t5. FNAME_RANDOMISATION: path to the file containing randomisation indices"
    echo -e "\t6. N_REPS: number of replications of k-fold cross-validation"
    echo -e "\t7. N_FOLDS: number of folds for k-fold cross-validation"
    # echo -e "\t8. N_EPOCHS: number of epochs for training the multi-layer perceptron model"
    # echo -e "\t9. N_BURNIN_EPOCHS: number of burn-in epochs for training the multi-layer perceptron model"
    echo "Examples:"
    echo "MLP=\${HOME}/Documents/mlp/target/release/mlp"
    echo "mkdir tmp"
    echo "Rscript empiricalprep.R trials ${DIR}/datasets/agridat/australia.soybean.txt tmp"
    echo "Rscript empiricalprep.R gp ${DIR}/datasets/azodi_2019/sorghum_geno.csv tmp"
    echo "bash simulate.sh \$MLP gp tmp BINARY 100 50 2"
    echo "bash mlp.sh \$MLP trials tmp/australia.soybean-yield.tsv tmp"
    echo "bash mlp.sh \$MLP gp tmp/simulated-DATA_TYPE_BINARY-N_500-P_1000-HIDDEN_LAYERS_1.tsv tmp/ tmp/output-simulated-DATA_TYPE_BINARY-N_500-P_1000-HIDDEN_LAYERS_1-RANOMISATION.tsv 3 5"
    echo "bash mlp.sh \$MLP gp tmp/sorghum-YLD.tsv tmp/ tmp/output-sorghum-YLD-RANDOMISATION.tsv 2 5"
    exit 0
fi
MLP=$1
ANALYSIS_TYPE=$2
FNAME_INPUT=$3
DIRNAME_OUTPUT=$4
if [[ -z $ANALYSIS_TYPE ]]; then echo "Error: Missing argument for analysis type (trials or gp)."; exit 1; fi
if [[ $ANALYSIS_TYPE != "trials" && $ANALYSIS_TYPE != "gp" ]]; then echo "Error: Invalid analysis type. Expected 'trials' or 'gp', got '${ANALYSIS_TYPE}'."; exit 1; fi
if [[ -z $MLP ]]; then echo "Error: Missing argument for MLP path."; exit 1; fi
if [[ ! -f $MLP ]]; then echo "Error: MLP executable not found at the specified path: '${MLP}'."; exit 1; fi
if [[ -z $FNAME_INPUT ]]; then echo "Error: Missing argument for input file name."; exit 1; fi
if [[ ! -f $FNAME_INPUT ]]; then echo "Error: Input file not found at the specified path: '${FNAME_INPUT}'."; exit 1; fi
if [[ -z $DIRNAME_OUTPUT ]]; then echo "Error: Missing argument for output directory."; exit 1; fi
if [[ ! -d $DIRNAME_OUTPUT ]]; then echo "Error: Output directory not found at the specified path: '${DIRNAME_OUTPUT}'."; exit 1; fi
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
    BNAME_OUTPUT=$(echo $BNAME_INPUT | sed "s/.tsv$/-MLP.json/g")
    BNAME_OUTPUT="output-${BNAME_OUTPUT}"
    FNAME_OUTPUT_JSON=${DIRNAME_OUTPUT}/$BNAME_OUTPUT
    FNAME_OUTPUT_MARGINALS_TMP=${FNAME_OUTPUT_JSON%.json*}-marginal_effects.tsv
    FNAME_OUTPUT_MARGINALS=$(echo ${FNAME_OUTPUT_MARGINALS_TMP} | sed "s/-marginal_effects//g")
    TMP_OUTDIR="${DIRNAME_OUTPUT}/tmp_dir-${BNAME_OUTPUT%.*}"
    echo "INPUT: $FNAME_INPUT"
    echo "OUTPUT_TMP: $FNAME_OUTPUT_MARGINALS_TMP"
    echo "OUTPUT: $FNAME_OUTPUT_MARGINALS"
    echo "TMP_OUTDIR: $TMP_OUTDIR"
    rm -R $TMP_OUTDIR $FNAME_OUTPUT_JSON $FNAME_OUTPUT_MARGINALS_TMP $FNAME_OUTPUT_MARGINALS 2> /dev/null
    time ${MLP} \
        -f ${FNAME_INPUT} \
        -o ${FNAME_OUTPUT_JSON} \
        -v \
        --n-epochs=${N_EPOCHS} \
        --f-patient-epochs=${F_PATIENT_EPOCHS} \
        --n-batches=${N_BATCHES} \
        --n-hidden-layers=${N_HIDDEN_LAYERS} \
        --n-hidden-nodes=${N_HIDDEN_NODES} \
        --marginals-order=${MARGINALS_ORDER} > ${FNAME_OUTPUT_JSON}.log
    # Clean-up just a bit
    mkdir -p $TMP_OUTDIR
    for f in $(grep "Find the loss curve saved as: " ${FNAME_OUTPUT_JSON}.log | cut -d ':' -f2  | cut -d ' ' -f2); do mv $f $TMP_OUTDIR; done
    for f in $(grep "Find the observed vs predicted scatterplot saved as: " ${FNAME_OUTPUT_JSON}.log | cut -d ':' -f2  | cut -d ' ' -f2); do mv $f $TMP_OUTDIR; done
    for f in $(grep "Find the marginal effects (perturbation estimates) barplot saved as: " ${FNAME_OUTPUT_JSON}.log | cut -d ':' -f2  | cut -d ' ' -f2); do mv $f $TMP_OUTDIR; done
    mv $FNAME_OUTPUT_JSON $TMP_OUTDIR
    mv $FNAME_OUTPUT_MARGINALS_TMP $FNAME_OUTPUT_MARGINALS
else
    echo "#########################################################################################################"
    echo "### Running multi-layer perceptron model for genomic prediction with repeated k-fold cross-validation ###"
    echo "#########################################################################################################"
    FNAME_RANDOMISATION=$5
    N_REPS=$6
    N_FOLDS=$7
    # MLP=${HOME}/Documents/mlp/target/release/mlp
    # ANALYSIS_TYPE=gp
    # FNAME_INPUT=${HOME}/Documents/mlp/tests/tmp/gp/simulated-DATA_TYPE_BINARY-N_500-P_1000-HIDDEN_LAYERS_1.tsv
    # DIRNAME_OUTPUT=${HOME}/Documents/mlp/tests/tmp/gp
    # FNAME_RANDOMISATION=${HOME}/Documents/mlp/tests/tmp/gp/output-simulated-DATA_TYPE_BINARY-N_500-P_1000-HIDDEN_LAYERS_1-RANDOMISATION.tsv
    # N_REPS=2
    # N_FOLDS=2
    if [[ -z $FNAME_RANDOMISATION ]]; then echo "Error: Missing argument for path to the file containing randomisation indices (FNAME_RANDOMISATION)."; exit 1; fi
    if [[ ! -f $FNAME_RANDOMISATION ]]; then echo "Error: Randomisation file not found at the specified path: '${FNAME_RANDOMISATION}'."; exit 1; fi
    if [[ -z $N_REPS ]]; then echo "Error: Missing argument for number of replications of k-fold cross-validation (N_REPS)."; exit 1; fi
    if [[ -z $N_FOLDS ]]; then echo "Error: Missing argument for number of folds for k-fold cross-validation (N_FOLDS)."; exit 1; fi
    N=$(echo $(cut -f1 $FNAME_INPUT | wc -l) - 1 | bc)
    M=$(echo "scale=0; $N / $N_FOLDS" | bc)
    P=$(head -n1 $FNAME_INPUT | cut -f2- | awk '{print NF}')
    # while [[ $M -lt 10 ]]; do
    #     N_FOLDS=$(echo "$N_FOLDS - 1" | bc)
    #     M=$(echo "scale=0; $N / $N_FOLDS" | bc)
    # done
    # if [[ $N_FOLDS -lt 2 ]]; then
    #     echo "Error: Not enough folds for k-fold cross-validation. Please reduce the number of folds or increase the number of observations."
    #     exit 1
    # else
    #     echo "Using $N_FOLDS folds for k-fold cross-validation (M=$M observations per fold)."
    # fi
    N_EPOCHS=100
    N_BURNIN_EPOCHS=10
    # F_PATIENT_EPOCHS=0.01
    F_PATIENT_EPOCHS=0.1
    # F_VALIDATION=1e-42
    F_VALIDATION=0.1
    N_BATCHES=1
    N_HIDDEN_LAYERS=1
    N_HIDDEN_NODES=700
    DROPOUT_RATE=0.0
    LEARNING_RATE=1e-5
    COST="MSE"
    OPTIMISERS="Adam,GradientDescent"
    # OPTIMISERS="GradientDescent"
    # OPTIMISERS="Adam"
    ACTIVATIONS="ReLU,Linear"
    # ACTIVATIONS="ReLU"
    # ACTIVATIONS="Linear"
    # WEIGHTS_INITIALISATIONS="He,Cauchy,Uniform"
    WEIGHTS_INITIALISATIONS="Cauchy,He"
    # WEIGHTS_INITIALISATIONS="He"
    BNAME_INPUT=$(basename $FNAME_INPUT)
    BNAME_OUTPUT=$(echo $BNAME_INPUT | sed "s/.tsv$/-MLP.tsv/g")
    BNAME_OUTPUT="output-${BNAME_OUTPUT}"
    FNAME_OUTPUT_CV=${DIRNAME_OUTPUT}/$BNAME_OUTPUT
    TMP_OUTDIR="${DIRNAME_OUTPUT}/tmp_dir-${BNAME_OUTPUT%.*}"
    rm -R $TMP_OUTDIR $FNAME_OUTPUT_JSON $FNAME_OUTPUT_CV 2> /dev/null
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
    echo -ne "datasets\treps\tfolds\tnt\tnv\tmodels\t" > ${FNAME_OUTPUT_CV}.tmp
    echo -ne "n_hidden_layers\tn_hidden_nodes\tdropout_rate\tlearning_rate\t" >> ${FNAME_OUTPUT_CV}.tmp
    echo -ne "n_epochs\tn_burnin_epochs\tn_patient_epochs\tn_validation\t" >> ${FNAME_OUTPUT_CV}.tmp
    echo -ne "n_batches\tactivation\tcost\toptimiser\tweights_initialisation\t" >> ${FNAME_OUTPUT_CV}.tmp
    echo -e "corr\tr2" >> ${FNAME_OUTPUT_CV}.tmp
    # head -n1 ${FNAME_OUTPUT_CV}.tmp
    IDX_RANDOMISATION=0
    for REP in $(seq 1 $N_REPS); do
        # REP=1
        # SEED=$(echo "$BASE_SEED + $REP" | bc)
        # IDX_SHUFFLED=($(shuf --random-source=<(yes $SEED) -e $(seq 2 $(echo $N + 1 | bc))))
        # echo "IDX_SHUFFLED: ${IDX_SHUFFLED[@]}"
        for FOLD in $(seq 1 $N_FOLDS); do
            # FOLD=1
            IDX_RANDOMISATION=$(echo "$IDX_RANDOMISATION + 2" | bc)
            # echo $IDX_RANDOMISATION
            # IDX_INI=$(echo "(($FOLD - 1) * $M) + 1" | bc)
            # IDX_FIN=$(echo "$FOLD * $M" | bc)
            # IDX_TRAINING=()
            # IDX_VALIDATION=()
            # for i in $(seq 0 $N); do
            #     if [[ ($i -ge $IDX_INI) && ($i -le $IDX_FIN) ]]; then
            #         # echo "$i; ${IDX_SHUFFLED[i]}"
            #         IDX_VALIDATION+=("${IDX_SHUFFLED[i]}")
            #     else
            #         IDX_TRAINING+=("${IDX_SHUFFLED[i]}")
            #     fi
            # done
            # # echo "IDX_TRAINING: ${IDX_TRAINING[@]}"
            # # echo "IDX_VALIDATION: ${IDX_VALIDATION[@]}"
            # TRAINING_IDX=${IDX_TRAINING[@]}
            # VALIDATION_IDX=${IDX_VALIDATION[@]}
            TRAINING_IDX=$(head -n$(echo "$IDX_RANDOMISATION - 1" | bc) $FNAME_RANDOMISATION | tail -n1)
            VALIDATION_IDX=$(head -n${IDX_RANDOMISATION} $FNAME_RANDOMISATION | tail -n1)
            head -n1 $FNAME_INPUT > ${TMP_OUTDIR}/TRAINING_SET.tmp
            head -n1 $FNAME_INPUT > ${TMP_OUTDIR}/VALIDATION_SET.tmp
            awk -v idx="$TRAINING_IDX" 'BEGIN {FS="\t"; OFS="\t"} { split(idx, a, ","); for (i in a) b[a[i]+1] } NR in b' $FNAME_INPUT >> ${TMP_OUTDIR}/TRAINING_SET.tmp # Note that we add 1 to the indexes because the indexes start with 1 corresponding to the first line of the data file after the header line
            awk -v idx="$VALIDATION_IDX" 'BEGIN {FS="\t"; OFS="\t"} { split(idx, a, ","); for (i in a) b[a[i]+1] } NR in b' $FNAME_INPUT >> ${TMP_OUTDIR}/VALIDATION_SET.tmp # Note that we add 1 to the indexes because the indexes start with 1 corresponding to the first line of the data file after the header line
            # Fit with hyperparameter optimisation for optimiser, and weights initialisation
            ${MLP} \
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
            SELECTED_N_HIDDEN_LAYERS=$(grep -A14 "Best hyperparameters found:" ${FNAME_OUTPUT_CV}.log | grep "\- Hidden Layers: " | awk -F': '  '{print $2}')
            SELECTED_N_HIDDEN_NODES=$(grep -A14 "Best hyperparameters found:" ${FNAME_OUTPUT_CV}.log | grep "\- Hidden Nodes: " | awk -F': '  '{print $2}')
            SELECTED_DROPOUT_RATE=$(grep -A14 "Best hyperparameters found:" ${FNAME_OUTPUT_CV}.log | grep "\- Dropout Rate: " | awk -F': '  '{print $2}')
            SELECTED_LEARNING_RATE=$(grep -A14 "Best hyperparameters found:" ${FNAME_OUTPUT_CV}.log | grep "\- Learning Rate: " | awk -F': '  '{print $2}')
            SELECTED_N_EPOCHS=$(grep -A14 "Best hyperparameters found:" ${FNAME_OUTPUT_CV}.log | grep "\- Epochs: " | awk -F': '  '{print $2}')
            SELECTED_N_BURNIN_EPOCHS=$(grep -A14 "Best hyperparameters found:" ${FNAME_OUTPUT_CV}.log | grep "\- Burnin Epochs: " | awk -F': '  '{print $2}')
            SELECTED_N_PATIENT_EPOCHS=$(grep -A14 "Best hyperparameters found:" ${FNAME_OUTPUT_CV}.log | grep "\- Patient Epochs: " | awk -F': '  '{print $2}')
            SELECTED_N_VALIDATION=$(grep -A14 "Best hyperparameters found:" ${FNAME_OUTPUT_CV}.log | grep "\- Validation Set: " | awk -F': '  '{print $2}')
            SELECTED_N_BATCHES=$(grep -A14 "Best hyperparameters found:" ${FNAME_OUTPUT_CV}.log | grep "\- Batches: " | awk -F': '  '{print $2}')
            SELECTED_ACTIVATION=$(grep -A14 "Best hyperparameters found:" ${FNAME_OUTPUT_CV}.log | grep "\- Activation: " | awk -F': '  '{print $2}')
            SELECTED_COST=$(grep -A14 "Best hyperparameters found:" ${FNAME_OUTPUT_CV}.log | grep "\- Cost: " | awk -F': '  '{print $2}')
            SELECTED_OPTIMISER=$(grep -A14 "Best hyperparameters found:" ${FNAME_OUTPUT_CV}.log | grep "\- Optimiser: " | awk -F': '  '{print $2}')
            SELECTED_WEIGHTS_INITIALISATION=$(grep -A14 "Best hyperparameters found:" ${FNAME_OUTPUT_CV}.log | grep "\- Weights Initialisation: " | awk -F': '  '{print $2}')
            # Clean-up just a bit
            for f in $(grep "Find the loss curve saved as: " ${FNAME_OUTPUT_CV}.log | cut -d ':' -f2  | cut -d ' ' -f2); do rm $f; done
            for f in $(grep "Find the observed vs predicted scatterplot saved as: " ${FNAME_OUTPUT_CV}.log | cut -d ':' -f2  | cut -d ' ' -f2); do rm $f; done
            # Predict
            ${MLP} \
                -f ${TMP_OUTDIR}/VALIDATION_SET.tmp \
                -m ${TMP_OUTDIR}/OUTPUT.tmp.json \
                -v \
                --predict-only >> ${FNAME_OUTPUT_CV}.log
            cut -f1 ${TMP_OUTDIR}/VALIDATION_SET.tmp > ${TMP_OUTDIR}/true.tmp
            cut -f1 ${TMP_OUTDIR}/OUTPUT.tmp-predictions.tsv > ${TMP_OUTDIR}/pred.tmp
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
            # echo "U_TRUE: $U_TRUE"
            # echo "U_PRED: $U_PRED"
            # echo "S_TRUE: $S_TRUE"
            # echo "S_PRED: $S_PRED"
            # echo "V_TRUE_PRED: $V_TRUE_PRED"
            # echo "MSE: $MSE"
            # echo "CORR: $CORR"
            # echo "R2: $R2"
            rm ${TMP_OUTDIR}/VALIDATION_SET.tmp
            rm ${TMP_OUTDIR}/OUTPUT.tmp.json
            rm ${TMP_OUTDIR}/OUTPUT.tmp-predictions.tsv
            # Update the output file
            # echo -ne "$SELECTED_ACTIVATION\t$SELECTED_OPTIMISER\t$SELECTED_WEIGHTS_INITIALISATION\t$CORR\t$R2" >> ${FNAME_OUTPUT_CV}.tmp
            echo -ne "$(basename $FNAME_INPUT)\t$REP\t$FOLD\t$NT\t$NV\tmlp\t" >> ${FNAME_OUTPUT_CV}.tmp
            echo -ne "$SELECTED_N_HIDDEN_LAYERS\t$SELECTED_N_HIDDEN_NODES\t$SELECTED_DROPOUT_RATE\t$SELECTED_LEARNING_RATE\t" >> ${FNAME_OUTPUT_CV}.tmp
            echo -ne "$SELECTED_N_EPOCHS\t$SELECTED_N_BURNIN_EPOCHS\t$SELECTED_N_PATIENT_EPOCHS\t$SELECTED_N_VALIDATION\t" >> ${FNAME_OUTPUT_CV}.tmp
            echo -ne "$SELECTED_N_BATCHES\t$SELECTED_ACTIVATION\t$SELECTED_COST\t$SELECTED_OPTIMISER\t$SELECTED_WEIGHTS_INITIALISATION\t" >> ${FNAME_OUTPUT_CV}.tmp
            echo -e "$CORR\t$R2" >> ${FNAME_OUTPUT_CV}.tmp
            tail -n1 ${FNAME_OUTPUT_CV}.tmp
        done
    done
    mv ${FNAME_OUTPUT_CV}.tmp $FNAME_OUTPUT_CV
    rm ${FNAME_OUTPUT_CV}.log
    rm -R $TMP_OUTDIR
fi

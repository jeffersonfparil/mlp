#!/bin/bash
if [[ $1 == "-h" || $1 == "--help" ]]; then
    echo "Usage: sh randomisationgprs.sh FNAME_INPUT DIRNAME_OUTPUT N_REPS N_FOLDS BASE_SEED"
    echo "Note that we start with index 1 for the first row of the data after the header line."
    echo "1. ANALYSIS_TYPE: 'trials' or 'gp' or 'remotesensing'."
    echo "2. FNAME_INPUT: path to the input file for the analysis:"
    echo "3. DIRNAME_OUTPUT: directory where output files will be saved."
    echo "4. N_REPS: number of replications of k-fold cross-validation."
    echo "5. N_FOLDS: number of folds for k-fold cross-validation."
    echo "6. BASE_SEED: base seed for random number generation."
    echo "Example: "
    echo "bash randomisationgprs.sh gp tmp/gp/simulated-DATA_TYPE_CONTINUOUS-N_500-P_1000-HIDDEN_LAYERS_1.tsv tmp/gp 3 10 42"
    exit 0
fi
ANALYSIS_TYPE=$1
FNAME_INPUT=$2
DIRNAME_OUTDIR=$3
N_REPS=$4
N_FOLDS=$5
BASE_SEED=$6

# ANALYSIS_TYPE="gp"
# FNAME_INPUT="tests/tmp/gp/simulated-DATA_TYPE_CONTINUOUS-N_500-P_1000-HIDDEN_LAYERS_1.tsv"
# DIRNAME_OUTDIR="tests/tmp/gp"
# N_REPS=3
# N_FOLDS=10
# BASE_SEED=42

if [[ -z $FNAME_INPUT ]]; then echo "Error: Missing argument for input file for the analysis (FNAME_INPUT)."; exit 1; fi
if [[ -z $DIRNAME_OUTDIR ]]; then echo "Error: Missing argument for output directory (DIRNAME_OUTDIR)."; exit 1; fi
if [[ -z $N_REPS ]]; then echo "Error: Missing argument for number of replications of k-fold cross-validation (N_REPS)."; exit 1; fi
if [[ -z $N_FOLDS ]]; then echo "Error: Missing argument for number of folds for k-fold cross-validation (N_FOLDS)."; exit 1; fi
if [[ -z $BASE_SEED ]]; then echo "Error: Missing argument for base seed for random number generation (BASE_SEED)."; exit 1; fi
if [[ $ANALYSIS_TYPE == "trials" ]]; then
    # Dummy reps and folds as we won't use randomisations for the trials analyses
    N_REPS=1
    N_FOLDS=2
fi
N=$(echo $(cut -f1 $FNAME_INPUT | wc -l) - 1 | bc)
M=$(echo "scale=0; $N / $N_FOLDS" | bc)
P=$(head -n1 $FNAME_INPUT | cut -f2- | awk '{print NF}')
if [[ $M -lt 5 ]]; then
    echo "Error: Not enough folds for k-fold cross-validation (less than 5). Please reduce the number of folds or increase the number of observations."
    exit 1
fi
OUTPUT_CSV=${DIRNAME_OUTDIR}/output-$(basename $FNAME_INPUT | sed 's/.tsv$//g')-RANDOMISATION.tsv
touch $OUTPUT_CSV
for REP in $(seq 1 $N_REPS); do
    # REP=1
    SEED=$(echo "$BASE_SEED + $REP" | bc)
    IDX_SHUFFLED=($(shuf --random-source=<(yes $SEED) -e $(seq 1 $N))) # Note that we start with index 1 for the first row of the data after the header line
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
        echo "IDX_TRAINING: ${IDX_TRAINING[@]}"
        echo "IDX_VALIDATION: ${IDX_VALIDATION[@]}"
        echo "${IDX_TRAINING[@]}" | sed -z 's/ /\n/g' | sort -n | sed -z 's/\n/,/g' | sed -z 's/^,//g' | sed -z 's/,$/\n/g' >> $OUTPUT_CSV
        echo "${IDX_VALIDATION[@]}" | sed -z 's/ /\n/g' | sort -n | sed -z 's/\n/,/g' | sed -z 's/^,//g' | sed -z 's/,$/\n/g' >> $OUTPUT_CSV
    done
done
#!/bin/bash
if [[ $1 == "-h" || $1 == "--help" ]]; then
    echo "Simulate trials or genomic prediction datasets using a multi-layer perceptron model."
    echo "Usage: sh simulate.sh MLP_PATH ANALYSIS_TYPE DIR_OUTPUT [ADDITIONAL_PARAMETERS]"
    echo "1. MLP: path to the mlp executable (https://github.com/jeffersonfparil/mlp)."
    echo "2. ANALYSIS_TYPE: 'trials' or 'gp'."
    echo "3. DIR_OUTPUT: directory where output files will be saved."
    echo "For 'trials', additional parameters are: "
    echo -e "\t4. N_YEARS: number of years"
    echo -e "\t5. N_SITES: number of sites"
    echo -e "\t6. N_TREATMENTS: number of treatments"
    echo -e "\t7. N_ENTRIES: number of entries"
    echo -e "\t8. N_REPLICATIONS: number of replications"
    echo -e "\t9. N_HIDDEN_LAYERS: number of hidden layers in the underlying model"
    echo "For 'gp', additional parameters are:"
    echo -e "\t4. DATA_TYPE: type of data to simulate (CONTINUOUS or BINARY)"
    echo -e "\t5. N: number of observations"
    echo -e "\t6. P: number of features"
    echo -e "\t7. HIDDEN_LAYERS: number of hidden layers in the underlying model"
    echo "Examples:"
    echo "MLP=\${HOME}/Documents/mlp/target/release/mlp"
    echo "mkdir tmp"
    echo "bash simulate.sh \$MLP trials tmp 5 3 2 10 2 3"
    echo "bash simulate.sh \$MLP gp tmp CONTINUOUS 100 50 2"
    echo "bash simulate.sh \$MLP gp tmp BINARY 100 50 2"
    exit 0
fi
MLP=$1
ANALYSIS_TYPE=$2
DIR_OUTPUT=$3
if [[ -z $MLP ]]; then echo "Error: Missing argument for MLP path."; exit 1; fi
if [[ ! -f $MLP ]]; then echo "Error: MLP executable not found at the specified path: '${MLP}'."; exit 1; fi
if [[ -z $ANALYSIS_TYPE ]]; then echo "Error: Missing argument for analysis type (trials or gp)."; exit 1; fi
if [[ -z $DIR_OUTPUT ]]; then echo "Error: Missing argument for output directory."; exit 1; fi

echo $MLP
echo $ANALYSIS_TYPE
echo $DIR_OUTPUT

if [[ $ANALYSIS_TYPE == "trials" ]]; then
    echo "###########################################"
    echo "### Simulating data for trials analysis ###"
    echo "###########################################"
    echo "Notes:"
    echo "- simulated file names will be of the format: simulated-YEARS_X-SITES_Y-TREATMENTS_Z-ENTRIES_W-REPLICATIONS_V-HIDDEN_LAYERS_HL.tsv"
    echo "- this means if you rerun the script with the same parameters, it will overwrite the previously simulated data file with the same name."
    N_YEARS=$4
    N_SITES=$5
    N_TREATMENTS=$6
    N_ENTRIES=$7
    N_REPLICATIONS=$8
    N_HIDDEN_LAYERS=$9
    if [[ -z $N_YEARS ]]; then echo "Error: Missing argument for number of years (N_YEARS)."; exit 1; fi
    if [[ -z $N_SITES ]]; then echo "Error: Missing argument for number of sites (N_SITES)."; exit 1; fi
    if [[ -z $N_TREATMENTS ]]; then echo "Error: Missing argument for number of treatments (N_TREATMENTS)."; exit 1; fi
    if [[ -z $N_ENTRIES ]]; then echo "Error: Missing argument for number of entries (N_ENTRIES)."; exit 1; fi
    if [[ -z $N_REPLICATIONS ]]; then echo "Error: Missing argument for number of replications (N_REPLICATIONS)."; exit 1; fi
    if [[ -z $N_HIDDEN_LAYERS ]]; then echo "Error: Missing argument for number of hidden layers (N_HIDDEN_LAYERS)."; exit 1; fi
    FNAME_OUTPUT=${DIR_OUTPUT}/simulated-YEARS_${N_YEARS}-SITES_${N_SITES}-TREATMENTS_${N_TREATMENTS}-ENTRIES_${N_ENTRIES}-REPLICATIONS_${N_REPLICATIONS}-HIDDEN_LAYERS_${N_HIDDEN_LAYERS}.tsv
    SEED=$(echo "$N_HIDDEN_LAYERS*($N_YEARS+$N_SITES+$N_TREATMENTS+$N_ENTRIES+$N_REPLICATIONS)" | bc)
    $MLP \
        --simulate-data-only \
        --simulation-fname-output ${FNAME_OUTPUT}.tmp \
        --simulation-n-observations $(echo "$N_YEARS*$N_SITES*$N_TREATMENTS*$N_ENTRIES*$N_REPLICATIONS" | bc) \
        --simulation-n-features-continuous 0 \
        --simulation-n-features-categorical "$N_YEARS,$N_SITES,$N_TREATMENTS,$N_ENTRIES,$N_REPLICATIONS" \
        --simulation-n-output-columns 1 \
        --simulation-n-hidden-layers ${N_HIDDEN_LAYERS} \
        --simulation-weights-distribution normal \
        --simulation-weights-distribution-param-1 0 \
        --simulation-weights-distribution-param-2 1 \
        --seed $SEED \
        --verbose
    sed 's/target_0/y/g' ${FNAME_OUTPUT}.tmp | sed 's/fcat_0/year/g' | sed 's/fcat_1/loc/g' | sed 's/fcat_2/trt/g' | sed 's/fcat_3/gen/g' | sed 's/fcat_4/blk/g' | \
        awk '{FS="\t"; OFS="\t"} {gsub(/level/,"year➵level",$2); print }' | \
        awk '{FS="\t"; OFS="\t"} {gsub(/level/,"loc➵level",$3); print }' | \
        awk '{FS="\t"; OFS="\t"} {gsub(/level/,"trt➵level",$4); print }' | \
        awk '{FS="\t"; OFS="\t"} {gsub(/level/,"gen➵level",$5); print }' | \
        awk '{FS="\t"; OFS="\t"} {gsub(/level/,"blk➵level",$6); print }' > ${FNAME_OUTPUT}.tmp2
    rm ${FNAME_OUTPUT}.tmp
    mv ${FNAME_OUTPUT}.tmp2 ${FNAME_OUTPUT}
    echo "OUTPUT: '${FNAME_OUTPUT}'"
elif [[ $ANALYSIS_TYPE == "gp" ]]; then
    echo "#######################################################"
    echo "### Simulating data for genomic prediction analysis ###"
    echo "#######################################################"
    echo "Notes:"
    echo "- simulated file names will be of the format: simulated-DATA_TYPE_X-N_Y-P_HIDDEN_LAYERS_HL.tsv"
    echo "- this means if you rerun the script with the same parameters, it will overwrite the previously simulated data file with the same name."
    DATA_TYPE=$4
    N=$5
    P=$6
    HIDDEN_LAYERS=$7
    if [[ -z $DATA_TYPE ]]; then echo "Error: Missing argument for data type (DATA_TYPE, i.e. \"CONTINUOUS\" or \"BINARY\")."; exit 1; fi
    if [[ -z $N ]]; then echo "Error: Missing argument for number of observations (N)."; exit 1; fi
    if [[ -z $P ]]; then echo "Error: Missing argument for number of features (P)."; exit 1; fi
    if [[ -z $HIDDEN_LAYERS ]]; then echo "Error: Missing argument for number of hidden layers (HIDDEN_LAYERS)."; exit 1; fi
    if [[ $DATA_TYPE != "CONTINUOUS" && $DATA_TYPE != "BINARY" ]]; then echo "Error: Invalid data type. Please specify 'CONTINUOUS' or 'BINARY' for DATA_TYPE."; exit 1; fi
    FNAME_OUTPUT=${DIR_OUTPUT}/simulated-DATA_TYPE_${DATA_TYPE}-N_${N}-P_${P}-HIDDEN_LAYERS_${HIDDEN_LAYERS}.tsv
    SEED=$(echo "$HIDDEN_LAYERS*($N+$P)" | bc)
    $MLP \
        --simulate-data-only \
        --simulation-fname-output ${FNAME_OUTPUT}.tmp \
        --simulation-n-observations $N \
        --simulation-n-features-continuous $P \
        --simulation-n-features-categorical 0 \
        --simulation-n-output-columns 1 \
        --simulation-n-hidden-layers ${HIDDEN_LAYERS} \
        --simulation-features-distribution normal \
        --simulation-features-distribution-param-1 0 \
        --simulation-features-distribution-param-2 1 \
        --simulation-weights-distribution normal \
        --simulation-weights-distribution-param-1 0 \
        --simulation-weights-distribution-param-2 1 \
        --seed $SEED \
        --verbose
    if [[ $DATA_TYPE == "CONTINUOUS" ]]; then
        mv ${FNAME_OUTPUT}.tmp ${FNAME_OUTPUT}
    else [[ $DATA_TYPE == "BINARY" ]]
        head -n1 ${FNAME_OUTPUT}.tmp > ${FNAME_OUTPUT}.tmp2
        tail -n+2 ${FNAME_OUTPUT}.tmp | 
        awk '{
            FS="\t"; OFS="\t"; 
            for (i=2; i<=NF; i++) {
                $i = sprintf("%.0f", $i)
            }
        }{print}' - >> ${FNAME_OUTPUT}.tmp2
        rm ${FNAME_OUTPUT}.tmp
        mv ${FNAME_OUTPUT}.tmp2 $FNAME_OUTPUT
    fi
    echo "OUTPUT: '${FNAME_OUTPUT}'"
else
    echo "Invalid analysis type. Please specify 'trials' or 'gp'."
    exit 1
fi

#!/bin/bash
MLP=$1
ANALYSIS_TYPE=$2
if [[ $ANALYSIS_TYPE == "trials" ]]; then
    echo "Simulating data for trials analysis..."
    N_YEARS=$3
    N_SITES=$4
    N_TREATMENTS=$5
    N_ENTRIES=$6
    N_REPLICATIONS=$7
    N_HIDDEN_LAYERS=$8
    FNAME_OUTPUT=input_simulated-YEARS_${N_YEARS}-SITES_${N_SITES}-TREATMENTS_${N_TREATMENTS}-ENTRIES_${N_ENTRIES}-REPLICATIONS_${N_REPLICATIONS}-HIDDEN_LAYERS_${N_HIDDEN_LAYERS}.tsv
    echo "######################"
    echo $FNAME_OUTPUT
    $MLP \
        --simulate-data-only \
        --simulation-n-observations $(echo "$N_YEARS*$N_SITES*$N_TREATMENTS*$N_ENTRIES*$N_REPLICATIONS" | bc) \
        --simulation-n-features-continuous 0 \
        --simulation-n-features-categorical "$N_YEARS,$N_SITES,$N_TREATMENTS,$N_ENTRIES,$N_REPLICATIONS" \
        --simulation-n-output-columns 1 \
        --simulation-n-hidden-layers ${N_HIDDEN_LAYERS} \
        --simulation-weights-distribution normal \
        --simulation-weights-distribution-param-1 0 \
        --simulation-weights-distribution-param-2 1 \
        --seed $(echo "$N_HIDDEN_LAYERS*($N_YEARS+$N_SITES+$N_TREATMENTS+$N_ENTRIES+$N_REPLICATIONS)" | bc) > log.tmp
    F=$(grep "Please find simulated data:" log.tmp | cut -d '`' -f2)
    sed 's/target_0/y/g' $F | sed 's/fcat_0/year/g' | sed 's/fcat_1/loc/g' | sed 's/fcat_2/trt/g' | sed 's/fcat_3/gen/g' | sed 's/fcat_4/blk/g' | \
        awk '{FS="\t"; OFS="\t"} {gsub(/level/,"year➵level",$2); print }' | \
        awk '{FS="\t"; OFS="\t"} {gsub(/level/,"loc➵level",$3); print }' | \
        awk '{FS="\t"; OFS="\t"} {gsub(/level/,"trt➵level",$4); print }' | \
        awk '{FS="\t"; OFS="\t"} {gsub(/level/,"gen➵level",$5); print }' | \
        awk '{FS="\t"; OFS="\t"} {gsub(/level/,"blk➵level",$6); print }' > file.tmp
    mv file.tmp $FNAME_OUTPUT
    rm $F *.tmp
elif [[ $ANALYSIS_TYPE == "gp" ]]; then
    echo "Simulating data for genomic prediction analysis..."
    N=$3
    P=$4
    HIDDEN_LAYERS=$5
    F_CONTINUOUS=input_simulated-CONTINUOUS-${HIDDEN_LAYERS}HL.tsv
    F_BINARY=input_simulated-BINARY-${HIDDEN_LAYERS}HL.tsv
    echo "######################"
    echo "$F_CONTINUOUS and $F_BINARY"
    $MLP \
        --simulate-data-only \
        --simulation-n-observations $N \
        --simulation-n-features-continuous $P \
        --simulation-n-features-categorical 0 \
        --simulation-n-output-columns 1 \
        --simulation-n-hidden-layers ${HIDDEN_LAYERS} \
        --simulation-weights-distribution normal \
        --simulation-weights-distribution-param-1 0 \
        --simulation-weights-distribution-param-2 1 \
        --seed 42 \
        --verbose
    F0=$(ls -lhtr input_simulated-*.tsv | tail -n1 |  rev | awk '{print $1}' | rev)
    mv $F0 $F_CONTINUOUS
    # Convert into binary genotype data
    head -n1 $F_CONTINUOUS > $F_BINARY
    tail -n+2 $F_CONTINUOUS | 
      awk '{
          FS="\t"; OFS="\t"; 
          for (i=2; i<=NF; i++) {
              $i = sprintf("%.0f", $i)
          }
      }{print}' - >> $F_BINARY
else
    echo "Invalid analysis type. Please specify 'trials' or 'gp'."
    exit 1
fi

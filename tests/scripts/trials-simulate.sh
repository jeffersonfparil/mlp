#!/bin/bash
MLP=../../../target/release/mlp
N_YEARS_SMALL=2
N_YEARS_LARGE=7
N_SITES_SMALL=2
N_SITES_LARGE=20
N_TREATMENTS_SMALL=2
N_TREATMENTS_LARGE=5
N_ENTRIES_SMALL=25
N_ENTRIES_LARGE=100
N_REPLICATIONS=3
for HIDDEN_LAYERS in $(seq 1 5)
do
    # HIDDEN_LAYERS=1
    F_SMALL=input_simulated-SMALL-${HIDDEN_LAYERS}HL.tsv
    F_LARGE=input_simulated-LARGE-${HIDDEN_LAYERS}HL.tsv
    echo "######################"
    echo "$F_SMALL and $F_LARGE"
    $MLP \
        --simulate-data-only \
        --simulation-n-observations $(echo "$N_YEARS_SMALL*$N_SITES_SMALL*$N_TREATMENTS_SMALL*$N_ENTRIES_SMALL*$N_REPLICATIONS" | bc) \
        --simulation-n-features-continuous 0 \
        --simulation-n-features-categorical "$N_YEARS_SMALL,$N_SITES_SMALL,$N_TREATMENTS_SMALL,$N_ENTRIES_SMALL,$N_REPLICATIONS" \
        --simulation-n-output-columns 1 \
        --simulation-n-hidden-layers ${HIDDEN_LAYERS} \
        --simulation-weights-distribution normal \
        --simulation-weights-distribution-param-1 0 \
        --simulation-weights-distribution-param-2 1 \
        --seed ${HIDDEN_LAYERS}
    F=$(ls -lhtr input_simulated-*.tsv | tail -n1 |  rev | awk '{print $1}' | rev)
    sed 's/target_0/y/g' $F | sed 's/fcat_0/year/g' | sed 's/fcat_1/loc/g' | sed 's/fcat_2/trt/g' | sed 's/fcat_3/gen/g' | sed 's/fcat_4/blk/g' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"year➵level",$2); print }' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"loc➵level",$3); print }' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"trt➵level",$4); print }' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"gen➵level",$5); print }' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"blk➵level",$6); print }' > tmp
    mv tmp $F
    mv $F $F_SMALL
    $MLP \
        --simulate-data-only \
        --simulation-n-observations $(echo "$N_YEARS_LARGE*$N_SITES_LARGE*$N_TREATMENTS_SMALL*$N_ENTRIES_SMALL*$N_REPLICATIONS" | bc) \
        --simulation-n-features-continuous 0 \
        --simulation-n-features-categorical "$N_YEARS_LARGE,$N_SITES_LARGE,$N_TREATMENTS_SMALL,$N_ENTRIES_SMALL,$N_REPLICATIONS" \
        --simulation-n-output-columns 1 \
        --simulation-n-hidden-layers ${HIDDEN_LAYERS} \
        --simulation-weights-distribution normal \
        --simulation-weights-distribution-param-1 0 \
        --simulation-weights-distribution-param-2 1 \
        --seed ${HIDDEN_LAYERS}
    F1=$(ls -lhtr input_simulated-*.tsv | tail -n1 |  rev | awk '{print $1}' | rev)
    sed 's/target_0/y/g' $F1 | sed 's/fcat_0/year/g' | sed 's/fcat_1/loc/g' | sed 's/fcat_2/trt/g' | sed 's/fcat_3/gen/g' | sed 's/fcat_4/blk/g' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"year➵level",$2); print }' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"loc➵level",$3); print }' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"trt➵level",$4); print }' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"gen➵level",$5); print }' | \
    awk '{FS="\t"; OFS="\t"} {gsub(/level/,"blk➵level",$6); print }' > tmp
    mv tmp $F1
    mv $F1 $F_LARGE
done
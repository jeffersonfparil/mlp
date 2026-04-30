#!/bin/bash

# cd mlp/
# cd tests/simulated
mkdir mlp_misc_output
MLP=../../target/release/mlp
N_EPOCHS=500
F_PATIENT_EPOCHS=0.01
N_BATCHES=2
N_HIDDEN_LAYERS=1
N_HIDDEN_NODES=64
MARGINALS_ORDER=1
for INPUT in $(ls input_simulated-*-*.tsv)
do
    # INPUT=$(ls input_simulated-*-*.tsv | head -n2 | tail -n1)
    # INPUT=input_simulated-NORMAL-1HL.tsv
    echo $INPUT
    # N_EPOCHS=$(echo "500 * ($N_HIDDEN_LAYERS / 2)" | bc)
    # N_HIDDEN_LAYERS=$(echo $(echo ${INPUT%.tsv*} | rev | cut -d'-' -f1 | rev | sed 's/HL//g') + 1 | bc)
    OUTPUT=$(echo $INPUT | sed 's/input_simulated/output_simulated/g' | sed "s/.tsv/-MLP_E${N_EPOCHS}_F${F_PATIENT_EPOCHS}_B${N_BATCHES}_H${N_HIDDEN_LAYERS}_M${MARGINALS_ORDER}.json/g")
    echo $OUTPUT
    time ${MLP} -f ${INPUT} -o ${OUTPUT} -v --n-epochs=${N_EPOCHS} --f-patient-epochs=${F_PATIENT_EPOCHS} --n-batches=${N_BATCHES} --n-hidden-layers=${N_HIDDEN_LAYERS} --n-hidden-nodes=${N_HIDDEN_NODES} --marginals-order=${MARGINALS_ORDER}
    TMP_OUTDIR=mlp_misc_output/${OUTPUT%.*}
    mkdir $TMP_OUTDIR
    mv $OUTPUT mlp_misc_output/${OUTPUT%.*}
    mv *.svg mlp_misc_output/${OUTPUT%.*}
    mv *.png mlp_misc_output/${OUTPUT%.*}
done
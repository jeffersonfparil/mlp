#!/bin/bash
mkdir mlp_misc_output
MLP=../../../target/release/mlp

if [[ $(basename $(pwd)) == "simulated" ]]
then
    echo "#####################"
    echo "MLP ON SIMULATED DATA"
    echo "#####################"
    N_EPOCHS=500
    F_PATIENT_EPOCHS=0.01
    N_BATCHES=1
    N_HIDDEN_LAYERS=1
    N_HIDDEN_NODES=64
    MARGINALS_ORDER=1
    for INPUT in $(ls input_simulated-*-*.tsv)
    do
        # INPUT=$(ls input_simulated-*-*.tsv | head -n2 | tail -n1)
        # INPUT=input_simulated-SMALL-1HL.tsv
        OUTPUT=$(echo $INPUT | sed 's/input_simulated/output_simulated/g' | sed "s/.tsv/-MLP_E${N_EPOCHS}_F${F_PATIENT_EPOCHS}_B${N_BATCHES}_H${N_HIDDEN_LAYERS}_M${MARGINALS_ORDER}.json/g")
        echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        echo "$INPUT --> $OUTPUT"
        echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        time ${MLP} \
            -f ${INPUT} \
            -o ${OUTPUT} \
            -v \
            --n-epochs=${N_EPOCHS} \
            --f-patient-epochs=${F_PATIENT_EPOCHS} \
            --n-batches=${N_BATCHES} \
            --n-hidden-layers=${N_HIDDEN_LAYERS} \
            --n-hidden-nodes=${N_HIDDEN_NODES} \
            --marginals-order=${MARGINALS_ORDER}
        TMP_OUTDIR=mlp_misc_output/${OUTPUT%.*}
        mkdir $TMP_OUTDIR
        mv $OUTPUT $TMP_OUTDIR
        mv *.svg $TMP_OUTDIR
        mv *.png $TMP_OUTDIR
    done
else 
    echo "#####################"
    echo "MLP ON EMPIRICAL DATA"
    echo "#####################"
    N_EPOCHS=1000 # we need more training epochs to fit the mess that is reality
    F_PATIENT_EPOCHS=0.01
    N_BATCHES=1
    N_HIDDEN_LAYERS=1
    N_HIDDEN_NODES=64
    MARGINALS_ORDER=1
    for INPUT in $(ls *.tsv | grep -v "^output")
    do
        # INPUT=$(ls *.tsv | grep -v "^output" | head -n1 | tail -n1); echo $INPUT
        OUTPUT=$(echo output-${INPUT} | sed "s/.tsv/-MLP_E${N_EPOCHS}_F${F_PATIENT_EPOCHS}_B${N_BATCHES}_H${N_HIDDEN_LAYERS}_M${MARGINALS_ORDER}.json/g")
        echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        echo "$INPUT --> $OUTPUT"
        echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
        time ${MLP} \
            -f ${INPUT} \
            -o ${OUTPUT} \
            -v \
            --n-epochs=${N_EPOCHS} \
            --f-patient-epochs=${F_PATIENT_EPOCHS} \
            --n-batches=${N_BATCHES} \
            --n-hidden-layers=${N_HIDDEN_LAYERS} \
            --n-hidden-nodes=${N_HIDDEN_NODES} \
            --marginals-order=${MARGINALS_ORDER}
        TMP_OUTDIR=mlp_misc_output/${OUTPUT%.*}
        mkdir $TMP_OUTDIR
        mv $OUTPUT $TMP_OUTDIR
        mv Loss_curve-*.svg $TMP_OUTDIR
        mv Observed_vs_predicted-*.svg $TMP_OUTDIR
        mv Marginal_effects-*.png $TMP_OUTDIR
    done
fi


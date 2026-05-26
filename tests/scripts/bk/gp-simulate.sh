#!/bin/bash
MLP=../../../target/release/mlp
N=700
P=42000
for HIDDEN_LAYERS in $(seq 1 3) # cannot have more hidden layers because of GPU memory limitations (H100s and V100s)
do
    # HIDDEN_LAYERS=3
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
done
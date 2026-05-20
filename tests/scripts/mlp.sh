

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
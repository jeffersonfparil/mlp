# mlp

Simple multilayer perceptron (MLP) from scratch

|**Build Status**|**License**|
|:--------------:|:---------:|
| <a href="https://github.com/jeffersonfparil/mlp/actions"><img src="https://github.com/jeffersonfparil/mlp/actions/workflows/rust.yaml/badge.svg"></a> | [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0) |

## Setup

1. Install [pixi](https://pixi.prefix.dev/):

```shell
wget -qO- https://pixi.sh/install.sh | sh
```

2. Setup the workspace:

```shell
git clone https://github.com/jeffersonfparil/mlp.git
cd mlp
pixi init
```

3. Add cargo, cuda-nvrtc:

```shell
cd mlp
pixi shell
pixi add rust
pixi add cuda-nvrtc==12.8.93
which cargo
ls -lhtr ${PIXI_PROJECT_ROOT}/.pixi/envs/default/lib/libnvrtc*
```

## Unit testing

```shell
cd mlp
pixi shell
# export LD_LIBRARY_PATH=${PIXI_PROJECT_ROOT}/.pixi/envs/default/lib
time cargo test -- --show-output
```

## More testing

```shell
cd mlp
pixi shell
# export LD_LIBRARY_PATH=${PIXI_PROJECT_ROOT}/.pixi/envs/default/lib
time cargo run -- -h
time cargo run -- -s --verbose
INPUT=$(ls -t1 | grep "input.*.tsv" | head -n1)
head $INPUT
time cargo run -- -f $INPUT -v --n-batches=2 --n-epochs=10
MODEL=$(ls -t1 | grep "output.*.json" | head -n1)
MARGINALS=$(ls -t1 | grep "output.*-marginal_effects.tsv" | head -n1)
head $MODEL
head $MARGINALS
time cargo run -- -f $INPUT -v -m $MODEL --predict
PREDICTED=$(ls -t1 | grep "output.*-predictions.tsv" | tail -n1)
head $PREDICTED
mv $MARGINALS marginal_main.tsv
time cargo run -- -f $INPUT -v -m $MODEL --marginals --marginals-order=2
MARGINALs=$(ls -t1 | grep "output.*-marginal_effects.tsv" | head -n1)
mv $MARGINALS marginal_2nd.tsv
time cargo run -- -f $INPUT -v -m $MODEL --marginals --marginals-order=3
MARGINALs=$(ls -t1 | grep "output.*-marginal_effects.tsv" | head -n1)
head marginal_main.tsv
head marginal_2nd.tsv
head $MARGINALS
bat $MARGINALS
```

## Compile for release

```shell
cd mlp
cargo build --release
./target/release/mlp -h
```

## Tests on empirical data

```shell
cd mlp/misc
mkdir agridat/
cd agridat/
curl -L https://codeload.github.com/kwstat/agridat/tar.gz/main | tar -xz --strip=2 agridat-main/data

for FILE in $(ls *.txt)
do
FILE=australia.soybean.txt
# FILE=baena.bean.uniformity.txt

FILE_INPUT=${FILE%.txt*}.tsv
FILE_OUTPUT=${FILE%.txt*}.json

awk -v OFS="\t" '{print $5,$1,$2,$3,$4}' $FILE > $FILE_INPUT # specific to australia.soybean.txt

if [ $(head -n1 $FILE_INPUT | sed -z 's/\t/\n/g' | grep -n -i "year" | wc -l) -gt 0 ]
then
    echo "There is/are field/s with the string 'year' in them which may need to be converted into strings!"
    for idx in $(head -n1 $FILE_INPUT | sed -z 's/\t/\n/g' | grep -n -i "year" | cut -d: -f1)
    do
        echo $idx
        for year in $(tail -n+2 $FILE_INPUT | cut -f$idx - | sort | uniq)
        do
            year=$(tail -n+2 $FILE_INPUT | cut -f$idx - | sort | uniq | head -n1)
            sed -i "s/$year/YEAR-${year}/g" $FILE_INPUT
        done
    done
else
    echo "Nothing to convert into strings here!"
fi
# bat $FILE_INPUT
head $FILE_INPUT
tail $FILE_INPUT
wc -l $FILE_INPUT


../../target/release/mlp -h
time ../../target/release/mlp \
    -f $FILE_INPUT \
    -o $FILE_OUTPUT \
    -t 0 \
    -v \
    --n-batches 1 \
    --n-hidden-layers 3 \
    --n-epochs 100

MODEL=$(ls -t1 | grep ".json" | head -n1)
MARGINALS=$(ls -t1 | grep "marginal_effects.tsv" | head -n1)
head $MODEL
head $MARGINALS
mv $MARGINALS ${MARGINALS%-marginal_effects.tsv*}-marginals_main_effects.tsv

time ../../target/release/mlp \
    -f $FILE_INPUT \
    -t 0 \
    -v \
    -m $MODEL \
    --marginals \
    --marginals-order=2
MARGINALS=$(ls -t1 | grep "marginal_effects.tsv" | head -n1)
bat $MARGINALS

head ${MARGINALS%-marginal_effects.tsv*}-marginals_main_effects.tsv
head $MARGINALS

```

## Example Fits

![](./misc/Observed_vs_predicted-HL2-ReLU-Adam-E10-FPE0.25-B1-LR0.001-T20260408052818.svg)

![](./misc/Observed_vs_predicted-HL2-ReLU-Adam-E20-FPE0.25-B1-LR0.001-T20260408052908.svg)

![](./misc/Observed_vs_predicted-HL2-ReLU-Adam-E50-FPE0.25-B1-LR0.001-T20260408053111.svg)

![](./misc/Observed_vs_predicted-HL2-ReLU-Adam-E100-FPE0.25-B1-LR0.001-T20260408053517.svg)

## Special characters

- Used in progress bars: `█`
- Used as delimiters between non-numeric or categorical variable names and their levels: `➵`
- Used as delimiters in marginals' combinations: `▓`

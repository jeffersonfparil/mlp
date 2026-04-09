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


awk -v OFS="\t" '{print $5,$1,$2,$3,$4}' australia.soybean.txt > test.tsv
if [ $(head -n1 test.tsv | sed -z 's/\t/\n/g' | grep -n -i "year" | wc -l) -gt 0 ]
then
    echo "There is/are field/s with the string 'year' in them which may need to be converted into strings!"
    for idx in $(head -n1 test.tsv | sed -z 's/\t/\n/g' | grep -n -i "year" | cut -d: -f1)
    do
        echo $idx
        for year in $(tail -n+2 test.tsv | cut -f$idx - | sort | uniq)
        do
            year=$(tail -n+2 test.tsv | cut -f$idx - | sort | uniq | head -n1)
            sed -i "s/$year/YEAR-${year}/g" test.tsv
        done
    done
else
    echo "Nothing to convert into strings here!"
fi
bat test.tsv


../../target/release/mlp -h
time ../../target/release/mlp \
    -f "test.tsv" \
    -t 0 \
    -v \
    --n-batches 1 \
    --n-hidden-layers 3 \
    --n-epochs 100

```

## Example Fits

![](./misc/Observed_vs_predicted-HL2-ReLU-Adam-E10-FPE0.25-B1-LR0.001-T20260408052818.svg)

![](./misc/Observed_vs_predicted-HL2-ReLU-Adam-E20-FPE0.25-B1-LR0.001-T20260408052908.svg)

![](./misc/Observed_vs_predicted-HL2-ReLU-Adam-E50-FPE0.25-B1-LR0.001-T20260408053111.svg)

![](./misc/Observed_vs_predicted-HL2-ReLU-Adam-E100-FPE0.25-B1-LR0.001-T20260408053517.svg)

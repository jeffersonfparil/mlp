#!/bin/bash
for f in $(ls *_pheno.csv)
do
    # f=$(ls *_pheno.csv | head -n1)
    echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
    echo $f
    P=$(head -n1 $f | awk -F, '{print NF}')
    echo "$P columns"
    head -n1 $f
    g=${f%_pheno.csv*}_geno.csv
    cut -d, -f1 $f > ids_pheno.tmp
    cut -d, -f1 $g > ids_geno.tmp
    DIFF=$(diff ids_pheno.tmp ids_geno.tmp | wc -l)
    if [[ $DIFF -gt 0 ]]
    then
        echo "Mismatched IDs in $f and $g!"
        next
    fi
    cut -d, -f2- $g > geno.tmp
    for f_col in $(seq 2 $P)
    do
        # f_col=2
        cut -d, -f$f_col $f > trait_pheno.tmp
        trait_name=$(head -n1 trait_pheno.tmp)
        echo $trait_name
        paste -d, trait_pheno.tmp geno.tmp | sed -z 's/,/\t/g' > ${f%_pheno.csv*}-${trait_name}.tsv
        # echo ${f%_pheno.csv*}-${trait_name}.tsv
        # bat --wrap never ${f%_pheno.csv*}-${trait_name}.tsv
    done
done
rm *.tmp
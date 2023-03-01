#!/bin/bash

pyseer_out_dir="enet_nostand_lse3_results"
mkdir -p $pyseer_out_dir
fasta_dir="fasta_seqs"

# run pyseer
pyseer \
--wg enet \
--alpha 0.0069 \
--min-af 0.01 \
--max-af 0.99 \
--cor-filter 0.25 \
--no-standardise \
--lambda-se 3 \
--phenotypes metadata/erythromycin_phenotype.tsv \
--kmers unitig_output.pyseer \
--load-vars enet_results/cached_vars \
--save-model $pyseer_out_dir/model \
--uncompressed \
--cpu 2 > \
$pyseer_out_dir/unitigs_hits.txt

# query ggcaller
awk -F '\t' 'NR>1 && NF=1' $pyseer_out_dir/unitigs_hits.txt > $pyseer_out_dir/unitigs_hits_query.txt

ggcaller \
--graph $fasta_dir/input.gfa \
--colours $fasta_dir/input.bfg_colors \
--threads 16 \
--out $pyseer_out_dir/ggcaller_query \
--data ggcaller_output/ggc_data \
--query $pyseer_out_dir/unitigs_hits_query.txt \
--query-id 1.0

# create annotation graphs
python $SCRATCH/private_pyseer/pyseer/annotate_and_evaluate.py \
--var_types kmers \
--var_files $pyseer_out_dir/unitigs_hits.txt \
--annotation_files $pyseer_out_dir/ggcaller_query/matched_queries.fasta \
--method_names $pyseer_out_dir \
--out $pyseer_out_dir \
--ggcaller_xls metadata/NIHMS74007-combined-Supplementary_Datasets.xls \
--genes_of_interest ermB mel mefE \
--num_genes_label 60 \
--exclude_unmapped

# copy over to main folder space from /disk/scratch
mkdir -p ~/bacrank/sparc/$pyseer_out_dir
cp -r $pyseer_out_dir/* ~/bacrank/sparc/$pyseer_out_dir/

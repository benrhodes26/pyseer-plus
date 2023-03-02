#! /usr/bin/env python
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import os
import subprocess
import numpy as np
import pandas as pd

parser = ArgumentParser(description='Train multiple pyseer models on subsampled data', 
                        formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--n_repeats', type=int, default=100, help="number of times to rerun")
parser.add_argument('--subsample_frac', type=float, default=0.8, help="percent of data to sample")
parser.add_argument('--pheno_file', type=str, default="metadata/erythromycin_phenotype.tsv")
parser.add_argument('--out_dir', type=str, default="enet_nostand_lse3_results")
parser.add_argument('--dry-run', action='store_true', default=False)
args = parser.parse_args()

if args.dry_run:
    n_repeats = 2
else:
    n_repeats = args.n_repeats

shared_args = [
    "--wg enet",
    "--alpha 0.0069",
    "--min-af 0.01",
    "--max-af 0.99",
    "--cor-filter 0.25",
    "--no-standardise",
    "--lambda-se 3",    
    "--kmers unitig_output.pyseer",
    "--load-vars enet_results/cached_vars",
    "--uncompressed",
    "--cpu 2"
]

pheno_df = pd.read_csv(args.pheno_file, sep="\t")

for i in range(n_repeats):
    
    out_dir = os.path.join(args.out_dir, str(i))
    os.makedirs(out_dir, exist_ok=True)

    ss_idxs = np.random.choice(range(len(pheno_df)), size=int(len(pheno_df)*args.subsample_frac), replace=False)
    ss_pheno_df = pheno_df.iloc[ss_idxs]
    pheno_i = os.path.join(out_dir, f"pheno_{i}.tsv")
    ss_pheno_df.to_csv(pheno_i, sep="\t", index=0)

    specific_args = [f"--phenotypes {pheno_i}", f"--save-model {out_dir}/model"]
    
    out_file = open(os.path.join(out_dir, "unitigs_hits.txt"), 'w')
    err_file = open(os.path.join(os.path.join(out_dir, "pyseer_err.txt")), 'w')
    out_str = " ".join(['pyseer', *specific_args, *shared_args])
    if args.dry_run:    
        subprocess.call(['echo', " ".join(['pyseer', *specific_args, *shared_args])], stdout=out_file, shell=False)
    else:
        subprocess.call(['echo', " ".join(['pyseer', *specific_args, *shared_args])], stdout=err_file, shell=False)
        subprocess.call(['pyseer', *specific_args, *shared_args], stdout=out_file, stderr=err_file, shell=False)

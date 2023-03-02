#! /usr/bin/env python
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import sys
import time
from pyseer.annotate_and_evaluate import parse_pyseer_hits
import os
import subprocess
import numpy as np
import pandas as pd

parser = ArgumentParser(
    description='Concurrently train multiple pyseer models on subsampled data and aggregate the results', 
    formatter_class=ArgumentDefaultsHelpFormatter
    )
parser.add_argument('--n_repeats', type=int, default=100, help="number of times to rerun")
parser.add_argument('--subsample_frac', type=float, default=0.8, help="percent of data to sample")
parser.add_argument('--pheno_file', type=str, default="metadata/erythromycin_phenotype.tsv")
parser.add_argument('--out_dir', type=str, default="enet_nostand_lse3_results")
parser.add_argument('--parallel', type=int, default=16)
parser.add_argument('--dry-run', action='store_true', default=False)
args = parser.parse_args()

n_repeats = args.n_repeats

#### SHARED ARGUMENTS TO ALL PYSEER SCRIPTS
shared_args = [
    "--wg=enet",
    "--alpha=0.0069",
    "--min-af=0.01",
    "--max-af=0.99",
    "--cor-filter=0.25",
    "--no-standardise",
    "--lambda-se=3",    
    "--kmers=unitig_output.pyseer",
    "--load-vars=enet_results/cached_vars",
    "--uncompressed",
    "--cpu=2"
]

# 2-column dataframe specifying isolate file and phenotype
pheno_df = pd.read_csv(args.pheno_file, sep="\t")

running_procs = []
for i in range(n_repeats):
    
    # if at max capacity, wait until something has finished
    while len(running_procs) == args.parallel:
        for proc in running_procs:
            retcode = proc.poll()
            if retcode is not None: # Process finished.
                running_procs.remove(proc)
        wait = .1 if args.dry_run else 20
        time.sleep(wait)

    hitsfile = os.path.join(args.out_dir, str(i))
    os.makedirs(hitsfile, exist_ok=True)

    # subsample the isolate+phenotype file
    ss_idxs = np.random.choice(range(len(pheno_df)), size=int(len(pheno_df)*args.subsample_frac), replace=False)
    ss_pheno_df = pheno_df.iloc[ss_idxs]
    pheno_i = os.path.join(hitsfile, f"pheno_{i}.tsv")
    ss_pheno_df.to_csv(pheno_i, sep="\t", index=0)

    #### RUN-SPECIFIC ARGUMENTS
    specific_args = [
        f"--phenotypes={pheno_i}",
        f"--save-model={hitsfile}/model"
        ]
    
    out_file = open(os.path.join(hitsfile, "unitigs_hits.txt"), 'w')
    err_file = open(os.path.join(os.path.join(hitsfile, "pyseer_err.txt")), 'w')
    out_str = " ".join(['pyseer', *specific_args, *shared_args])
    
    print("Running job ", i)
    if args.dry_run:    
        p = subprocess.Popen(['echo', " ".join(['pyseer', *specific_args, *shared_args])], stdout=out_file, shell=False)
    else:
        subprocess.call(['echo', " ".join(['pyseer', *specific_args, *shared_args])], stdout=err_file, shell=False)
        p = subprocess.Popen(['pyseer', *specific_args, *shared_args], stdout=out_file, stderr=err_file, shell=False)
    running_procs.append(p)
        
# wait until all jobs have terminated
while running_procs:
    for proc in running_procs:
        retcode = proc.poll()
        if retcode is not None: # Process finished.
            running_procs.remove(proc)
    wait = 1 if args.dry_run else 60
    time.sleep(wait)

print("Finished all pyseer jobs!")
if args.dry_run:
    sys.exit()

### combine hits from all runs into a single hit list
print("Finding all hits across all runs...")
all_variants = set()
failures = []
for i in range(n_repeats):
    try:
        hitsfile = os.path.join(args.out_dir, str(i), "unitigs_hits.txt")
        df = parse_pyseer_hits(hitsfile)
        all_variants.update(set(df['variant'].values))
    except Exception as e:
        failures.append(i)
        print(f"failed to load {i}th result: ", e)

print(f"Found {len(all_variants)} unique variants across all runs")
print("Combining hits into a single file...")

successes = [i for i in range(n_repeats) if i not in failures]
all_df = pd.DataFrame(0,
                      index=list(all_variants),
                      columns=list(df.columns) + ["ss"]
                      )
for i in successes:
    hitsfile = os.path.join(args.out_dir, str(i), "unitigs_hits.txt")
    df = parse_pyseer_hits(hitsfile)
    cols = ["absbeta", "-logpval (unadjusted)", "-logpval (lmm-adjusted)"]
    all_df.loc[df['variant'], cols] += df[cols].values / len(successes)
    all_df.loc[df['variant'], ["ss"]] += (df[['absbeta']] != 0).values / len(successes)

all_df.to_csv(os.path.join(args.out_dir, f"unitigs_hits_ss_{n_repeats}.txt"), sep="\t")
print("Finished.")

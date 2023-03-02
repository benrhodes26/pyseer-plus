#! /usr/bin/env python
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pyseer.annotate_and_evaluate import parse_pyseer_hits
import os
import numpy as np
import pandas as pd

parser = ArgumentParser(
    description='combine results from multiple pyseer models', 
    formatter_class=ArgumentDefaultsHelpFormatter
    )
parser.add_argument('--n_repeats', type=int, default=100, help="number of times to rerun")
parser.add_argument('--out_dir', type=str, default="enet_nostand_lse3_results")
args = parser.parse_args()

n_repeats = args.n_repeats

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

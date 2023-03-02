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

print("combining hits into a single file...")
hitsfile = os.path.join(args.out_dir, "0", "unitigs_hits.txt")

all_df = parse_pyseer_hits(hitsfile)
all_df["ss"] = (all_df["absbeta"] != 0).astype(np.float32)
for i in range(1, n_repeats):

    # load dataframe of hits
    hitsfile = os.path.join(args.out_dir, str(i), "unitigs_hits.txt")
    df = parse_pyseer_hits(hitsfile)

    all_df["absbeta"] += df["absbeta"]
    all_df["-logpval (unadjusted)"] += df["-logpval (unadjusted)"]
    all_df["-logpval (lmm-adjusted)"] += df["-logpval (lmm-adjusted)"]
    all_df["ss"] += (df["absbeta"] != 0).astype(np.float32)

for col in ["absbeta", "-logpval (unadjusted)", "-logpval (lmm-adjusted)", "ss"]:
    all_df[col] /= n_repeats

all_df.to_csv(os.path.join(args.out_dir, f"unitigs_hits_ss_{n_repeats}.txt"), sep="\t", index=0)
print("Finished.")

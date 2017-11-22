# Copyright 2017 Marco Galardini and John Lees

'''Functions to read data into pyseer and iterate over instances'''

import sys
from .utils import set_env
# avoid numpy taking up more than one thread
with set_env(MKL_NUM_THREADS='1',
             NUMEXPR_NUM_THREADS='1',
             OMP_NUM_THREADS='1'):
    import numpy as np
import pandas as pd
from sklearn import manifold

def load_phenotypes(infile):
    p = pd.Series([float(x.rstrip().split()[-1])
                   for x in open(infile)],
                  index=[x.split()[0]
                         for x in open(infile)])
    return p


def load_structure(infile, p, max_dimensions, mds_type = "metric", n_cpus = 1):
    from .cmdscale import cmdscale

    m = pd.read_table(infile,
                      index_col=0)
    m = m.loc[p.index, p.index]

    # MDS
    if mds_type is "classic":
        projection, evals = cmdscale(m)
    else:
        metric_mds = True
        if mds_type is "non-metric":
            metric_mds = False
        elif mds_type is not "metric":
            sys.stderr.write("Unsupported mds type chosen. Assuming metric\n")

        mds = manifold.MDS(max_dimensions, metric_mds, n_jobs = n_cpus, dissimilarity='precomputed')
        projection = mds.fit_transform(m.values)

    m = pd.DataFrame(projection,
                     index=m.index)
    for i in range(m.shape[1]):
        m[i] = m[i] / max(abs(m[i]))
    return m


def load_covariates(infile, covariates, p):
    c = pd.read_table(infile,
                      header=None,
                      index_col=0)
    c.columns = ['covariate%d' % (x+2) for x in range(c.shape[1])]
    c = c.loc[p.index]
    # which covariates to use?
    if covariates is None:
        cov = pd.DataFrame([])
    else:
        cov = []
        for col in covariates:
            cnum = int(col.rstrip('q'))
            if cnum == 1 or cnum > c.shape[1] + 1:
                sys.stderr.write('Covariates columns values should be > 1 and lower ' +
                                 'than total number of columns (%d)\n' % (c.shape[1] + 1))
                return None
            if col[-1] == 'q':
                # quantitative
                cov.append(c['covariate%d' % cnum])
            else:
                # categorical, dummy-encode it
                categories = set(c['covariate%d' % cnum])
                categories.pop()
                for i, categ in enumerate(categories):
                    cov.append(pd.Series([1 if x == categ
                                          else 0
                                          for x in c['covariate%d' % cnum].values],
                                         index=c.index,
                                         name='covariate%d_%d' % (cnum, i)))
        cov = pd.concat(cov, axis=1)
    return cov


def iter_variants(p, m, cov, var_type, infile, all_strains, sample_order,
               min_af, max_af,
               filter_pvalue, lrt_pvalue, null_fit, firth_null,
               uncompressed):
    while True:
        if var_type is "vcf":
            l = next(infile)
        else:
            l = infile.readline()

        # check for EOF
        if not l:
            raise StopIteration

        # Parse depending on input file type. Need to end with a variant name and pres/abs dictionary
        d = {}
        if var_type == "kmers":
            if not uncompressed:
                l = l.decode()
            var_name, strains = l.split()[0], l.rstrip().split('|')[1].lstrip().split()

            d = {x.split(':')[0]: 1
                 for x in strains}

        elif var_type == "vcf":
            # Do not support multiple alleles. Use 'bcftools norm' to split these
            if len(l.alts) > 1:
                sys.stderr.write("Multiple alleles at " + l.contig + "_" + str(l.pos) + ". Skipping\n")
                yield (None, None, None, None, None, None,
                   None, None, None, None,
                   None, None)
                continue
            elif "PASS" not in l.filter.keys() and "." not in l.filter.keys():
                yield (None, None, None, None, None, None,
                   None, None, None, None,
                   None, None)
                continue

            var_name = "_".join([l.contig, str(l.pos)] + [str(allele) for allele in l.alleles])
            for sample, call in l.samples.items():
                # This is dominant encoding. Any instance of '1' will count as present
                # Could change to additive, summing instances, or reccessive only counting
                # when all instances are 1.
                # Shouldn't matter for bacteria, but some people call hets
                for haplotype in call['GT']:
                    if str(haplotype) is not "." and haplotype != 0:
                        d[sample] = 1
                        break

        elif var_type == "Rtab":
            split_line = l.rstrip().split()
            var_name, strains = split_line[0], split_line[1:]
            for present, sample in zip(strains, sample_order):
                if present is not '0':
                    d[sample] = 1

        # Use common dictionary to format design matrix etc
        kstrains = sorted(set(d.keys()).intersection(all_strains))
        nkstrains = sorted(all_strains.difference(set(kstrains)))

        # default for missing samples is absent kmer
        # currently up to user to be careful about matching pheno and var files
        for x in nkstrains:
            d[x] = 0

        af = float(len(kstrains)) / len(all_strains)
        # filter by AF
        if af < min_af or af > max_af:
            # pass it to the actual tests to keep track
            yield (var_name, None, None, None, None, af,
                   None, None, None, None,
                   kstrains, nkstrains)
            continue

        v = p.values
        k = np.array([d[x] for x in p.index
                      if x in d])
        c = cov.values

        yield (var_name, v, k, m, c, af,
               filter_pvalue, lrt_pvalue, null_fit, firth_null,
               kstrains, nkstrains)


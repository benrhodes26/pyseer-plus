import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from collections import defaultdict
from .utils import plot, subplots, save_fig


def prf1_curves_of_ranked_list(tgt_feats, feats, ranking_metric, metric_name, results_container):
    """Get cumulative precision, recall and F1 curves for a list of ranked hits (desc order)
    
    :param tgt_feats: indices of causal features
    :param feats: indices of features we have ranks for
    :param ranking_metric: list of non-negative scores for each feature (features with score=0 discarded)
    :param ranking_metric_name: name of the ranking metric
    :param results_container: dictionary to store results

    :return: None, results are stored in results_container
    """
    tgt_feats = set(tgt_feats)
    
    # rank the features
    ranked_feats = [x for m, x in sorted(zip(ranking_metric, feats), reverse=True) if m > 0]
    
    for k in range(len(ranked_feats)):

        topk = set(ranked_feats[:k+1])
        true_positives = tgt_feats.intersection(topk)
        precision = len(true_positives) / (k+1)
        recall = len(true_positives) / len(tgt_feats)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)

        results_container[f"{metric_name}_precision"].append(precision)
        results_container[f"{metric_name}_recall"].append(recall)
        results_container[f"{metric_name}_F1"].append(f1)


def plot_all_prf1_curves(save_dir, method_names, pr_curve_dict, max_len=None):
    """Plot precision, recall and F1 curves for all methods
    
    :param save_dir: directory to save plots
    :param method_names: list of method names
    :param pr_curve_dict: dictionary of precision, recall and F1 curves for each method
    :param max_len: maximum length of the curves to plot (default: plot everything)
    :return: None, plots are saved to save_dir
    """

    fig, axs = subplots(1, 3, f"precision/recall curves", "list length", height=8)
    for ax, mtype in zip(axs, ["precision", "recall", "F1"]):
        ax.set_ylabel(mtype)
        for k, v in pr_curve_dict.items():
            if mtype not in k: continue
            yvals = np.array(v)
            if max_len: yvals = yvals[:max_len]
            if yvals.size == 0: yvals = np.array([0.])
            ax.plot(yvals, label=" ".join(k.split("_")[:-1]), 
                    alpha=0.8, linewidth=1.5, markevery=max(1, len(yvals)//10))

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94),
                   fancybox=True, shadow=True, ncol=len(method_names)*2, fontsize=8)
        
    name = f"prf1_curves_"
    if max_len: name += str(max_len)
    save_fig(fig, f"{save_dir}/", name)
    

def plot_causal_cf_curves(save_dir, method_names, tgt_feats, pr_curve_dict, max_len=None):
    """Plot a 'checked-and-found' which contains the necessary info to compute precision+recall for any list length
    
    :param save_dir: directory to save plots
    :param method_names: list of method names
    :param tgt_feats: causal variants/features
    :param pr_curve_dict: dictionary of precision, recall and F1 curves for each method
    :param max_len: maximum length of the curves to plot (default: plot everything)
    :return: None, plots are saved to save_dir
    """

    fig, ax = plot(f"CF curves", "# variants checked", "# causal variants found", figsize=(10, 6))
    ax.set_ylim(0, len(tgt_feats)+1)

    for k, v in pr_curve_dict.items():
        if "recall" not in k:
            continue
        yvals = np.array(v) * len(tgt_feats)
        # print("num_causal feats:", len(tgt_feats))  # TODO
        # print("max recall:", np.array(v)[-1])
        if yvals.size == 0: yvals = np.array([0.])
        if max_len: yvals = yvals[:max_len]
        ax.plot(yvals, label=" ".join(k.split("_")[:-1]), 
                alpha=0.8, linewidth=1.5, markevery=max(1, len(yvals)//10))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94),
               fancybox=True, shadow=True, ncol=len(method_names)*2, fontsize=8)
    
    name = f"checked_and_found_curves_"
    if max_len: name += str(max_len)
    save_fig(fig, f"{save_dir}/", name)


def plot_gene_scatter_summaries(all_dfs,
                                method_names, 
                                good_genes, 
                                bad_genes, 
                                save_dir,
                                x_metric=("mean", np.mean), 
                                y_metric=("max", "max"), 
                                plot_name="", 
                                num_genes_label=40, 
                                lmm_pval=False
                                ):
    """Plot scatter plots of gene-level metrics
    
    :param all_dfs: list of dataframes, one for each method
    :param method_names: list of method names
    :param good_genes: list of genes to include in the plot
    :param bad_genes: list of genes to exclude from the plot
    :param save_dir: directory to save plots
    :param x_metric: metric to use on the x-axis (default: mean)
    :param y_metric: metric to use on the y-axis (default: max)
    :param plot_name: name of the plot (default: "")
    :param num_genes_label: number of genes to label on the plot (default: 40)
    :param lmm_pval: whether to use lmm-adjusted p-values (default: False)
    :return: None, plots are saved to save_dir
    """

    n_plots = 2 * len(method_names)  # coefs and pvals for each method
    num_cols = 2
    num_rows = int(np.ceil(n_plots/num_cols))
    fig, axs = subplots(num_rows, num_cols, width=15, height=12,
                        sharex=False, sharey=False, max_width=15, tight_layout=False)
    all_figtext = []
    for i in range(num_rows):
        df = all_dfs[i]
        if 'gene' not in df.columns:
            continue

        # remove any genes I am not interested in
        for g in bad_genes:
            df = df[df['gene'] != g]

        pval_type = '-logpval (lmm-adjusted)' if lmm_pval else '-logpval (unadjusted)'
        for j, col in enumerate(['absbeta', pval_type]):
            
            ax = axs[i][j] if num_rows > 1 else axs[j]

            # remove all zeros for this metric
            tmp = df[df[col] > 0]

            # group by gene
            gene_groups = tmp.groupby("gene")

            # compute max/avg value of current metric per gene
            grouped_info = gene_groups.agg(
                x_val=(col, x_metric[1]),
                y_val=(col, y_metric[1]),
            )
            
            ax.scatter(grouped_info["x_val"], grouped_info["y_val"], alpha=0.8,
                        s=gene_groups.size()/2, edgecolors='k', linewidth=0.3)
            
            col = "|beta|" if j == 0 else "-log(pval)"
            ax.set_xlabel(x_metric[0] + " " + col)
            ax.set_ylabel(y_metric[0] + " " + col)

            k = num_genes_label // 2
            to_annotate = list(set(grouped_info.sort_values("x_val", ascending=False).index[:k]).union(
                            set(grouped_info.sort_values("y_val", ascending=False).index[:k])))
            for g in good_genes:
                if (g not in to_annotate) and (g in grouped_info.index):
                    to_annotate.append(g)
            
            anno_idxs = [list(grouped_info.index).index(a) for a in to_annotate]
            figtext = []
            for i, txt in zip(anno_idxs, to_annotate):
                if txt in good_genes:
                    bbox=dict(facecolor='green', alpha=0.3)
                else:
                    bbox=None
                figtext.append(ax.text(grouped_info["x_val"][i], grouped_info["y_val"][i],
                                        txt, fontsize=8, bbox=bbox))
            all_figtext.append(figtext)
    
    # label each row with the corresponding method name
    if num_rows > 1:
        for ax, row in zip(axs[:,0], method_names):
            ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - 5, 0),
                        xycoords=ax.yaxis.label, textcoords='offset points',
                        size='large', ha='right', va='center')
    else:
        fig.suptitle(method_names[0])
    
    plt.subplots_adjust(wspace=0.35, hspace=0.25)

    if all_figtext:
        try:
            from adjustText import adjust_text
            print("Adjusting annotations for readability...")
            for a, ax in zip(all_figtext, axs.flat):
                adjust_text(a, ax=ax, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
            print("done!")
        except ImportError:
            print("WARNING: text annotations on gene summary figure may be hard to read -" \
                "please install the adjustText package to ameliorate this")
            pass

        name = f"{plot_name}_gene_summaries"
        save_fig(fig, f"{save_dir}/", name)


def plot_gene_cf_curves(annotated_ranked_hits, max_list_len=None, plot_name="", 
                        save_dir="", genes=('pbpX', 'penA', 'pbp1A')):

    # TODO: need to adjust for a list of dfs and metric_names as input
    print("gene-base CF curves not implemented yet")
    sys.exit()
    fig, axs = subplots(1, 4, " ".join(genes) + " hits", "# variants checked", height=8)
    plotting_info = list(zip(*get_metric_plot_info()))
    for col, name, c, l, m in plotting_info:

        # sort by a particular metric
        sorted_df = annotated_ranked_hits.sort_values(col, ascending=False)

        # remove any rows with score of zero under that metric
        cutoff = np.argmin(sorted_df[col])
        if max_list_len: cutoff = min(cutoff, max_list_len)
        sorted_df = sorted_df.iloc[:cutoff]

        # do the plotting
        total_hits = ((sorted_df['gene'] == genes[0]) | 
                      (sorted_df['gene'] == genes[1]) | 
                      (sorted_df['gene'] == genes[2])).values
        
        axs[0].plot(np.cumsum(total_hits), label=name, c=c, linestyle=l,
                     marker=m, alpha=0.8, markevery=max(1, len(total_hits)//5))
        axs[0].set_ylabel("/".join(genes) + " found", fontsize=12)

        for i, g in enumerate(genes):
            axs[i+1].plot(np.cumsum((sorted_df['gene'] == g).values), label=name, c=c, 
                            linestyle=l, marker=m, alpha=0.8, markevery=max(1, len(total_hits)//10))
            axs[i+1].set_ylabel(g + " found", fontsize=12)
    
    handles, labels = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.9), 
               fancybox=True, shadow=True, ncol=len(plotting_info), fontsize=8)
               
    name = plot_name + "_" + '_'.join(genes) + "_CFcurves"
    if max_list_len: name += f"_{max_list_len}"
    save_fig(fig, f"{save_dir}/", name)


def eval_hits_against_vars_of_interest(tgt_feats, var_files, var_dfs, method_names, use_adjusted_pval, save_dir):
    """For various ranking-metrics we construct a ranked lists of features and then compute 
        the precision/recall/F1 of the topk hits, for all values of k in [1, len(ranked_list)]

    :param tgt_feats: list of target variants
    :param var_files: list of files containing pyseer hits
    :param var_dfs: list of dataframes containing pyseer hits
    :param method_names: list of method names corresponding to the ranked lists
    :param use_adjusted_pval: whether to use the adjusted p-values or not
    :param save_dir: directory to save the results
    """

    # compute precision/recall/F1 curves for each metric and store the results in a `pr_curve_dict`
    pr_curve_dict = defaultdict(list)
    pval_type = '-logpval (lmm-adjusted)' if use_adjusted_pval else '-logpval (unadjusted)'
    for col in ['absbeta', pval_type]:
        for var_file, df, name in zip(var_files, var_dfs, method_names):
            name += f"_{col}"
            prf1_curves_of_ranked_list(tgt_feats, df['variant'].values, df[col], name, pr_curve_dict)

            prf = np.array([pr_curve_dict[name + "_precision"], pr_curve_dict[name + "_recall"], pr_curve_dict[name + "_F1"]]).T
            prf_df = pd.DataFrame(prf, columns=["precision", "recall", "F1"])
            prf_df.to_csv(os.path.join(os.path.dirname(var_file), f"{col}_precision_recall_f1.tsv"), sep='\t')

    plot_all_prf1_curves(save_dir, method_names, pr_curve_dict, max_len=len(tgt_feats)*5)  # zoomed in
    plot_all_prf1_curves(save_dir, method_names, pr_curve_dict, max_len=len(df))  # zoomed out

    plot_causal_cf_curves(save_dir, method_names, tgt_feats, pr_curve_dict, max_len=len(tgt_feats)*5)  # zoomed in
    plot_causal_cf_curves(save_dir, method_names, tgt_feats, pr_curve_dict, max_len=len(df))  # zoomed out


def parse_ggcaller_xls(annotation_file, group_mge=False, group_hypothetical=False, use_function=False):
    """Extract a dictionary mapping COG id to gene name
    
    Spreadsheet must formatted like S1 & S2 here: 
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5708525/#SD1

    Such a spreadsheet is needed when annotating unitig hits generated by ggcaller:
    (https://ggcaller.readthedocs.io/en/latest/usage.html#interpreting-results)
    """
    annotation_dict = {}
    df = pd.read_excel(annotation_file)
    for index, row in df.iterrows():
        
        cog_id = row[0]

        # determine whether protein is MGE (mobile genetic element)
        mge_rows = row[4:11]
        pseudogene = row[24]
        mge_cols_sum = mge_rows.str.count("Y").sum()
        all_cols_sum = row[4:].str.count("Y").sum()
        is_mge = (mge_cols_sum == 1) and ((all_cols_sum == 1) or (all_cols_sum == 2 and pseudogene == "Y"))
        
        # is the gene function unknown?
        unknown_function = row[21:24].str.count("Y").sum() > 0

        if is_mge and group_mge:
            annotation_dict[cog_id] = "MGE"
        elif unknown_function and group_hypothetical:
            annotation_dict[cog_id] = "hypothetical"
        else:
            gene_name, gene_function = row[2], row[3]
            if pd.isnull(gene_name):
                if use_function:
                    annotation_dict[cog_id] = gene_function
                else:
                    annotation_dict[cog_id] = cog_id
            else:
                annotation_dict[row[0]] = gene_name

            annotation_dict[row[0]] = annotation_dict[row[0]].replace(" (fragment)", "")

    return annotation_dict


def parse_ggcaller_query(fasta, delim="_", position=-1):
    """Taken from https://github.com/samhorsfield96/ggCaller_manuscript"""
    try:
        from Bio import SeqIO
    except ImportError:
        sys.stderr.write("please install biopython in order to run this script")
        sys.exit(1)

    var_to_cog = {}
    for seq_record in SeqIO.parse(fasta, "fasta"):
    
        # each variant maps to a set of genes (represented as cog ids)
        cogs = set()
        split_desc = seq_record.description.split("annotation=")
        desc = split_desc[-1]
        desc = desc.split(";")
        for entry in desc:    
            if entry != "":
                split_entry = entry.split(delim)
                cogs.add(split_entry[position])

        # parse number of significant hits
        desc = split_desc[0]
        desc = desc.split("QUERY=")[-1]
        desc = desc.split(";")
        desc = [x.strip() for x in desc]
        var_to_cog.update({d: cogs for d in desc})

    print(f"{len(var_to_cog)} variants found in ggcaller query file")
    return var_to_cog


def parse_pyseer_hits(var_file):
    """Load a dataframe of pyseer hits, transforming the pvals and betas so that they can be used to rank hits"""
    with open(var_file, 'r') as f:
        headers = f.readline().strip().split('\t')

    cols = ['variant', 'af', 'filter-pvalue', 'lrt-pvalue', 'beta']
    if 'beta_idx' in headers:
        cols.append('beta_idx')

    df = pd.read_csv(var_file, sep='\t', usecols=cols)
    df['beta'] = df['beta'].apply(np.abs)
    df['filter-pvalue'] = df['filter-pvalue'].apply(lambda x: -np.log(np.maximum(x, np.e**-700))/np.log(10))
    df['lrt-pvalue'] = df['lrt-pvalue'].apply(lambda x: -np.log(np.maximum(x, np.e**-700))/np.log(10))
    df = df.rename(columns={'beta': 'absbeta', 'filter-pvalue': '-logpval (unadjusted)', 'lrt-pvalue': '-logpval (lmm-adjusted)'})
    
    return df


def get_annotated_variants(var_types, var_files, annotation_files, ggcaller_xls=None,
                            exclude_unmapped=False, genes_of_interest=None):
    """Return a dataframe of the form pd.DataFrame(['variant', 'absbeta', '-logpval', 'gene'])
    
    Parameters
    ----------
    var_types : list of str
        The type of variant. Must be one of ['vcf', 'kmers']
    var_files : list of str
        The path to the file containing the variants identified by pyseer
    annotation_files : list of str
        The path to the file containing an annotation file (e.g. gff or matched_queries.fasta from ggcaller)
    ggcaller_xls : str
        The path to the tsv mapping a gene identifier to gene name (e.g 'CLS0342' to 'pbp2x')
    exclude_unmapped : bool
        Whether to exclude variants that could not be mapped to a gene
    genes_of_interest : list of str
        A list of genes for which we output a separate tsv file containing any hits found there
    """

    all_dfs = []
    for var_type, var_file, anno_file in zip(var_types, var_files, annotation_files):
        
        pyseer_hits = parse_pyseer_hits(var_file)

        if not anno_file:
            var_df = pyseer_hits

        elif var_type == "vcf":
            assert anno_file.endswith('.gff'), \
                f"Expected a .gff annotation file to accompany a vcf file, but got: {anno_file}"
            
            # extract position from var_name and search gff annotation file for gene name
            # var_name = "_".join([variant.contig, str(variant.pos)] + [str(allele) for allele in variant.alleles])
            raise NotImplementedError("vcf-based variants not available yet.")
        
        elif var_type == "kmers":
            
            if ".fasta" in anno_file:
                # TODO: strictly speaking, we don't require a spreadsheet. A 2-column csv mapping cog_id to gene_name would suffice
                assert ggcaller_xls, f"Cannot annotate the variants in {var_file} using the annotation {anno_file}. \
                    Presuming .fasta annotation file was generated by ggcaller, and require a compatible spreadsheet specified \
                    via --ggcaller_xls (see https://github.com/samhorsfield96/ggCaller_manuscript#pangenome-wide-association-study-pgwas)"
                
                var_to_cogs = parse_ggcaller_query(anno_file)
                cog_to_genename = parse_ggcaller_xls(ggcaller_xls)

                var_df = []
                for i, row in pyseer_hits.iterrows():
                    var = row['variant']
                    if var not in var_to_cogs:
                        if not exclude_unmapped:
                            var_df.append((*row.values, "unmapped"))
                        continue
                    cogs = var_to_cogs[var]
                    for cog in cogs:
                        if cog in cog_to_genename:
                            var_df.append((*row.values, cog_to_genename[cog]))
                        else:
                            var_df.append((*row.values, cog))
                
                var_df = pd.DataFrame(var_df, columns=list(pyseer_hits.columns) + ['gene'])

            elif ".gff" in anno_file:
                # TODO: call ./kmer_mapping/annotate_hits.py
                raise NotImplementedError
            
            else:
                raise ValueError(f"Expected .fasta or .gff annoation file, but got {anno_file}")
        else:
            raise ValueError("Expected --var_types to contain 'vcf' or 'kmers', but got ", var_type)
        
        if anno_file:
            varfile_dir = os.path.dirname(var_file)
            var_df.to_csv(os.path.join(varfile_dir, "annotated_hits.tsv"), sep='\t')
            if genes_of_interest:
                for g in genes_of_interest:
                    subdf = var_df[var_df['gene'] == g].reset_index(drop=True)
                    subdf.drop(columns=['gene'], inplace=True)
                    subdf.to_csv(os.path.join(varfile_dir, f"annotated_hits_{g}.tsv"), sep='\t', index=0)
        
        all_dfs.append(var_df)

    return all_dfs


def parse_args():
    """load & augment experiment configuration"""
    parser = ArgumentParser(description='Train model.', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out', type=str, default=os.getcwd(), 
                        help="directory to save output figures'")
    parser.add_argument('--var_types', nargs='+', default=[],
                        help="list whose elements should be either 'vcf' or 'kmers'")
    parser.add_argument('--var_files', nargs='+', default=[], 
                        help="list of file names containing pyseer outputs")
    parser.add_argument('--annotation_files', nargs='+', default=[],
                         help="list of file names containing annotation files")
    parser.add_argument('--method_names', nargs='+', default=[], 
                        help="list of method names")
    parser.add_argument('--genes_of_interest', nargs='+', default=[], 
                        help="list of genes")
    parser.add_argument('--genes_not_of_interest', nargs='+', default=[],
                         help="list of genes to ignore")
    parser.add_argument('--variants_of_interest', type=str, default="", 
                        help="file of new-line separated variant names (e.g. ground-truth causal variants in a simulation)")
    parser.add_argument('--ggcaller_xls', type=str, default="", 
                        help="directory containing spreadsheet for ggcaller")
    parser.add_argument('--num_genes_label', type=int, default=50,
                         help="number of genes in scatter plot to label'")
    parser.add_argument('--exclude_unmapped', action='store_true', default=False, 
                        help='Exclude variants that are not mapped to a gene')
    parser.add_argument('--use_lmm_pval', action='store_true', default=False,
                         help='Use the adjusted pvalue generated by pyseer')

    return parser.parse_args()


def main(options=None):
    np.set_printoptions(precision=3)
    if not options:
        options = parse_args()
    assert len(options.var_types) == len(options.var_files) == len(options.annotation_files) == len(options.method_names), \
        "--var_types, --var_files, --annotation_files, and --method_names must all be the same length, but " \
        "got {} {} {} {}".format(len(options.var_types), len(options.var_files), len(options.annotation_files), len(options.method_names))

    save_dir = options.out
    genes_of_interest = options.genes_of_interest

    # list of dataframes
    var_dfs = get_annotated_variants(options.var_types, options.var_files, options.annotation_files,
                                      options.ggcaller_xls, options.exclude_unmapped)

    # how well do pyseer hits recover certain variants? useful for e.g bacgwasim or real-genotype + simulated-phenotype
    if options.variants_of_interest:
        vars_of_interest = pd.read_csv(options.variants_of_interest, header=None, sep='\t').values.flatten()
        eval_hits_against_vars_of_interest(vars_of_interest, options.var_files, var_dfs,
                                           options.method_names, options.use_lmm_pval, save_dir)
    
    # make scatter plots for each ranking metric, grouping variants by gene
    plot_gene_scatter_summaries(var_dfs, options.method_names, genes_of_interest,
                                options.genes_not_of_interest, save_dir, plot_name="avg_vs_max",
                                num_genes_label=options.num_genes_label, lmm_pval=options.use_lmm_pval)

    # plot checked-and-found (CF) curves for genes of interest at different zoom-levels
    if genes_of_interest:
        num_vars_on_x_axis = [100, 1000, None]  # None -> show all variants
        for l in num_vars_on_x_axis:
            plot_gene_cf_curves(var_dfs, max_list_len=l, save_dir=save_dir, genes=genes_of_interest)


if __name__ == "__main__":
    main()

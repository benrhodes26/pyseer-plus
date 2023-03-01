import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def save_fig(fig, save_dir, name="", both_formats=False, close=True):
    os.makedirs(save_dir, exist_ok=True)
    fig.savefig(os.path.join(save_dir, f"{name}.pdf"), bbox_inches='tight', dpi=300)
    
    if both_formats:
        fig.savefig(os.path.join(save_dir, f"{name}.png"), bbox_inches='tight', dpi=300)
    if close:
        plt.close(fig)  # otherwise figure may hang around in memory


def plot(title, xlabel, ylabel, figsize=(6, 4), xscale="linear", yscale="linear", use_grid=True):
    """A super basic plotting function"""
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(xscale=xscale, yscale=yscale, xlabel=xlabel, ylabel=ylabel)  # a convenient way to set lots of properties at once!
    ax.grid(use_grid)
    fig.suptitle(title)  # why not fig.set_title()? I have no idea, and this seems like a bad design choice to me...
    
    return fig, ax


def subplots(nrows, ncols, title=None, xlabel=None, ylabel=None, width=9, height=6,
             scale="auto", sharex=True, sharey=True, grid=True, max_width=12, tight_layout=True):
    """A function for making nicely formatted grids of subplots with shared labels on the x-axis and y axis"""
    assert nrows > 1 or ncols > 1, "use plot() to create a single subplot"
    
    fig_width = width * ncols
    fig_height = height * nrows
    
    if scale == "auto":
        scale = min(1.0, max_width/fig_width)  # width of figure cannot exceede max_width inches
        scale = min(scale, max_width/fig_height)  # height of figure cannot exceede max_width inches
    
    figsize=(scale * fig_width, scale * fig_height)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, sharex=sharex, sharey=sharey)
    
    for ax in axs.flat:
        ax.grid(grid)  # maybe add grid lines
        
    if title: fig.suptitle(title)
    if xlabel: fig.supxlabel(xlabel)  # shared x label
    if ylabel: fig.supylabel(ylabel)  # shared y label
    
    fig.tight_layout()  # adjust the padding between and around subplots to neaten things up
    
    return fig, axs


def prf1_curves_by_threshold_percentiles(targets, hit_ids, ranks, n_thresholds=100):
    """Compute 3 metrics (precision, recall & F1) for a range of thresholds
      and insert results into results_container

    :param targets: binary array specifying whether a feature is causal (x=1) or not (x=0)
    :param ranks: array specifiying a rank for each feature
    :param thresholds: (optional) list of thresholds for which the metric should be computed
    """
    targets = set(targets)
    thresholds = np.percentile(np.unique(ranks), np.arange(n_thresholds))[::-1]  # largest to smalleset
    precisions, recalls, f1s = [], [], []
    for i, threshold in enumerate(thresholds):

        pred_idxs = np.where(ranks >= threshold)[0]
        preds = set(hit_ids[pred_idxs])
        true_positives = targets.intersection(preds)

        precision = len(true_positives) / (len(preds) + 1e-6)
        recall = len(true_positives) / len(targets)
        f1 = (2 * precision * recall) / (precision + recall + 1e-6)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
    return precisions, recalls, f1s


def prf1_curves_of_hit_list(tgt_feats, feats, ranking_metric, ranking_metric_name, results_container):
    """Get cumulative precision, recall and F1 curves for a list of hits,
      which we order by (descending) ranking_metric
    
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

        results_container[f"{ranking_metric_name}_precision"].append(precision)
        results_container[f"{ranking_metric_name}_recall"].append(recall)
        results_container[f"{ranking_metric_name}_F1"].append(f1)


def plot_all_prf1_curves(conf, pr_curve_dict, max_len=None):

    _, ranking_metric_names, colors, linestyles, markers = get_metric_plot_info()

    fig, axs = subplots(1, 3, f"{conf.method_name} trade-off curves", "list length", height=8)
    for ax, mtype in zip(axs, ["precision", "recall", "F1"]):
        ax.set_ylabel(mtype)
        for metric_name, c, ls, m in zip(ranking_metric_names, colors, linestyles, markers):

            yvals = np.array(pr_curve_dict[metric_name + "_" + mtype])
            if max_len: yvals = yvals[:max_len]
            if yvals.size == 0: yvals = np.array([0.])

            ax.plot(yvals, label=metric_name, c=c, linestyle=ls, marker=m, 
                    alpha=0.8, linewidth=1.5, markevery=max(1, len(yvals)//10))

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94),
                   fancybox=True, shadow=True, ncol=len(ranking_metric_names), fontsize=8)
        
    name = f"{conf.method_name}_prf1_curves_"
    if max_len: name += str(max_len)
    save_fig(fig, f"{conf.save_dir}/", name)
    

def plot_causal_cf_curves(conf, tgt_feats, pr_curve_dict, max_len=None):

    _, ranking_metric_names, colors, linestyles, markers = get_metric_plot_info()

    fig, ax = plot(f"{conf.dname} CF curves", "# variants checked", "# causal variants found", figsize=(10, 6))
    ax.set_ylim(0, len(tgt_feats)+1)

    for metric_name, c, ls, m in zip(ranking_metric_names, colors, linestyles, markers):

        yvals = np.array(pr_curve_dict[metric_name + "_" + "recall"]) * len(tgt_feats)
        if yvals.size == 0: yvals = np.array([0.])
        if max_len: yvals = yvals[:max_len]

        ax.plot(yvals, label=metric_name, c=c, linestyle=ls, marker=m, 
                alpha=0.8, linewidth=1.5, markevery=max(1, len(yvals)//10))

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94),
                fancybox=True, shadow=True, ncol=len(ranking_metric_names)+1, fontsize=8)
    
    name = f"{conf.method_name}_checked_and_found_curves_"
    if max_len: name += str(max_len)
    save_fig(fig, f"{conf.save_dir}/", name)


def plot_gene_scatter_summaries(all_dfs, method_names, genes_of_interest, save_dir, x_val=("mean", np.mean),
                                 y_val=("max", "max"), plot_name="", num_genes_label=40):

    n_plots = 2 * len(method_names)  # coefs and pvals for each method
    num_cols = 2
    num_rows = int(np.ceil(n_plots/num_cols))
    fig, axs = subplots(num_rows, num_cols, width=15, height=12,
                        sharex=False, sharey=False, max_width=15, tight_layout=False)
    all_figtext = []
    for i in range(num_rows):
        df = all_dfs[i]
        for j, col in enumerate(['absbeta', '-logpval (unadjusted)']):
            
            ax = axs[i][j] if num_rows > 1 else axs[j]

            # remove all zeros for this metric
            nonzero_df = df[df[col] > 0]
            gene_groups = nonzero_df.groupby("gene")
            
            # compute max/avg value of current metric per gene
            grouped_info = gene_groups.agg(
                x_val=(col, x_val[1]),
                y_val=(col, y_val[1]),
            )
            ax.scatter(grouped_info["x_val"], grouped_info["y_val"], alpha=0.8,
                        s=gene_groups.size()/2, edgecolors='k', linewidth=0.3)
            
            col = "|beta|" if j == 0 else "-log(pval)"
            ax.set_xlabel(x_val[0] + " " + col)
            ax.set_ylabel(y_val[0] + " " + col)

            k = num_genes_label // 2
            to_annotate = list(set(grouped_info.sort_values("x_val", ascending=False).index[:k]).union(
                            set(grouped_info.sort_values("y_val", ascending=False).index[:k])))
            for g in genes_of_interest:
                if g not in to_annotate:
                    to_annotate.append(g)
            
            anno_idxs = [list(grouped_info.index).index(a) for a in to_annotate]
            figtext = []
            for i, txt in zip(anno_idxs, to_annotate):
                if txt in genes_of_interest:
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

    name = f"{plot_name}_gene_summaries_v1"
    save_fig(fig, f"{save_dir}/", name)


def plot_gene_cf_curves(annotated_ranked_hits, max_list_len=None, plot_name="", 
                        save_dir="", genes=('pbpX', 'penA', 'pbp1A')):

    # TODO: need to adjust for a list of dfs and metric_names as input
    raise NotImplementedError
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


def eval_causal_groundtruth(conf, tgt_feats, vars_df):
    """For various ranking-metrics we construct a ranked lists of features and then compute 
        the precision/recall/F1 of the topk hits, for all values of k in [1, len(ranked_list)]

    :param conf: dict, configuration object
    :param tgt_feats: np.ndarray containing indices of causal features
    :param feats_df: pd.DataFrame 
    """
    # TODO: need to rewrite to account for a list of vars_dfs
    raise NotImplemented
    df_cols, ranking_metric_names = get_metric_plot_info()[:2]
    ranking_metrics = [vars_df[col] for col in df_cols]
    
    # compute precision/recall/F1 curves for each metric and store the results in a `pr_curve_dict`
    pr_curve_dict = defaultdict(list)
    for metric, name in zip(ranking_metrics, ranking_metric_names):
        prf1_curves_of_hit_list(tgt_feats, vars_df.index, metric, name, pr_curve_dict)

    plot_all_prf1_curves(conf, pr_curve_dict, max_len=len(tgt_feats)*10)
    plot_all_prf1_curves(conf, pr_curve_dict, max_len=len(vars_df))

    plot_causal_cf_curves(conf, tgt_feats, pr_curve_dict, max_len=len(tgt_feats)*3)
    plot_causal_cf_curves(conf, tgt_feats, pr_curve_dict, max_len=len(tgt_feats)*10)
    plot_causal_cf_curves(conf, tgt_feats, pr_curve_dict, max_len=len(tgt_feats)*30)
    plot_causal_cf_curves(conf, tgt_feats, pr_curve_dict, max_len=len(vars_df))


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
    df = pd.read_csv(var_file, sep='\t', usecols=['variant', 'af', 'filter-pvalue', 'lrt-pvalue', 'beta', 'notes'])
    df['beta'] = df['beta'].apply(np.abs)
    df['filter-pvalue'] = df['filter-pvalue'].apply(lambda x: -np.log(x)/np.log(10))
    df['lrt-pvalue'] = df['lrt-pvalue'].apply(lambda x: -np.log(x)/np.log(10))
    df = df.rename(columns={'beta': 'absbeta', 'filter-pvalue': '-logpval (unadjusted)', 'lrt-pvalue': '-logpval (lmm-adjusted)'})
    return df


def get_annotated_variants(var_types, var_files, annotation_files, ggcaller_xls=None, exclude_unmapped=False):
    """Return a dataframe of the form pd.DataFrame(['variant', 'gene', 'absbeta', '-logpval'])"""

    all_dfs = []
    for var_type, var_file, anno_file in zip(var_types, var_files, annotation_files):

        var_to_cogs = pd.DataFrame(columns=['variant', 'gene'])

        if var_type == "snps":
            # TODO! pyseer labels snps as:
            # var_name = "_".join([variant.contig, str(variant.pos)] + [str(allele) for allele in variant.alleles])
            # extract position and search gff annotation file for gene name
            raise NotImplementedError
        
        elif var_type == "kmers":
            
            if ".fasta" in anno_file:
                # TODO: strictly speaking, we don't require a spreadsheet. A 2-column csv mapping cog_id to gene_name would suffice
                assert ggcaller_xls, f"Cannot annotate the variants in {var_file} using the annotation {anno_file}. \
                    Presuming .fasta annotation file was generated by ggcaller, and require a compatible spreadsheet specified \
                    via --ggcaller_xls (see https://github.com/samhorsfield96/ggCaller_manuscript#pangenome-wide-association-study-pgwas)"
                
                pyseer_hits = parse_pyseer_hits(var_file)
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
            raise ValueError("Expected --var_types to contain 'snps' or 'kmers', but got ", var_type)
        
        # TODO: run some basic sanity checks on var_df
        
        all_dfs.append(var_df)

    return all_dfs


def get_metric_plot_info(version=3):
    # TODO: make these command line options (along with a list of vcfs/kmers)
    if version == 1:
        metric_names = ["abscoef", "log_pval", "enet_abscoef", "enet_log_pval", "MDS_inclusion_rate", "ss", "abscoef_ss"]
        plotting_names = ["Lasso |beta|", "Lasso + pval", "Enet |beta|", "Enet + pval", "MDS", "SS", "SS |beta|"]
        colors = ['gray', 'black', 'salmon', 'red', "deepskyblue", 'springgreen', 'wheat']
        linestyles = ['--'] * 4 + ['-'] * 3
        markers = ['.']*2 + ["*"]*2 + ["+", "^", "d"]

    elif version == 2:
        metric_names = ["enet_abscoef", "enet_abscoef_se1", "enet_abscoef_se2", "enet_abscoef_se3",
                        "abscoef_ss", "abscoef_se1_ss",  "abscoef_se2_ss", "abscoef_se3_ss"]
        plotting_names = ["Enet |beta|", "Enet se1 |beta|", "Enet se2 |beta|", "Enet se3 |beta|", 
                          "SS |beta|", "SS se1 |beta|", "SS se2 |beta|", "SS se3 |beta|"]             
        colors = ['red', "deepskyblue", 'springgreen', 'wheat']*2
        linestyles = ['-'] * 4 + ['--'] * 4
        markers = ["*", "+", "^", "d"]*2

    elif version == 3:
        metric_names = ["enet_abscoef", "enet_abscoef_se1", "enet_abscoef_se2", "enet_abscoef_se3", "enet_nonorm_abscoef", 
                        "enet_nonorm_abscoef_se1", "enet_nonorm_abscoef_se2",  "enet_nonorm_abscoef_se3"]
        plotting_names = ["Enet |beta|", "Enet se1 |beta|", "Enet se2 |beta|", "Enet se3 |beta|", "~normEnet |beta|", 
                          "~normEnet se1 |beta|", "~normEnet se2 |beta|", "~normEnet se3 |beta|"]             
        colors = ['red', "deepskyblue", 'springgreen', 'wheat']*2
        linestyles = ['-'] * 4 + ['--'] * 4
        markers = ["*", "+", "^", "d"]*2
    else:
        raise ValueError("must specify valid numeric version as input to this function")

    assert len(plotting_names) == len(colors) == len(linestyles) == len(metric_names), \
        "expected these lists to be of same length"
    
    return metric_names, plotting_names, colors, linestyles, markers



def parse_args():
    """load & augment experiment configuration"""
    parser = ArgumentParser(description='Train model.', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out', type=str, default=os.getcwd(), help="directory to save output figures'")
    parser.add_argument('--var_types', nargs='+', default=[], help="list whose elements should be either 'snps' or 'kmers'")
    parser.add_argument('--var_files', nargs='+', default=[], help="list of file names containing pyseer outputs")
    parser.add_argument('--annotation_files', nargs='+', default=[], help="list of file names containing annotation files")
    parser.add_argument('--method_names', nargs='+', default=[], help="list of method names")
    parser.add_argument('--genes_of_interest', nargs='+', default=[], help="list of gene names")
    parser.add_argument('--causal_vars', nargs='+', default=[], help="list of causal vars (for simulated phenotypes)")
    parser.add_argument('--ggcaller_xls', type=str, default=os.getcwd(), help="directory to save output figures'")
    parser.add_argument('--num_genes_label', type=int, default=40, help="number of genes in scatter plot to label'")
    parser.add_argument('--exclude_unmapped', action='store_true', default=False, help='Exclude variants that are not mapped to a gene')
    # parser.add_argument('--pvalue_type', type=str, default="un", help="directory to save output figures'")

    return parser.parse_args()


def main():
    np.set_printoptions(precision=3)
    options = parse_args()
    save_dir = options.out
    genes_of_interest = options.genes_of_interest

    # list of dataframes of form pd.DataFrame(['variant', 'gene', 'absbeta', '-logpval'])
    var_dfs = get_annotated_variants(options.var_types, options.var_files, options.annotation_files,
                                      options.ggcaller_xls, options.exclude_unmapped)

    # eval ranking for synthetic dataset e.g 'bacgwasim' dataset or real-genotype + simulated-phenotype
    if options.causal_vars:
        eval_causal_groundtruth(options, options.causal_vars, var_dfs)
    
    # make scatter plots for each ranking metric, grouping variants by gene
    plot_gene_scatter_summaries(var_dfs, options.method_names, genes_of_interest, 
                                save_dir, plot_name="avg_vs_max", num_genes_label=options.num_genes_label)
    plot_gene_scatter_summaries(var_dfs, options.method_names, genes_of_interest, 
                                save_dir, x_val=("mean", np.mean), y_val=("count", len),
                                plot_name="avg_vs_len", num_genes_label=options.num_genes_label)

    # plot checked-and-found (CF) curves for genes of interest at different zoom-levels
    if genes_of_interest:
        num_vars_on_x_axis = [100, 1000, None]  # None -> show all variants
        for l in num_vars_on_x_axis:
            plot_gene_cf_curves(var_dfs, max_list_len=l, save_dir=save_dir, genes=genes_of_interest)


if __name__ == "__main__":
    main()

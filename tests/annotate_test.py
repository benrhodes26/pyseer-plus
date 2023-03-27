import unittest
import numpy as np
import pandas as pd
from pyseer.annotate_and_evaluate import main
from pyseer.utils import dotdict


class TestAnnotateEvaluate(unittest.TestCase):
    gt_file = 'tests/unit_tests_data/true_hits.txt'
    pred1_file = 'tests/unit_tests_data/predicted_hits_v1.txt'
    pred2_file = 'tests/unit_tests_data/predicted_hits_v2.txt'
    pred3_file = 'tests/unit_tests_data/predicted_hits_v3.txt'

    def test_causal_variant_precision_recall(self):
        options = dotdict({
        "out" : 'tests/tmp_results/',
        "var_types" : ['vcf'],
        "annotation_files": [""],
        "method_names": ['X'],
        "variants_of_interest": self.gt_file,
        "genes_of_interest": [],
        "genes_not_of_interest": [],
        "ggcaller_xls": "",
        "num_genes_label": 50,
        "exclude_unmapped": False,
        "use_lmm_pval": False
        })

        options["var_files"] = [self.pred1_file]
        main(options)
        res_df = pd.read_csv('tests/unit_tests_data/X_absbeta_precision_recall_f1.tsv', sep="\t")
        precision, recall = res_df['precision'].values[-1], res_df['recall'].values[-1]
        self.assertTrue(abs(precision - 8/2217) < 1E-15)
        self.assertTrue(abs(recall - 1.0) < 1E-15)

        options["var_files"] = [self.pred2_file]
        main(options)
        res_df = pd.read_csv('tests/unit_tests_data/X_absbeta_precision_recall_f1.tsv', sep="\t")
        precision, recall = res_df['precision'].values[-1], res_df['recall'].values[-1]
        self.assertTrue(abs(precision - 1.0) < 1E-15)
        self.assertTrue(abs(recall - 1.0) < 1E-15)

        options["var_files"] = [self.pred3_file]
        main(options)
        res_df = pd.read_csv('tests/unit_tests_data/X_absbeta_precision_recall_f1.tsv', sep="\t")
        precision, recall = res_df['precision'].values[-1], res_df['recall'].values[-1]
        self.assertTrue(abs(precision - 0.5) < 1E-15)
        self.assertTrue(abs(recall - 0.5) < 1E-15)


if __name__ == '__main__':
    # TestAnnotateEvaluate().test_causal_variant_precision_recall()
    unittest.main()

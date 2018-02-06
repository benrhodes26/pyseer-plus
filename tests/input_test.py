import os
import gzip
import unittest
import numpy as np
import pandas as pd
from pysam import VariantFile
from collections import deque
from pyseer.input import load_phenotypes
from pyseer.input import load_structure
from pyseer.input import load_lineage
from pyseer.input import load_covariates
from pyseer.input import load_burden
from pyseer.input import read_variant
from pyseer.input import hash_pattern


DATA_DIR = 'tests'
P = os.path.join(DATA_DIR, 'subset.pheno')
M = os.path.join(DATA_DIR, 'distances_smaller.tsv.gz')
LIN = os.path.join(DATA_DIR, 'lineage_clusters.txt')
COV = os.path.join(DATA_DIR, 'covariates.txt')
B = os.path.join(DATA_DIR, 'burden_regions.txt')
KMER = os.path.join(DATA_DIR, 'kmers.gz')
PRES = os.path.join(DATA_DIR, 'presence_absence_smaller.Rtab')
VCF = os.path.join(DATA_DIR, 'variants.vcf.gz')


class TestLoadFunctions(unittest.TestCase):
    def test_load_phenotypes(self):
        p = pd.read_table(P,
                          index_col=0)
        t = load_phenotypes(P, None)
        self.assertTrue(abs((p['binary'].values - t.values).max()) < 1E-7)
        t = load_phenotypes(P, 'binary')
        self.assertTrue(abs((p['binary'].values - t.values).max()) < 1E-7)
        t = load_phenotypes(P, 'continuous')
        self.assertTrue(abs((p['continuous'].values - t.values).max()) < 1E-7)
        with self.assertRaises(KeyError):
            load_phenotypes(P, 'test')
        with self.assertRaises(FileNotFoundError):
            load_phenotypes('nope', None)

    def test_load_structure(self):
        p = pd.read_table(P,
                          index_col=0)['binary']
        t = load_structure(M, p, 10, 'classic', 1, None)
        tr = np.array([-0.05277648, 0.97716044, 0.89997233, -0.00807741,
                       0.44765289, -0.10104701,  0.01159616, -0.06489092,
                       0.34748335, -0.20009074, 0.16696374, -0.00144086,
                       -0.04317903, -0.24258361,  0.8920215 , 0.00246485])
        tc = np.array([-0.05277648, -0.24282047, -0.10282981, 0.9252433,
                       -0.03556807, -0.08743139, 0.39253432, 0.95790659,
                       -0.31324724, -1., -0.95997958, 0.01828477, -0.15606428,
                       0.4633444, 0.9253031, -0.30580183, 0.21006771,
                       0.30042054, -0.9428476, 0.00626202])
        self.assertTrue(abs((t.values[0] - tr).max()) < 1E-7)
        self.assertTrue(abs((t.values[:,0] - tc).max()) < 1E-7)
        t = load_structure(M, p, 5, 'non-metric', 1, 42)
        tr = np.array([-0.06635193, -0.38792784, -0.74925127,
                       0.79345277, 0.51067032])
        tc = np.array([-0.06635193, -0.33070992, 0.08796615, -0.33782627,
                       -0.05940981, -0.59254645, 1., 0.27093241, 0.85027172,
                       0.1260554, -0.42917097, -0.50911587, 0.15936359,
                       0.28853126, 0.41050079, 0.06470051, 0.47632535,
                       -0.87963914, -0.75617326, 0.13955464])
        self.assertTrue(abs((t.values[0] - tr).max()) < 1E-7)
        self.assertTrue(abs((t.values[:,0] - tc).max()) < 1E-7)
        t = load_structure(M, p, 5, 'metric', 1, 42)
        tr = np.array([-0.97249805, -0.24747933, 0.49918088,
                       -0.04765291, 0.34207924])
        tc = np.array([-0.97249805, 0.89582851, -0.25740033, -0.16657464,
                       0.22295658, 0.42219513, -0.44404529, -0.83914526,
                       -0.34305642, 0.58112243, 0.56611122, 0.16864438,
                       -0.42256615, -0.30980933, -0.61344531, -0.3418638,
                       -0.55246211, 0.74549807, 1., 0.66051035])
        self.assertTrue(abs((t.values[0] - tr).max()) < 1E-7)
        self.assertTrue(abs((t.values[:,0] - tc).max()) < 1E-7)
        # bogus mds_type, should default to "metric"
        t = load_structure(M, p, 5, 'test', 1, 42)
        self.assertTrue(abs((t.values[0] - tr).max()) < 1E-7)
        self.assertTrue(abs((t.values[:,0] - tc).max()) < 1E-7) 
        # no file to be found
        with self.assertRaises(FileNotFoundError):
            t = load_structure('nope', p, 5, 'test', 1, 42)
        # no intersection
        with self.assertRaises(SystemExit) as cm:
            x = p.copy(deep=True)
            x.index = ['nope_%d' % i for i in range(p.shape[0])]
            t = load_structure(M, x, 5, 'test', 1, 42)
        self.assertEqual(cm.exception.code, 1)

    def test_load_lineage(self):
        p = pd.read_table(P,
                          index_col=0)['binary']
        t = load_lineage(LIN, p)
        tr = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 1, 0, 0, 0, 0, 0])
        tc = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0])
        self.assertTrue(abs((t[0][0] - tr).max()) < 1E-7)
        self.assertTrue(abs((t[0][:,0] - tc).max()) < 1E-7) 
        tl = ['BAPS_1', 'BAPS_10', 'BAPS_12', 'BAPS_14',
              'BAPS_15', 'BAPS_16', 'BAPS_19', 'BAPS_2',
              'BAPS_20', 'BAPS_22', 'BAPS_27', 'BAPS_28',
              'BAPS_29', 'BAPS_3', 'BAPS_4', 'BAPS_5',
              'BAPS_6', 'BAPS_7', 'BAPS_9']
        self.assertEqual(t[1], tl)
        # no file found
        with self.assertRaises(FileNotFoundError):
            t = load_lineage('nope', p)
        # no intersection
        with self.assertRaises(SystemExit) as cm:
            x = p.copy(deep=True)
            x.index = ['nope_%d' % i for i in range(p.shape[0])]
            t = load_lineage(LIN, x)
        self.assertEqual(cm.exception.code, 1)

    def test_load_covariates(self):
        p = pd.read_table(P,
                          index_col=0)['binary']
        t = load_covariates(COV, ['2q', '3'], p)
        th = np.array([[1, 1, 0],
                       [2, 0, 1],
                       [3, 0, 0],
                       [4, 1, 0],
                       [5, 0, 1]])
        tt = np.array([[46,  1,  0],
                       [47,  0,  1],
                       [48,  0,  0],
                       [49,  1,  0],
                       [50,  0,  1]])
        self.assertTrue(abs((t.head(5).values - th).max()) < 1E-7)
        self.assertTrue(abs((t.tail(5).values - tt).max()) < 1E-7)
        # no file found
        with self.assertRaises(FileNotFoundError):
            t = load_covariates('nope', ['2q', '3'], p)
        # incorrect columns
        t = load_covariates(COV, ['2q', '3', '4'], p)
        self.assertEqual(t, None)
        t = load_covariates(COV, ['2q', '3', '1'], p)
        self.assertEqual(t, None)
        # no intersection
        with self.assertRaises(SystemExit) as cm:
            x = p.copy(deep=True)
            x.index = ['nope_%d' % i for i in range(p.shape[0])]
            t = load_covariates(COV, ['2q', '3'], x)
        self.assertEqual(cm.exception.code, 1)
        # no covariates to load
        t = load_covariates(COV, None, p)
        self.assertEqual(t.shape, (0, 0))
        t = load_covariates(COV, [], p)
        self.assertEqual(t.shape, (0, 0))

    def test_load_burden(self):
        t = []
        load_burden(B, t)
        tl = [('CDS1', 'FM211187:3910-3951'),
              ('CDS2', 'FM211187:4006-4057')]
        self.assertEqual(t, tl)
        # no file found
        with self.assertRaises(FileNotFoundError):
            load_burden('nope', [])
        # not the right file
        with self.assertRaises(ValueError):
            load_burden(P, [])


class TestVariantLoading(unittest.TestCase):
    def test_read_variant(self):
        with self.assertRaises(ValueError):
            read_variant(None, None, 'test',
                         None, None, None,
                         None, None)

    def test_read_variant_kmer(self):
        p = pd.read_table(P,
                          index_col=0)['binary']
        infile = gzip.open(KMER)
        t = read_variant(infile, p, 'kmers',
                         False, [], False,
                         p.index, [])
        eof, k, var_name, kstrains, nkstrains, af = t
        self.assertEqual(eof, False)
        self.assertTrue(abs((k -
                         np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
                                   0, 0])).max()) < 1E-7)
        self.assertEqual(var_name,
                         'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        self.assertEqual(kstrains,
                         ['sample_43'])
        self.assertEqual(nkstrains,
                         sorted(['sample_%d' % x
                                 for x in list(range(1, 43)) +
                                          list(range(44, 51))]))
        self.assertEqual(af, 0.02)
        # not providing samples
        with self.assertRaises(ZeroDivisionError):
            t = read_variant(infile, p, 'kmers',
                             False, [], False,
                             set(), [])
        # providing burden - no effect
        t = read_variant(infile, p.head(5), 'kmers',
                         True, [], False,
                         p.head(5).index, [])
        eof, k, var_name, kstrains, nkstrains, af = t
        self.assertEqual(eof, False)
        self.assertTrue(abs((k -
                         np.array([1, 1, 0, 1, 0])).max()) < 1E-7)
        self.assertEqual(var_name,
                         'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
        self.assertEqual(kstrains,
                         ['sample_1', 'sample_2', 'sample_4'])
        self.assertEqual(nkstrains,
                         ['sample_3', 'sample_5'])
        self.assertEqual(af, 0.6)
        # uncompressed option
        with self.assertRaises(TypeError):
            t = read_variant(infile, p, 'kmers',
                             False, [], True,
                             p.index, [])
        # different type
        with self.assertRaises(ValueError):
            t = read_variant(infile, p.head(5), 'Rtab',
                             False, [], False,
                             p.head(5).index, [])
        with self.assertRaises(AttributeError):
            t = read_variant(infile, p.head(5), 'vcf',
                             False, [], False,
                             p.head(5).index, [])
        # read until exhaustion
        while not t[0]:
            t = read_variant(infile, p, 'kmers',
                             False, [], False,
                             p.index, [])
        eof, k, var_name, kstrains, nkstrains, af = t
        self.assertEqual(eof, True)
        self.assertEqual(k, None)
        self.assertEqual(var_name, None)
        self.assertEqual(kstrains, None)
        self.assertEqual(nkstrains, None)
        self.assertEqual(af, None)
        # different file
        infile = open(PRES)
        with self.assertRaises(IndexError):
            t = read_variant(infile, p.head(5), 'kmers',
                             True, [], True,
                             p.head(5).index, [])
        infile = gzip.open(VCF)
        with self.assertRaises(IndexError):
            t = read_variant(infile, p.head(5), 'kmers',
                             True, [], False,
                             p.head(5).index, [])

    def test_read_variant_rtab(self):
        p = pd.read_table(P,
                          index_col=0)['binary']
        infile = open(PRES)
        header = infile.readline().rstrip()
        sample_order = header.split()[1:]
        t = read_variant(infile, p, 'Rtab',
                         False, [], False,
                         p.index, sample_order)
        eof, k, var_name, kstrains, nkstrains, af = t
        self.assertEqual(eof, False)
        self.assertEqual(abs((k - np.ones(50)).max()), 0.0)
        self.assertEqual(var_name,
                         'COG_1')
        self.assertEqual(kstrains,
                         sorted(['sample_%d' % x
                                 for x in range(1, 51)]))
        self.assertEqual(nkstrains,
                         [])
        self.assertEqual(af, 1.0)
        # not providing samples
        with self.assertRaises(ValueError):
            t = read_variant(infile, p, 'Rtab',
                             False, [], False,
                             set(), [])
        with self.assertRaises(ZeroDivisionError):
            t = read_variant(infile, p, 'Rtab',
                             False, [], False,
                             set(), sample_order)
        # providing burden - no effect
        t = read_variant(infile, p.head(5), 'Rtab',
                         True, [], False,
                         p.head(5).index, sample_order)
        eof, k, var_name, kstrains, nkstrains, af = t
        self.assertEqual(eof, False)
        self.assertTrue(abs((k -
                         np.array([1, 1, 1, 1, 1])).max()) < 1E-7)
        self.assertEqual(var_name,
                         'COG_4')
        self.assertEqual(kstrains,
                         ['sample_1', 'sample_2', 'sample_3',
                          'sample_4', 'sample_5'])
        self.assertEqual(nkstrains,
                         [])
        self.assertEqual(af, 1.0)
        # uncompressed option - no effect
        t = read_variant(infile, p.head(5), 'Rtab',
                         False, [], True,
                         p.head(5).index, sample_order)
        eof, k, var_name, kstrains, nkstrains, af = t
        self.assertEqual(eof, False)
        self.assertTrue(abs((k -
                         np.array([1, 1, 1, 1, 1])).max()) < 1E-7)
        self.assertEqual(var_name,
                         'COG_5')
        self.assertEqual(kstrains,
                         ['sample_1', 'sample_2', 'sample_3',
                          'sample_4', 'sample_5'])
        self.assertEqual(nkstrains,
                         [])
        self.assertEqual(af, 1.0)
        # different type
        with self.assertRaises(IndexError):
            t = read_variant(infile, p.head(5), 'kmers',
                             False, [], True,
                             p.head(5).index, sample_order)
        with self.assertRaises(AttributeError):
            t = read_variant(infile, p.head(5), 'vcf',
                             False, [], False,
                             p.head(5).index, [])
        # read until exhaustion
        while not t[0]:
            t = read_variant(infile, p, 'Rtab',
                             False, [], False,
                             p.index, sample_order)
        eof, k, var_name, kstrains, nkstrains, af = t
        self.assertEqual(eof, True)
        self.assertEqual(k, None)
        self.assertEqual(var_name, None)
        self.assertEqual(kstrains, None)
        self.assertEqual(nkstrains, None)
        self.assertEqual(af, None)
        # different file
        infile = gzip.open(KMER)
        with self.assertRaises(ValueError):
            t = read_variant(infile, p.head(5), 'Rtab',
                             False, [], False,
                             p.head(5).index, sample_order)
        infile = gzip.open(VCF)
        with self.assertRaises(ValueError):
            t = read_variant(infile, p.head(5), 'Rtab',
                             False, [], False,
                             p.head(5).index, sample_order)

    def test_read_variant_vcf(self):
        p = pd.read_table(P,
                          index_col=0)['binary']
        infile = VariantFile(VCF)
        t = read_variant(infile, p, 'vcf',
                         False, [], False,
                         p.index, [])
        eof, k, var_name, kstrains, nkstrains, af = t
        self.assertEqual(eof, False)
        self.assertEqual(abs((k - np.zeros(50)).max()), 0.0)
        self.assertEqual(var_name,
                         'FM211187_16_G_A')
        self.assertEqual(kstrains,
                         [])
        self.assertEqual(nkstrains,
                         sorted(['sample_%d' % x
                                 for x in range(1, 51)]))
        self.assertEqual(af, 0.0)
        # not providing samples
        t = read_variant(infile, p, 'vcf',
                         False, [], False,
                         set(), [])
        eof, k, var_name, kstrains, nkstrains, af = t
        self.assertEqual(eof, False)
        self.assertEqual(k, None)
        self.assertEqual(var_name, None)
        self.assertEqual(kstrains, None)
        self.assertEqual(nkstrains, None)
        self.assertEqual(af, None)
        # providing burden
        burden_regions = deque([])
        load_burden(B, burden_regions)
        t = read_variant(infile, p.head(5), 'vcf',
                         True, burden_regions, False,
                         p.head(5).index, [])
        eof, k, var_name, kstrains, nkstrains, af = t
        self.assertEqual(eof, False)
        self.assertTrue(abs((k -
                         np.array([0, 0, 0, 0, 0])).max()) < 1E-7)
        self.assertEqual(var_name,
                         'CDS1')
        self.assertEqual(kstrains,
                         [])
        self.assertEqual(nkstrains,
                         ['sample_1', 'sample_2', 'sample_3',
                          'sample_4', 'sample_5'])
        self.assertEqual(af, 0.0)
        # uncompressed option - no effect
        t = read_variant(infile, p.head(5), 'vcf',
                         False, [], True,
                         p.head(5).index, [])
        eof, k, var_name, kstrains, nkstrains, af = t
        self.assertEqual(eof, False)
        self.assertTrue(abs((k -
                         np.array([0, 1, 0, 0, 0])).max()) < 1E-7)
        self.assertEqual(var_name,
                         'FM211187_3982_C_A')
        self.assertEqual(kstrains,
                         ['sample_2'])
        self.assertEqual(nkstrains,
                         ['sample_1', 'sample_3',
                          'sample_4', 'sample_5'])
        self.assertEqual(af, 0.2)
        # different type
        with self.assertRaises(AttributeError):
            t = read_variant(infile, p.head(5), 'kmers',
                             False, [], True,
                             p.head(5).index, [])
        with self.assertRaises(AttributeError):
            t = read_variant(infile, p.head(5), 'Rtab',
                             False, [], False,
                             p.head(5).index, [])
        # read until exhaustion
        while not t[0]:
            t = read_variant(infile, p, 'vcf',
                             False, [], False,
                             p.index, [])
        eof, k, var_name, kstrains, nkstrains, af = t
        self.assertEqual(eof, True)
        self.assertEqual(k, None)
        self.assertEqual(var_name, None)
        self.assertEqual(kstrains, None)
        self.assertEqual(nkstrains, None)
        self.assertEqual(af, None)
        # different file
        infile = gzip.open(KMER)
        with self.assertRaises(AttributeError):
            t = read_variant(infile, p.head(5), 'vcf',
                             False, [], False,
                             p.head(5).index, [])
        infile = open(PRES)
        with self.assertRaises(AttributeError):
            t = read_variant(infile, p.head(5), 'vcf',
                             False, [], False,
                             p.head(5).index, [])


class TestHashing(unittest.TestCase):
    def test_hash_pattern(self):
        p = pd.read_table(P,
                          index_col=0)['binary']
        h = hash_pattern(p.values)
        self.assertEqual(h, b'gwi2uQb68G5LfLr7qJuVpw==\n')
        # wrong input types
        with self.assertRaises(AttributeError):
            hash_pattern([0, 1, 0, 1, 1, 1])
            hash_pattern(p)


if __name__ == '__main__':
    unittest.main()
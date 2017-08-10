"""
@author: niebler
"""
import math

from scipy.sparse import coo_matrix


def __group_to_sparse_vector(grp, max_id, binary=False):
    key = grp[0]
    if binary:
        data = [1 for k in grp[1]]
    else:
        data = [k[1] for k in grp[1]]
    cols = [k[0] for k in grp[1]]
    rows = [0] * len(grp[1])
    vector = coo_matrix((data, (rows, cols)), shape=(1, max_id))
    return key, vector


def __create_coocc_pairs(fragment):
    return [(fragment[0], k) for k in fragment[1:]] + [(k, fragment[0]) for k in fragment[1:]]


class SimplePageSim(object):
    def __init__(self, vocabsize=0, min_coocc=0):
        self._vocabsize = vocabsize
        self._vocabmap = None
        self._min_coccc = min_coocc
        pass

    def fit(self, paths):
        """
        creates a vocabulary map from a set of paths.
        :param paths: a set of paths in pure form.
        :return: a dictionary, mapping a pure pathElement to its ID.
        """
        pages_with_counts = paths \
            .flatMap(lambda x: x) \
            .map(lambda x: (x, 1)) \
            .reduceByKey(lambda a, b: a + b) \
            .top(self._vocabsize, key=lambda x: x[1])
        self._vocabmap = dict(zip([pwc[0] for pwc in pages_with_counts], range(len(pages_with_counts))))

    def __replace_pages_with_ids(self, paths):
        """
        converts a path of text items to a path of their respective int IDs. Saves space and computation time.
        :param paths: a set of paths in pure form
        :param vocabulary_map_bc: a spark broadcast of a vocabulary map.
        :return: an RDD with the transformed paths
        """
        vocabulary_map_bc = sc.broadcast(self._vocabmap)
        self._fitted_paths = paths \
            .map(lambda path: [vocabulary_map_bc.value[pathElement] for pathElement in path]) \
            .cache()

    def __get_cooccs_with_counts(self, transformed_paths, window_size=3, minfrequency=1):
        """
        counts all id cooccurrences in the transformed paths in a given window size.
        finally filters the cooccurrences by a given minimum frequency to reduce noise.
        :param transformed_paths: a set of paths with ids instead of pure path elements
        :param window_size: a given window size, in which ids will be counted as cooccurring.
        :param minfrequency: a minimum frequency for cooccurrence counts, by which the final cooccurrence set is filtered.
        :return: an RDD with triples ((id1, id2), count)
        """
        return transformed_paths \
            .flatMap(lambda path: [path[i: i + window_size] for i in range(len(path))]) \
            .flatMap(__create_coocc_pairs) \
            .filter(lambda pair: pair[0] != pair[1]) \
            .map(lambda pair: (pair, 1)) \
            .reduceByKey(lambda a, b: a + b) \
            .filter(lambda (_, count): count >= minfrequency) \
            .cache()

    def __get_cooccs_with_tfidf(self, transformed_paths, window_size=3, min_tfidf=1):
        """
        counts all id cooccurrences in the transformed paths in a given window size.
        finally filters the cooccurrences by a given minimum frequency to reduce noise.
        :param transformed_paths: a set of paths with ids instead of pure path elements
        :param window_size: a given window size, in which ids will be counted as cooccurring.
        :param min_tfidf: a minimum value for tfidf, by which the final cooccurrence set is filtered.
        :return: an RDD with triples ((id1, id2), count)
        """
        path_count = transformed_paths.count()
        return transformed_paths.zipWithIndex() \
            .flatMap(lambda (path, index): [(path[i: i + window_size], index) for i in range(len(path))]) \
            .map(lambda (pairs, index): (__create_coocc_pairs(pairs), index)) \
            .flatMap(lambda (pairs, index): [(k, index) for k in pairs]) \
            .filter(lambda (pair, index): pair[0] != pair[1]) \
            .groupByKey() \
            .mapValues(lambda occ_list: [k for k in occ_list]) \
            .map(lambda (transition, occ_list): (transition, len(set(occ_list)), len(occ_list))) \
            .map(lambda (transition, df, length): (transition, math.log(path_count / df), (length / df))) \
            .map(lambda (transition, idf, tf): (transition, idf * tf)) \
            .filter(lambda (_, count): count >= min_tfidf) \
            .cache()

    def __cooccs_to_vec(self, cooccs_with_count, vector_size, binary=False):
        """
        creates vectors from cooccurrence counts, with a given vector_size. if binary is enabled, only occurrences will be
        taken into account.
        :param cooccs_with_count: an RDD of coocc tuples.
        :param vector_size: the size of the generated vectors. must be higher than the maximum ID in the vocabulary map.
        :param binary: if enabled, only occurrences will be saved in the sparse vectors, all value >0 will be saved as 1.
        :return: an RDD of sparse vectors. Each vector represents a word and is identified by its ID, according to the
        vocabulary map.
        """
        return cooccs_with_count.map(lambda (coocc, count): (coocc[0], (coocc[1], count))) \
            .groupByKey() \
            .map(lambda grp: __group_to_sparse_vector(grp, vector_size, binary)) \
            .cache()

    def transform(self, paths, windowsize=3, binary=False, tfidf=False):
        """
        the whole pipeline.
        :param sc: a Spark Context.
        :param paths: an RDD of paths.
        :param windowsize: a given window size, in which ids will be counted as cooccurring.
        :param binary: if enabled, only occurrences will be saved in the sparse vectors, every value >0 will be saved as 1.
        :param tfidf: if enabled, the coorcurences will be ifidf random
        :param minfrequency: a minimum frequency for cooccurrence counts, by which the final cooccurrence set are filtered.
        :return: an RDD of sparse vectors. Each vector represents a word and is identified by its ID, according to the
        vocabulary map.
        """
        transformed_paths = self.__replace_pages_with_ids(paths, self._vocabmap)
        if tfidf:
            cooccs = self.__get_cooccs_with_tfidf(transformed_paths, windowsize, self._min_coccc)
        else:
            cooccs = self.__get_cooccs_with_counts(transformed_paths, windowsize, self._min_coccc)
        vectors = self.__cooccs_to_vec(cooccs, self._vocabsize, binary)
        reverse_vocabulary_map = sc.broadcast({v: k for k, v in self._vocabmap.items()})
        return vectors.map(lambda (node, vector): (reverse_vocabulary_map.value[node], vector))

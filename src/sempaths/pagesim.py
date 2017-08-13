"""
@author: niebler
"""
import math

from sempaths.utils import create_coocc_pairs, group_to_sparse_vector


class PageSim(object):
    def __init__(self, spark_context, vocabsize=0, min_coocc=0):
        self._sc = spark_context
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
            .reduceByKey(lambda a, b: a + b)
        if self._vocabsize == 0:
            pages_with_counts = pages_with_counts.collect()
            self._vocabsize = len(pages_with_counts)
        else:
            pages_with_counts = pages_with_counts.top(self._vocabsize, key=lambda x: x[1])
        self._vocabmap = dict(zip([pwc[0] for pwc in pages_with_counts], range(len(pages_with_counts))))

    def __replace_pages_with_ids(self, paths):
        """
        converts a path of text items to a path of their respective int IDs. Saves space and computation time.
        :param paths: a set of paths in pure form
        :param vocabulary_map_bc: a spark broadcast of a vocabulary map.
        :return: an RDD with the transformed paths
        """
        vocabulary_map_bc = self._sc.broadcast(self._vocabmap)
        self._fitted_paths = paths \
            .map(lambda path: [vocabulary_map_bc.value[pathElement] for pathElement in path]) \
            .cache()

    def __get_cooccs_with_counts(self, window_size=3):
        """
        counts all id cooccurrences in the transformed paths in a given window size.
        finally filters the cooccurrences by a given minimum frequency to reduce noise.
        :param transformed_paths: a set of paths with ids instead of pure path elements
        :param window_size: a given window size, in which ids will be counted as cooccurring.
        :param minfrequency: a minimum frequency for cooccurrence counts, by which the final cooccurrence set is filtered.
        :return: an RDD with triples ((id1, id2), count)
        """
        min_coocc = self._min_coccc
        self._cooccs = self._fitted_paths \
            .flatMap(lambda path: [path[i: i + window_size] for i in range(len(path))]) \
            .flatMap(create_coocc_pairs) \
            .filter(lambda pair: pair[0] != pair[1]) \
            .map(lambda pair: (pair, 1)) \
            .reduceByKey(lambda a, b: a + b) \
            .filter(lambda pair_with_count: pair_with_count[1] >= min_coocc) \
            .cache()

    def __get_cooccs_with_tfidf(self, window_size=3, min_tfidf=1):
        """
        counts all id cooccurrences in the transformed paths in a given window size.
        finally filters the cooccurrences by a given minimum frequency to reduce noise.
        :param transformed_paths: a set of paths with ids instead of pure path elements
        :param window_size: a given window size, in which ids will be counted as cooccurring.
        :param min_tfidf: a minimum value for tfidf, by which the final cooccurrence set is filtered.
        :return: an RDD with triples ((id1, id2), count)
        """
        path_count = self._fitted_paths.count()
        min_coocc = self._min_coccc
        self._cooccs = self._fitted_paths.zipWithIndex() \
            .flatMap(lambda path_with_index: [(path_with_index[0][i: i + window_size], path_with_index[1]) for i in
                                              range(len(path_with_index[0]))]) \
            .map(lambda pairs_with_index: (create_coocc_pairs(pairs_with_index[0]), pairs_with_index[1])) \
            .flatMap(lambda pairs_with_index: [(k, pairs_with_index[1]) for k in pairs_with_index[0]]) \
            .filter(lambda pairs_with_index: pairs_with_index[0][0] != pairs_with_index[0][1]) \
            .groupByKey() \
            .mapValues(lambda occ_list: [k for k in occ_list]) \
            .map(lambda transition_occ_list: (
        transition_occ_list[0], len(set(transition_occ_list[1])), len(transition_occ_list[1]))) \
            .map(lambda transition_df_length: (transition_df_length[0], math.log(path_count / transition_df_length[1]),
                                               (transition_df_length[2] / transition_df_length[1]))) \
            .map(lambda transition_idf_tf: (transition_idf_tf[0], transition_idf_tf[1] * transition_idf_tf[2])) \
            .filter(lambda transition_count: transition_count[1] >= min_coocc) \
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
        return cooccs_with_count.map(
            lambda coocc_with_count: (coocc_with_count[0][0], (coocc_with_count[0][1], coocc_with_count[1]))) \
            .groupByKey() \
            .map(lambda grp: group_to_sparse_vector(grp, vector_size, binary)) \
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
        self.__replace_pages_with_ids(paths)
        if tfidf:
            self.__get_cooccs_with_tfidf(windowsize)
        else:
            self.__get_cooccs_with_counts(windowsize)
        vectors = self.__cooccs_to_vec(self._cooccs, self._vocabsize, binary)
        reverse_vocabulary_map = self._sc.broadcast({v: k for k, v in self._vocabmap.items()})
        return vectors.map(lambda node_vector: (reverse_vocabulary_map.value[node_vector[0]], node_vector[1])).collect()

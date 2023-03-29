from abc import ABC, abstractmethod
from itertools import product
from typing import List, Tuple

import numpy as np

from preprocessing import TokenizedSentencePair

EPS = 1e-16

class BaseAligner(ABC):
    """
    Describes a public interface for word alignment models.
    """

    @abstractmethod
    def fit(self, parallel_corpus: List[TokenizedSentencePair]):
        """
        Estimate alignment model parameters from a collection of parallel sentences.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
        """
        pass

    @abstractmethod
    def align(self, sentences: List[TokenizedSentencePair]) -> List[List[Tuple[int, int]]]:
        """
        Given a list of tokenized sentences, predict alignments of source and target words.

        Args:
            sentences: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            alignments: list of alignments for each sentence pair, i.e. lists of tuples (source_pos, target_pos).
            Alignment positions in sentences start from 1.
        """
        pass


class DiceAligner(BaseAligner):
    def __init__(self, num_source_words: int, num_target_words: int, threshold=0.5):
        self.cooc = np.zeros((num_source_words, num_target_words), dtype=np.uint32)
        self.dice_scores = None
        self.threshold = threshold

    def fit(self, parallel_corpus):
        for sentence in parallel_corpus:
            # use np.unique, because for a pair of words we add 1 only once for each sentence
            for source_token in np.unique(sentence.source_tokens):
                for target_token in np.unique(sentence.target_tokens):
                    self.cooc[source_token, target_token] += 1
        self.dice_scores = (2 * self.cooc.astype(np.float32) /
                            (self.cooc.sum(0, keepdims=True) + self.cooc.sum(1, keepdims=True)))

    def align(self, sentences):
        result = []
        for sentence in sentences:
            alignment = []
            for (i, source_token), (j, target_token) in product(
                    enumerate(sentence.source_tokens, 1),
                    enumerate(sentence.target_tokens, 1)):
                if self.dice_scores[source_token, target_token] > self.threshold:
                    alignment.append((i, j))
            result.append(alignment)
        return result


class WordAligner(BaseAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        self.num_source_words = num_source_words
        self.num_target_words = num_target_words
        self.translation_probs = np.full((num_source_words, num_target_words), 1 / num_target_words, dtype=np.float32)
        self.num_iters = num_iters

    def _e_step(self, parallel_corpus: List[TokenizedSentencePair]) -> List[np.array]:
        """
        Given a parallel corpus and current model parameters, get a posterior distribution over alignments for each
        sentence pair.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            posteriors: list of np.arrays with shape (src_len, target_len). posteriors[i][j][k] gives a posterior
            probability of target token k to be aligned to source token j in a sentence i.
        """
        # posteriors[p][i] = theta[src, trg] / (sum_{s in src} theta[s, trg])
        posteriors = []
        
        for sentence in parallel_corpus:
            source, target = sentence.source_tokens, sentence.target_tokens
            probs = self.translation_probs[np.ix_(source, target)]
            posteriors.append(probs / np.sum(probs, axis=0))

        return posteriors

    def _compute_elbo(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]) -> float:
        """
        Compute evidence (incomplete likelihood) lower bound for a model given data and the posterior distribution
        over latent variables.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo: the value of evidence lower bound
        """
        elbo = 0.
         
        for i, sentence in enumerate(parallel_corpus):
            source, target = sentence.source_tokens, sentence.target_tokens
            m, n = len(target), len(source)
            probs = self.translation_probs[np.ix_(source, target)]
            elbo += np.sum(posteriors[i] * (np.log(probs + EPS) - np.log(posteriors[i] + EPS))) - m * np.log(n)
        
        return elbo
                     
    def _m_step(self, parallel_corpus: List[TokenizedSentencePair], posteriors: List[np.array]):
        """
        Update model parameters from a parallel corpus and posterior alignment distribution. Also, compute and return
        evidence lower bound after updating the parameters for logging purposes.

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices
            posteriors: posterior alignment probabilities for parallel sentence pairs (see WordAligner._e_step).

        Returns:
            elbo:  the value of evidence lower bound after applying parameter updates
        """
        self.translation_probs[:, :] = 0.                   
                           
        for i, sentence in enumerate(parallel_corpus):
            source, target = sentence.source_tokens, sentence.target_tokens
            np.add.at(self.translation_probs, np.ix_(source, target), posteriors[i])
        
        self.translation_probs /= np.sum(self.translation_probs, axis=1, keepdims=True)
                           
        return self._compute_elbo(parallel_corpus, posteriors)

                           
    def fit(self, parallel_corpus):
        """
        Same as in the base class, but keep track of ELBO values to make sure that they are non-decreasing.
        Sorry for not sticking to my own interface ;)

        Args:
            parallel_corpus: list of sentences with translations, given as numpy arrays of vocabulary indices

        Returns:
            history: values of ELBO after each EM-step
        """
        history = []
        for i in range(self.num_iters):
            posteriors = self._e_step(parallel_corpus)
            elbo = self._m_step(parallel_corpus, posteriors)
            history.append(elbo)
        return history

    
    def align(self, sentences):
        alignments = []
        
        for sentence in sentences:
            source, target = sentence.source_tokens, sentence.target_tokens
            aligned_source_tokens = np.argmax(self.translation_probs[np.ix_(source, target)], axis=0)
            alignments.append([(j + 1, i + 1) for i, j in enumerate(aligned_source_tokens)])
    
        return alignments


class WordPositionAligner(WordAligner):
    def __init__(self, num_source_words, num_target_words, num_iters):
        super().__init__(num_source_words, num_target_words, num_iters)
        self.alignment_probs = {}

    def _get_probs_for_lengths(self, src_length: int, tgt_length: int):
        """
        Given lengths of a source sentence and its translation, return the parameters of a "prior" distribution over
        alignment positions for these lengths. If these parameters are not initialized yet, first initialize
        them with a uniform distribution.

        Args:
            src_length: length of a source sentence
            tgt_length: length of a target sentence

        Returns:
            probs_for_lengths: np.array with shape (src_length, tgt_length)
        """
        probs = self.alignment_probs.get((src_length, tgt_length), None)
        
        if probs is None:
            self.alignment_probs[(src_length, tgt_length)] = \
                np.full((src_length, tgt_length), 1 / src_length, dtype=np.float32)
            probs = self.alignment_probs[(src_length, tgt_length)]
        
        return probs 

    def _e_step(self, parallel_corpus):
        posteriors = []
        
        for sentence in parallel_corpus:
            source, target = sentence.source_tokens, sentence.target_tokens
            alignment_probs = self._get_probs_for_lengths(len(source), len(target))
            probs = self.translation_probs[np.ix_(source, target)] * alignment_probs
            posteriors.append(probs / np.sum(probs, axis=0))

        return posteriors

    def _compute_elbo(self, parallel_corpus, posteriors):
        elbo = 0.
         
        for i, sentence in enumerate(parallel_corpus):
            source, target = sentence.source_tokens, sentence.target_tokens
            
            translation_probs = self.translation_probs[np.ix_(source, target)]
            alignment_probs = self._get_probs_for_lengths(len(source), len(target))
            
            elbo += np.sum(posteriors[i] * (np.log(translation_probs + EPS) + 
                                            np.log(alignment_probs + EPS) - 
                                            np.log(posteriors[i] + EPS)))
        
        return elbo
    
    def _m_step(self, parallel_corpus, posteriors):
        self.translation_probs[:, :] = 0.          
        
        for key in self.alignment_probs.keys():
            self.alignment_probs[key][:, :] = 0.
                           
        for i, sentence in enumerate(parallel_corpus):
            source, target = sentence.source_tokens, sentence.target_tokens
            np.add.at(self.translation_probs, np.ix_(source, target), posteriors[i])
            
            if (len(source), len(target)) not in self.alignment_probs:
                self.alignment_probs[(len(source), len(target))] = np.zeros((len(source), len(target)))
                
            self.alignment_probs[(len(source), len(target))] += posteriors[i]
        
        self.translation_probs /= np.sum(self.translation_probs, axis=1, keepdims=True) # row-wise
           
        for key in self.alignment_probs.keys():
            self.alignment_probs[key] /= np.sum(self.alignment_probs[key], axis=0) # column-wise
                           
        return self._compute_elbo(parallel_corpus, posteriors)

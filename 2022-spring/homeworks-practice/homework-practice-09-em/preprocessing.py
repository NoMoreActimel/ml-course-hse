from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """
    sentence_pairs, alignments = [], []
    
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('<s '):
                source_sentence, target_sentence = [], []
                source_alignment, target_alignment = [], []
                
                for i, lst in enumerate([source_sentence, target_sentence, source_alignment, target_alignment]):
                    line = file.readline()
                    start, end = line.find('>') + 1, line.find('</')
                    tokens = line[start:end].split()
                    
                    for token in tokens:
                        if i < 2:
                            lst.append(token)
                        else:
                            delimeter = token.find('-')
                            lst.append((int(token[:delimeter]), int(token[delimeter + 1:])))
                
                sentence_pairs.append(SentencePair(source_sentence, target_sentence))
                alignments.append(LabeledAlignment(source_alignment, target_alignment))
    
    return sentence_pairs, alignments
       

def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    def process_sentence_to_dicts(sentence: List[str], token_index_dict: dict, freq_dict: defaultdict):
        """
        Adds token-index pairs into first dictionary and changes frequency dictionary
        """
        for token in sentence:
            freq_dict[token] += 1
            if token not in token_index_dict:
                token_index_dict[token] = len(token_index_dict)
                
        return
    
    
    source_dict, target_dict = {}, {}
    source_freq_dict, target_freq_dict = defaultdict(int), defaultdict(int)
    
    for sentence_pair in sentence_pairs:
        process_sentence_to_dicts(sentence_pair.source, source_dict, source_freq_dict)
        process_sentence_to_dicts(sentence_pair.target, target_dict, target_freq_dict)
            
    if freq_cutoff is not None:
        sorted_source_freq = sorted(source_freq_dict.items(), key=lambda x: (-x[1], x[0]))[:freq_cutoff]
        most_freq_source_tokens = set(token for token, freq in sorted_source_freq)
        
        sorted_target_freq = sorted(target_freq_dict.items(), key=lambda x: (-x[1], x[0]))[:freq_cutoff]
        most_freq_target_tokens = set(token for token, freq in sorted_target_freq)

        source_dict = {token: index for token, index in source_dict.items() if token in most_freq_source_tokens}
        target_dict = {token: index for token, index in target_dict.items() if token in most_freq_target_tokens}
        
    return source_dict, target_dict
        

def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """
    def tokenize_sentence(sentence, dictionary):
        """
        Returns sentence tokenized by given dictionary or None if at least one word was missing
        """
        tokenized = []
        
        for token in sentence:
            index = dictionary.get(token, None)
            if index is None: 
                return None
            tokenized.append(index)
            
        return np.array(tokenized)
    
    
    tokenized_sentence_pairs = []
    
    for sentence_pair in sentence_pairs:
        in_corpus = True
        
        tokenized_source_sentence = tokenize_sentence(sentence_pair.source, source_dict)
        tokenized_target_sentence = tokenize_sentence(sentence_pair.target, target_dict)
        
        if tokenized_source_sentence is None or tokenized_target_sentence is None:
            continue
        
        tokenized_sentence_pair = TokenizedSentencePair(tokenized_source_sentence, tokenized_target_sentence)
        tokenized_sentence_pairs.append(tokenized_sentence_pair)
    
    return tokenized_sentence_pairs

"""
Provide tools to support extractive summarisation of textual data
- Select sentences of relevance
[Unsupervised]
    - TextRank (inherited from PageRank), this method encapsulates semantics better than tf-idf
    - Tfidf (weighted word -> overall sentence weight)

TODO:
Provide tools to support abstractive summarisation
[Supervised]
    - Using BERT
    - Memory Networks
    - To expand T5 configurations within written classes within; pass over of T5's native arguments
    - Controllable Networks

[Unsupervised]
    - Latent Dirichlet Process
    - Simple k-means
    
Use of Python logger to record events for debugging

Consider pushing all model loading during instantiation of objects

"""

import re
import collections
import numpy as np
import pandas as pd
import spacy
spacy.prefer_gpu()
import scipy
import pytextrank
import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Union, Iterable, Tuple

class ExtractiveTextReducer:
    def __init__(self):
        """
        Object class to reduce amount of data in text documents.
        Undergo reduction of text to look at using abstractive and extractive summarisation methods.
        
        """
        self.__corpus__ = None

    def fit(self,
        corpus: Union[str, Iterable[str]],
        sent_delim: str='\.\s+|\r|\n'):
        """
        Fit extractive text reducer object on corpus.
        
        It is reccommended that preprocessing has taken place before loading directly into text summariser in iterable format level to avoid inconsistencies due to document delimitation from the puctuations.
        
        @corpus: Union[str, Iterable[str]]
            Collection of texts; sentences by sentences. If single string object is passed, it will be split into list of strings (sentences).
        
        @sent_delim: str
            Regex pattern to identify sentences within corpus and split into list if corpus input is string.
        
        """
        if type(corpus) == str:
            self.__corpus__ = [sent+'.' if ('\.' in sent_delim and sent[-1] != '.') else sent 
                for sent in re.split(sent_delim, corpus) ]
        elif isinstance(corpus, Iterable):
            self.__corpus__ = corpus
        else:
            raise Exception('[WARN] Invalid corpus input supplied!')
            
    def tfidf_score(self,
        scr_mech: str='mean',
        top_ptile_return: float=75., 
        **kwarg) -> Tuple[scipy.sparse.csr.csr_matrix, collections.OrderedDict]:
        """
        Vectorise a corpus with tf-idf and find overall document importance through aggregated scores
        
        @scr_mech: str
            Method of aggregating the score of the sentence tf-idf matrix
            Accepts ['mean', 'median', 'sum'] which implements functions by numpy (i.e. np.mean); function is applied column-wise
            
        @top_ptile_return: float
            Returns the top Nth-percentile of sentences, accepts ranges [0.0, 100.0]
            Value of 75.0 returns 75th percentile rank and above (i.e. top 25%)
        
        @kwarg
            To be passed on to object class <sklearn.feature_extraction.text TfidfVectorizer>
        
        ---
        Returns scipy sparse tf-idf matrix and ordered dictionary of top Nth aggregated tf-idf scores (key) to sentences (value). Top Nth is determined by input specified to `top_ptile_return`.
        
        """
        if not (0. <= top_ptile_return <= 100.):
            raise Exception('[WARN] You have specified invalid percentile range!')
        
        try:
            # TODO: extend to custom agg. function
            assert scr_mech in ['mean', 'median', 'sum']
            if scr_mech == 'mean':
                scr_mech = np.mean
            elif scr_mech == 'median':
                scr_mech = np.median
            else:
                scr_mech = np.sum
                
        except Exception as e:
            print('[WARN] You have specified a non-supported document scoring aggregator mechanism!')
            raise Exception(e)
        
        vectorizer = TfidfVectorizer(**kwarg)
        X = vectorizer.fit_transform(np.array(self.__corpus__)) # enforce into numpy array
        sent_impt = {scr_mech(array): sent for array, sent in zip(X.toarray(), self.__corpus__) }
        score_cutoff_pt = np.percentile(list(sent_impt.keys()), top_ptile_return)
        sent_impt = {score: sent for score, sent in sent_impt.items() if score >= score_cutoff_pt}
        sent_impt = collections.OrderedDict(sorted(sent_impt.items(), reverse=True))
        
        return X, sent_impt
        
    def textrank_score(self,
        preferred_spacy_core: str='en_core_web_sm',
        tr_phrase_lim: int=15,
        top_ptile_return: float=75. ) -> Tuple[dict, dict]:
        """
        Perform textrank to sieve out phrases of importance across all documents.
        Build sentence importance rank by the composition of these phrases within sentence.
            i.e. more important phrases within sentence (doc), more important sentence
        
        @preferred_spacy_core: str
            Language mdoel core of spacy downloadable with i.e. `python -m spacy download en_core_web_sm`
        
        @tr_phrase_lim: int
            Number of phrases at maximum to account for when determinng importance of a sentence
            
        @top_ptile_return: float
            Returns the top Nth-percentile of sentences, accepts ranges [0.0, 100.0]
            Value of 75.0 returns 75th percentile rank and above (i.e. top 25%)
            
        ---
        Returns dictionary of phrase importance (by eigencentrality) across corpus and dictionary of top Nth sentence rank importance (key) to sentences (value). Top Nth is determined by input specified to `top_ptile_return`.
        
        References: https://github.com/DerwenAI/pytextrank
        
        """
        if not (0. <= top_ptile_return <= 100.):
            raise Exception('[WARN] You have specified invalid percentile range!')
        
        try:
            assert preferred_spacy_core in ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg']
        except Exception as e:
            print('[WARN] You have specified a non-english spacy language core model!')
            raise Exception(e)
        
        nlp = spacy.load(preferred_spacy_core)
            # default spacy pipeline: tagger -> parser -> ner
            # see https://spacy.io/usage/processing-pipelines
        
        tr = pytextrank.TextRank()
        nlp.add_pipe(tr.PipelineComponent, name='textrank', last=True)
            
        doc = nlp('. '.join(self.__corpus__))
        
        phrase_impt = { phrase.text : {
            'eigenc_scr' : phrase.rank,
            'count' : phrase.count
            }
            for phrase in doc._.phrases
        }
        
        # Presumably first sentence returned is most important from `pytextrank` source code
        sent_impt = {rank+1: str(sent)[:-1]
            for rank, sent in enumerate(
                doc._.textrank.summary(limit_phrases=tr_phrase_lim, limit_sentences=len(self.__corpus__))
            )   
        }
            # note Python 3.8 gives ordered dict by default
            # pytextrank will auto append period char `.` regardless it ends with a period or not

        rank_cutoff_pt = np.percentile(list(sent_impt.keys()), 100-top_ptile_return)
            # invert ptile. as rank 1 == "100th" percentile
        sent_impt = {rank: sent.replace('..', '.') for rank, sent in sent_impt.items() if rank <= rank_cutoff_pt}
            # `<=` because rank 1 is "100th" percentile

        return phrase_impt, sent_impt

    

class AbstractiveTextReducer:
    """
    TODO, network architectures to be used for abstractive text summarisation
        [1] BERT - SQuAD dataset, transfer learning (KIV)
        [2] seq2seq
        [3] Memory Networks
        [4] Controllable Networks
    
    """
    def __init__(self):
        """
        Object class to reduce amount of data in text documents.
        Undergo reduction of text to look at using abstractive and extractive summarisation methods.
            
        """
        self.__corpus__ = None

    def fit(self,
        corpus: Union[str, Iterable[str]],
        sent_delim: str='\.\s+|\r|\n') -> None:
        """
        Fit extractive text reducer object on corpus.
        
        It is reccommended that preprocessing has taken place before loading directly into text summariser in iterable format level to avoid inconsistencies due to document delimitation from the puctuations.
        
        @corpus: Union[str, Iterable[str]]
            Collection of texts; sentences by sentences. If single string object is passed, it will be split into list of strings (sentences).
        
        @sent_delim: str
            Regex pattern to identify sentences within corpus and split into list if corpus input is string.
            
        """
        if type(corpus) == str:
            self.__corpus__ = [sent+'.' if ('\.' in sent_delim and sent[-1] != '.') else sent 
                for sent in re.split(sent_delim, corpus) ]
        elif isinstance(corpus, Iterable):
            self.__corpus__ = corpus
        else:
            raise Exception('[WARN] Invalid corpus input supplied!')
            
    def t5_summarise(self,
        pretrain_size: str='t5-small',
        cmpt_device: str='cpu',
        hard_replace_pronouns: dict=dict([]),
        **t5_mdl_params) -> str:
        """
        Use of Google's/Hugging Face T5 architecture to perform text summarisation.
        
        @pretrain_size: str
            Language model core of T5 architecture, defaults to using small pretrained neural network weights i.e. "t5-small". Refer to original documentation for model list @ https://huggingface.co/models
        
        @cmpt_device: str
            Compute device to either be backed by system GPU or CPU, defaults using CPU to perform computation.
        
        @hard_replace_pronouns: dict
            Used to perform regular expression string replace for purpose of disambiguating co-references. By default, empty dictionary is set and no replacement will be initiated. This replacement ignores case in regexp.. Contents in dictionary should be organised as such;
                
                {'my_regexp_pattern_str': 'replacement_str'}
            
            Considering abstractive / neural-networks like BERT can account for context (thus implied coreferences), it is best that raw texts are parsed instead of pre-processing with coreference resolution.
        
        @**t5_mdl_params:
            Model hyperparameters to be passed on to the T5 class model object
       
        ---
        Returns a continuous text abstractive summary. This may be composed of more than 1 sentences returned
        
        """
        model = T5ForConditionalGeneration.from_pretrained(pretrain_size)
        tokenizer = T5Tokenizer.from_pretrained(pretrain_size)
        device = torch.device(cmpt_device)
        
        preprocess_text = [txt.strip()+"." if txt.strip()[-1]!="." else txt.strip() for txt in self.__corpus__]
        preprocess_text = "summarize: " + " ".join(preprocess_text)
        tokenized_text = tokenizer.encode(preprocess_text, return_tensors="pt").to(device)
        
        # TODO: Expand configurability of T5 package within method
        # t5_mdl_params = {"pretrain_size":'t5-large',
            # "num_beams":4,
            # "no_repeat_ngram_size":2,
            # "min_length":30,
            # "max_length":100,
            # "early_stopping":True}
        summary_ids = model.generate(tokenized_text, **t5_mdl_params)
        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        if not hard_replace_pronouns:
            pass # nothing to replace
        else:
            for pattern, replacement in hard_replace_pronouns.items():
                output = re.sub(pattern, replacement, output, flags=re.IGNORECASE)
                
        return output


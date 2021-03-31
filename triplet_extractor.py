"""
Primarily leverage StanfordNLP (Python package `openie`) to run the extraction of relation triplets. Behind the package,
grammar rules are in-built to deduce dependancy trees to extract subject-predicate-object.

To reduce potentially (likely) cases of relation triplet duplication, word-mover distance is employed to find
similarity between the document sentences. This uses Gensim's package providing interface with various word vector types.
Facebook's FastText is employed here.
    - Gensim word vectors and data reference available at: https://github.com/RaRe-Technologies/gensim-data

TODO
1. Storing of triplets into NetworkX graphs
2. Representation using NetworkX
- For lower overhead/shuffles between different applications
- All in Python, programmatically easier to maintain/faster?

Use of Python logger to record events for debugging

Consider pushing all model loading during instantiation of objects

"""

import re
from typing import Union, Iterable, Tuple
import pandas as pd
import numpy as np
import spacy
spacy.prefer_gpu()
import gensim
import gensim.downloader as api
from gensim.models.fasttext import FastText
from gensim.parsing.preprocessing import STOPWORDS
from openie import StanfordOpenIE
import openie5_client

from helperutils import *

def levenshtein(s1, s2):
    """
    First Python implementation method of the Levenshtein distance between strings
    Credits: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def wmd_matrix(
    triplet_list: Iterable[dict],
    model: gensim.models.keyedvectors.Word2VecKeyedVectors) -> dict and Iterable[Iterable[float]]:
    """
    Use Gensim package to find relation triplet similarities between N-triplets based on word-mover distance.
    
    FastText word-embedding/architecture is employed with details:
        - pretrained embedding model: fasttext-wiki-news-subwords-300
        - WMD(s1, s2) == 0 imply s1 identical to s2
        
    For downstream grouping triplets of similarity, the values are inverted to range 0 <= X <= 1;
        where WMD == 1 imply identicality
    This is for the purpose of agglomerative clustering; conform to method employed in scikit-learn documentation
        @https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
    
    @triplet_dict: list/Iterable
        List of relation-triplets, each element a dictionary format. Keys of triplets should conform to format:
            [
             {
              'subject': 'apple',
              'relation': 'is a',
              'object': 'fruit'
             },
             {
              'subject':'fish',
              'relation':'live in the'
              'object':'ocean'
             }
            ]
            
    @model: gensim.models.keyedvectors.Word2VecKeyedVectors
        Gensim keyed-vector Word2Vec model object preloaded with word-embedding;
        Embedding does not necessarily have to be Word2Vec
    
    ---
    Returns indexed relation-triplet dictionary and adjusted N x N numpy array of WMD. 
    
    """
    
    # Concatenate subj-pred-obj into single sentence document, indexed
    triplets_idx = {idx: trip for idx, trip in enumerate(triplet_list)}
    
    triplets = {idx:re.sub('\s+', ' ', ' '.join([trip['subject'],
                                             trip['relation'],
                                             trip['object']])
                          ).strip()
                for idx, trip in triplets_idx.items() }
    triplets = {idx:[w for w in sent if w not in STOPWORDS] for idx, sent in triplets.items()}
    
    # Build similarity maxtrix
    wmd_array = []
    for idx in range(len(triplets)): # iterate using range; dict may be unordered
        current_str = triplets[idx]
        wmd_array.append([model.wmdistance(current_str, triplets[inr_idx])
                          for inr_idx in range(len(triplets))])
    wmd_array = np.array(wmd_array)
    
    max_d = np.max(wmd_array)
    min_d = np.min(wmd_array)
    scale = lambda x: (((x - min_d) / (max_d - min_d)) - 1.) * -1.
    
    wmd_array = np.array([
        [scale(i) for i in row]
        for row in wmd_array
    ])
        
    return triplets_idx, wmd_array



class RelationExtractorTools:
    def __init__(self):
        """
        Leverage Python-wrapper to Stanford CoreNLP Java engine for purpose of relation-triplet extractions.
        
        """
        # Initialise class attributes (visibility ease)
        self.__corpus__ = None
        self.__pron_det_pos_words__ = None
        self.__triples_corpus__ = None
        self.__entities_in_doc__ = None
        self.__wvmodel__ = None
        
        # For purpose of parsing relation triplets later
        # Load pretrained embedding model
        #plog('Loading pretrained word embeddings. This will take some time to load...')
        #self.__wvmodel__ = api.load('fasttext-wiki-news-subwords-300')
        #plog('Pretrained word embeddings loaded!')
        
    def fit(self,
        corpus: Union[str, Iterable[str]], 
        sent_delim: str='\.\s+|\r|\n',
        preferred_spacy_core: str='en_core_web_sm'
        ) -> None:
        """
        It is reccommended that preprocessing has taken place before loading directly into object class in iterable format level to avoid inconsistencies due to document delimitation from the puctuations.
        
        Upon calling method `.fit()` on corpus, collection sets on entities and PoS will be identified to harmonise the extracted relation triplets. This may take some computation time. Note that this uses spaCy's pre-trained models which is fitted on Onto Notes 5 (chargeable to access the raw training data!)
        
        @corpus: Union[str, Iterable[str]]
            Collection of texts; sentences by sentences. If single string object is passed, it will be split into list of strings (sentences).
        
        @sent_delim: str
            Regex pattern to identify sentences within corpus and split into list if corpus input is string.
            
        @preferred_spacy_core: str
            Language mdoel core of spacy downloadable with i.e. `python -m spacy download en_core_web_sm`        
        
        """
        # Initialise corpus
        if type(corpus) == str:
            self.__corpus__ = [sent+'.' if ('\.' in sent_delim and sent[-1] != '.') else sent 
                for sent in re.split(sent_delim, corpus) ]
        elif isinstance(corpus, Iterable):
            self.__corpus__ = corpus
        else:
            raise Exception('[WARN] Invalid corpus input supplied!')
            
        ## Collect pronoun-variants and determinants
        nlp = spacy.load(preferred_spacy_core)
        self.__pron_det_pos_words__ = {token.text for doc in nlp.pipe(self.__corpus__, disable=['parser', 'ner'])
                                       for token in doc if token.pos_ in ['PRON', 'DET']}
        
        ## Collect recognised entities
            # Default NER scheme: Onto Notes 5
            # TODO: integration of pre/re-trainng modules for larger set of recognised entities
            # N.B.: temp. disabled functionality to clean triplets via NER
        self.__entities_in_doc__ = {(ent.text, ent.label_)
                                       for doc in nlp.pipe(self.__corpus__, disable=['tagger', 'parser'])
                                           for ent in doc.ents}
        self.__entities_in_doc__  = pd.DataFrame(self.__entities_in_doc__, columns=['entities', 'ner_label'])
        self.__entities_in_doc__['xjoin'] = 1
        
    def extract_triplets(self) -> Iterable[dict]:
        """
        Starts a java server powering CoreNLP backend to obtain triplets.

        Sentences (documents) will be joined with a `\s` to form a continuous string for purpose of triplet-text annotation.

        ---
        Returns list relation-triplets dictionary in following format:
            [
             {'subject': `subj_txt`,
              'relation': `rel_txt`,
              'object': `obj_txt` },
              ...
            ]

        """
        stg_corpus = [txt.strip()+"." if txt.strip()[-1]!="." else txt.strip() for txt in self.__corpus__]
        stg_corpus = ' '.join(self.__corpus__)

        with StanfordOpenIE() as client:
            triples_corpus = client.annotate(stg_corpus)

        self.__triples_corpus__ = triples_corpus

        return triples_corpus
            
    def better_extract_triplets_prototype(self, port='8000', **kwargs) -> Iterable[dict]:
        import openie5_client
        
        extractions = []
        with openie5_client.OpenIEClient(port=port, **kwargs) as extractor:
            for sentence in self.__corpus__:
                extractions.extend(extractor.extract(sentence))
            
        triples_corpus = []
        
        for triplet_set in extractions:
            sentence = triplet_set['sentence']
            data = triplet_set['extraction'] # access extraction key
            arg1 = data['arg1']['text']
            rel = data['rel']['text']
            triplet = {}

            # create triplets
            for arg2 in data['arg2s']:
                triplet['subject'] = arg1
                triplet['relation'] = rel
                triplet['object'] = arg2['text']
                triplet['sentence'] = sentence
                triples_corpus.append(triplet.copy())

        self.__triples_corpus__ = triples_corpus
        #plog("Triplets extracted.")
                
        return triples_corpus
        
    
    def parse_triplets(self,
        levenshtein_thold: float=20.,
        coph_scr: float=2.) -> Iterable[dict]:
        """
        Parse relation triplets over the following conditions
            1. Remove triplets with pronouns and determinants in subj/obj; i.e. "we", "she" "I", "their", etc.
            2. Harmonise duplicated triplets, return only the superset triplet
                - Semantic comparison option using word mover distance & agglomerative clustering
                - FastText via Gensim W2V keyed-vector architecture format; partial overcome OOV issues
            3. Remove triplets with no entities either in the subject or object // [N.B.! KIV; function temporarily removed]
                - Reference Onto Notes 5 entities @https://spacy.io/api/annotation#named-entities
                - Note hard match is being used, case insensitive;
                    - KIV: Matching using `SentencePiece` over hard-matches
                - Noun chunk needs to be a subset of the recognised entities during class instantiation
        
        @levenshtein_thold: float
            Maximum percentile of Levenshtein distance to consider a subject/object noun chunk similar to known NER.
            // [N.B.! KIV; function temporarily removed]
            // Values to this argument have no effect on method
        
        @coph_scr: float
            Cophenetic distance to determine size of grouped relation triplets. Low values reduce ability to remove duplicates.

        ---
        Returns list (Iterable) of dictionary containing relation triplets
        
        ---
        Notes:
        [N.B.! KIV; function temporarily removed] Super/subsets of NER will be filtered out due to Levensthein distance used as yardstick
            - i.e. "The green public bus" vs. "bus" -> thrown out
            
        """
        
        # Remove pronoun and determiners
        parse_triples = [triple for triple in self.__triples_corpus__
                         if (triple['subject'] not in self.__pron_det_pos_words__ and
                             triple['object'] not in self.__pron_det_pos_words__ ) ]
        
        # Harmonise potentially duplicative triplets by constructing matrix of Word Mover Distances
        stg_triples_idx, wmd_array = wmd_matrix(parse_triples, self.__wvmodel__)
        stg_triples_idx_grp = get_similarity_repr(wmd_array, cophenetic_dist=coph_scr, grouped_idx=True)
            # list of triplets' indices-lists
            # i.e. [ [1,5,7], [2], [9,4,3] ]
        
        ## Retrieve longest relation-triplet strings in each group
        stg_triples_idx_len = [
            [len(re.sub('\s+', ' ', ' '.join([stg_triples_idx[trip]['subject'], 
                                              stg_triples_idx[trip]['relation'], 
                                              stg_triples_idx[trip]['object']])
                       ).strip()
                )
             for trip in trip_grp]
            for trip_grp in stg_triples_idx_grp]
        
        stg_triples_selected = []
        for trip_idx, trip_len in zip(stg_triples_idx_grp, stg_triples_idx_len):
            group_max_len = max(trip_len)
            idx_max_len = [trip_idx[pxtn_idx] for pxtn_idx, str_len in enumerate(trip_len)
                if str_len==group_max_len][0] # first is position is retrieved if tied
            stg_triples_selected.append(idx_max_len)
        
        parse_triples = [triple for idx, triple in stg_triples_idx.items() if idx in stg_triples_selected]
        
#         # Find triples of subject/object near matching identified collection of NER
#         stg_triples = [(triple['subject'], triple['relation'], triple['object']) for triple in parse_triples]
#         stg_triples = pd.DataFrame(stg_triples, columns=['subject', 'relation', 'object'])
#         stg_triples['xjoin'] = 1
#         stg_triples = stg_triples.merge(self.__entities_in_doc__, on='xjoin').drop(columns='xjoin')
        
#         stg_triples['subj_ent_leven'] = stg_triples[['subject', 'entities']]\
#         .apply(lambda row: levenshtein(row['subject'], row['entities']), axis=1)
#         stg_triples['obj_ent_leven'] = stg_triples[['object', 'entities']]\
#         .apply(lambda row: levenshtein(row['object'], row['entities']), axis=1) # consider subj/obj in ent over leven.
        
#         subj_ent_leven_thold_ptile = np.percentile(stg_triples['subj_ent_leven'].values, levenshtein_thold)
#         obj_ent_leven_thold_ptile = np.percentile(stg_triples['obj_ent_leven'].values, levenshtein_thold)
#         subj_obj_similar_ent_mask = (stg_triples['subj_ent_leven'] <= subj_ent_leven_thold_ptile) & \
#             (stg_triples['obj_ent_leven'] <= obj_ent_leven_thold_ptile)
        
#         stg_triples = stg_triples.loc[subj_obj_similar_ent_mask, ['subject', 'relation', 'object']].values.T
#         parse_triples = [{'subject':subj, 'relation':rel, 'object':obj}
#             for subj, rel, obj in zip(stg_triples[0], stg_triples[1], stg_triples[2])] # revert to original list[dict]
        
        
        return parse_triples
    
    
    
        














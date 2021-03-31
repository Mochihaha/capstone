"""
TODO:
Provide tools to support loading of various text formats
    - PDF format

Provide tools to support trivial-cleaning of text data
    - Paragraphs -> Sentences -> as "Docs"
    - Stop words
    - HTML tags
    - Accented characters
    - Punctuation

Provide tools to resolve linguistic-related issues
    1. Coreference resolution with spaCy extension neuralcoref
        [Associated issues with installing]
        - unable to detect pytorch -> run conda install -c pytorch pytorch
            - Reference: https://medium.com/@valeryyakovlev/anaconda-no-module-named-torch-ead10946de66
        - unable to detect Microsoft Visual C++ 14.0 when already installed in Windows 10
            - notably occurred when installing spaCy extension `neuralcoref`
            - Add directory containing `` to environment path
            - i.e. `C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC` add to PATH in Environmental variables
                - This PC (right-click) > Properties > Advanced System Settings > Environment Variables
                - Add to `PATH` in both system and user
            - Reference: https://stackoverflow.com/questions/29846087/microsoft-visual-c-14-0-is-required-unable-to-find-vcvarsall-bat
        - incompatible `neuralcoref` & `spaCy` packages
            - Reference: https://github.com/huggingface/neuralcoref/issues/222
            - install `neuralcoref` through pulling from its GitHub first before installing `spaCy`, else kernel hangs
            
    2. Word contractions

Use of Python logger to record events for debugging

Consider pushing all model loading during instantiation of objects

"""

import re
import spacy
spacy.prefer_gpu()
import neuralcoref
from typing import Union, Iterable, Tuple

class ProcessorTools:
    def __init__(self):
        """
        Leverage Python-wrapper to Stanford CoreNLP Java engine for purpose of relation-triplet extractions.

        """
        self.__corpus__ = None

    def fit(self,
        corpus: Union[str, Iterable[str]], 
        sent_delim: str='\.\s+|\r|\n'
        ) -> None:
        """
        Fit processor object on corpus.
        
        It is reccommended that preprocessing has taken place before loading directly into object class in iterable format level to avoid inconsistencies due to document delimitation from the puctuations.
        
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
        
    def coref_resolve(self,
        preferred_spacy_core: str='en_core_web_sm',
        **neuralcoref_kwargs) -> Tuple[spacy.tokens.doc.Doc, str]:
        """
        Perform coreference resolution of texts. Especially relevant when text are active voices that have extensive usages of pronouns like "he", "we", "I" and etc.
        
        `neuralcoref` extends spaCy's NLP pipeline model and is appended to the end of pipeline by construct.
        
        @preferred_spacy_core: str
            Language mdoel core of spacy downloadable with i.e. `python -m spacy download en_core_web_sm`
            
        @**neuralcoref_kwargs
            Passes parameters on to `neuralcoref.NeuralCoref` method. Refer more at https://github.com/huggingface/neuralcoref
        
        ---
        Returns fitted document on spaCy pipeline and coreference resolved text. Returned text may be a series of sentences chained together that forms entire string object.
        
        Attribute `self.__corpus__` (list of sentences) will be re-concatenated into a single corpus, string object.
        
        """
        nlp = spacy.load(preferred_spacy_core)
        coref = neuralcoref.NeuralCoref(nlp.vocab, **neuralcoref_kwargs)
        nlp.add_pipe(coref, name='neuralcoref', last=True)
            # spaCy default pipeline: tagger (PoS) >>> (Dep.) parser >>> ner
        
        doc_corpus = ' '.join([txt.strip()+'.' if txt.strip()[-1] != '.' else txt.strip()
                               for txt in self.__corpus__])
        
        doc = nlp(doc_corpus)
        
        return doc, doc._.coref_resolved




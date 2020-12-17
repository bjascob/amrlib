from abc import ABC, abstractmethod


# Abstract base class for sentence-to-graph (aka parse) models)
# This is here simply to define a common interface for all STOG type models
class STOGInferenceBase(ABC):
    @abstractmethod
    def __init__(self, model_dir, model_fn, **kwargs):
        pass

    # parse a list of sentences (strings) to AMR graph strings
    # add_metadata causes meta-data such as 'snt' and 'tokens' to be added to the returned file
    # return a list of AMR graphs (string format)
    @abstractmethod
    def parse_sents(self, sents, add_metadata=True):
        pass

    # parse a list of spacy spans (ie.. span has list of tokens)
    # add_metadata causes meta-data such as 'snt' and 'tokens' to be added to the returned file
    # Similar to above code but SpaCy parsing is already done spans is a spacy object
    @abstractmethod
    def parse_spans(self, spans, add_metadata=True):
        pass



# Abstract base class for sentence-to-graph (aka parse) models)
# This is here simply to define a common interface for all STOG type models
class GTOSInferenceBase(ABC):
    @abstractmethod
    def __init__(self, model_dir, model_fn, **kwargs):
        pass

    # Generate sentences from a list of AMR text graphs
    @abstractmethod
    def generate(self, graphs):
        pass

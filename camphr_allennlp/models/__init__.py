"""
These submodules contain the classes for AllenNLP models,
all of which are subclasses of `Model`.
"""

from camphr_allennlp.models.model import Model
from camphr_allennlp.models.archival import archive_model, load_archive, Archive
from camphr_allennlp.models.simple_tagger import SimpleTagger
from camphr_allennlp.models.basic_classifier import BasicClassifier

"""
A :class:`~allennlp.data.fields.field.Field` is some piece of data instance
that ends up as an array in a model.
"""

from camphr_allennlp.data.fields.field import Field
from camphr_allennlp.data.fields.adjacency_field import AdjacencyField
from camphr_allennlp.data.fields.array_field import ArrayField
from camphr_allennlp.data.fields.flag_field import FlagField
from camphr_allennlp.data.fields.index_field import IndexField
from camphr_allennlp.data.fields.label_field import LabelField
from camphr_allennlp.data.fields.list_field import ListField
from camphr_allennlp.data.fields.metadata_field import MetadataField
from camphr_allennlp.data.fields.multilabel_field import MultiLabelField
from camphr_allennlp.data.fields.namespace_swapping_field import NamespaceSwappingField
from camphr_allennlp.data.fields.sequence_field import SequenceField
from camphr_allennlp.data.fields.sequence_label_field import SequenceLabelField
from camphr_allennlp.data.fields.span_field import SpanField
from camphr_allennlp.data.fields.text_field import TextField

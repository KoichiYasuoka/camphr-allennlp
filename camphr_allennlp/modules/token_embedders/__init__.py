"""
A `TokenEmbedder` is a `Module` that
embeds one-hot-encoded tokens as vectors.
"""

from camphr_allennlp.modules.token_embedders.token_embedder import TokenEmbedder
from camphr_allennlp.modules.token_embedders.embedding import Embedding
from camphr_allennlp.modules.token_embedders.token_characters_encoder import TokenCharactersEncoder
from camphr_allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from camphr_allennlp.modules.token_embedders.empty_embedder import EmptyEmbedder
from camphr_allennlp.modules.token_embedders.bag_of_word_counts_token_embedder import (
    BagOfWordCountsTokenEmbedder,
)
from camphr_allennlp.modules.token_embedders.pass_through_token_embedder import PassThroughTokenEmbedder
from camphr_allennlp.modules.token_embedders.pretrained_transformer_embedder import (
    PretrainedTransformerEmbedder,
)
from camphr_allennlp.modules.token_embedders.pretrained_transformer_mismatched_embedder import (
    PretrainedTransformerMismatchedEmbedder,
)

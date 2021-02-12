"""
Custom PyTorch
`Module <https://pytorch.org/docs/master/nn.html#torch.nn.Module>`_ s
that are used as components in AllenNLP `Model` s.
"""

from camphr_allennlp.modules.attention import Attention
from camphr_allennlp.modules.bimpm_matching import BiMpmMatching
from camphr_allennlp.modules.conditional_random_field import ConditionalRandomField
from camphr_allennlp.modules.elmo import Elmo
from camphr_allennlp.modules.feedforward import FeedForward
from camphr_allennlp.modules.gated_sum import GatedSum
from camphr_allennlp.modules.highway import Highway
from camphr_allennlp.modules.input_variational_dropout import InputVariationalDropout
from camphr_allennlp.modules.layer_norm import LayerNorm
from camphr_allennlp.modules.matrix_attention import MatrixAttention
from camphr_allennlp.modules.maxout import Maxout
from camphr_allennlp.modules.residual_with_layer_dropout import ResidualWithLayerDropout
from camphr_allennlp.modules.scalar_mix import ScalarMix
from camphr_allennlp.modules.seq2seq_encoders import Seq2SeqEncoder
from camphr_allennlp.modules.seq2vec_encoders import Seq2VecEncoder
from camphr_allennlp.modules.text_field_embedders import TextFieldEmbedder
from camphr_allennlp.modules.time_distributed import TimeDistributed
from camphr_allennlp.modules.token_embedders import TokenEmbedder, Embedding
from camphr_allennlp.modules.softmax_loss import SoftmaxLoss

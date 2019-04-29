from .scaled_dot_attention import ScaledDotProductAttention
from .seq_self_attention import SeqSelfAttention
from .seq_weighted_attention import SeqWeightedAttention
from .multi_head_attention import MultiHeadAttention

"""
att_layer = MultiHeadAttention(
    head_num=3,
    name='Multi-Head',
)(input_layer)
"""
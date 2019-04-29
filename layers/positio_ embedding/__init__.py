from .pos_embd import PositionEmbedding
from .trig_pos_embd import TrigPosEmbedding

"""
model.add(PositionEmbedding(
    input_shape=(None,),
    input_dim=10,     # The maximum absolute value of positions.
    output_dim=2,     # The dimension of embeddings.
    mask_zero=10000,  # The index that presents padding (because `0` will be used in relative positioning).
    name='Pos-Embd',
))

model.add(TrigPosEmbedding(
    input_shape=(None, 100),
    mode=TrigPosEmbedding.MODE_ADD,  # Use `add` mode (default)
    name='Pos-Embd',
))
"""
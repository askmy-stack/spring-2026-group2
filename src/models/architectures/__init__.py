from .vanilla_lstm import VanillaLSTM
from .bilstm import BiLSTM
from .attention_bilstm import AttentionBiLSTM
from .cnn_lstm import CNNLSTM
from .feature_bilstm import FeatureBiLSTM

MODEL_REGISTRY = {
    "vanilla_lstm": VanillaLSTM,
    "bilstm": BiLSTM,
    "attention_bilstm": AttentionBiLSTM,
    "cnn_lstm": CNNLSTM,
    "feature_bilstm": FeatureBiLSTM,
}

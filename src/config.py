from dataclasses import dataclass

@dataclass
class TrainConfig:
    seed: int = 42
    bert_model_name: str = "bert-base-uncased"
    max_length: int = 256

    batch_size: int = 16
    lr: float = 2e-5
    epochs: int = 2
    weight_decay: float = 0.01

    # CNN + LSTM hyperparams
    cnn_num_filters: int = 128
    cnn_kernel_sizes: tuple = (3, 4, 5)
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1
    dropout: float = 0.2

"""Configuration management for GeoSAE."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import yaml


@dataclass
class ModelConfig:
    """Configuration for the neural network model."""

    # Encoder configuration
    encoder_channels: List[int] = field(default_factory=lambda: [3, 128, 256])
    encoder_fc_dim: int = 256

    # Decoder configuration
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 1])

    # Activation function
    activation: str = "softplus"
    softplus_beta: float = 1.0

    # Number of potential fields (stratigraphic surfaces)
    num_potential_fields: int = 7


@dataclass
class TrainingConfig:
    """Configuration for training."""

    # Training parameters
    batch_size: int = 128
    learning_rate: float = 0.001
    lr_decay_step: int = 500
    lr_decay_factor: float = 0.5
    num_iterations: int = 5000

    # Pre-training parameters
    pretrain_iterations: int = 5000
    pretrain_planes: List[float] = field(default_factory=lambda: [0, 25, 50, 75])

    # Loss weights
    lambda_smoothness: float = 0.1
    lambda_variance: float = 1.0
    lambda_sequence: float = 1.0
    lambda_attitude: float = 1.0

    # Device
    device: str = "cuda"

    # Random seed
    seed: int = 42


@dataclass
class DataConfig:
    """Configuration for data handling."""

    # Coordinate normalization range
    normalize_range: Tuple[float, float] = (-1.0, 1.0)

    # Grid resolution for prediction (x, y, z in meters)
    grid_resolution: Tuple[float, float, float] = (50.0, 50.0, 10.0)

    # Data augmentation
    use_augmentation: bool = False
    noise_std: float = 0.01


@dataclass
class Config:
    """Main configuration class."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def save(self, path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            "model": {
                "encoder_channels": self.model.encoder_channels,
                "encoder_fc_dim": self.model.encoder_fc_dim,
                "decoder_channels": self.model.decoder_channels,
                "activation": self.model.activation,
                "softplus_beta": self.model.softplus_beta,
                "num_potential_fields": self.model.num_potential_fields,
            },
            "training": {
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "lr_decay_step": self.training.lr_decay_step,
                "lr_decay_factor": self.training.lr_decay_factor,
                "num_iterations": self.training.num_iterations,
                "pretrain_iterations": self.training.pretrain_iterations,
                "pretrain_planes": self.training.pretrain_planes,
                "lambda_smoothness": self.training.lambda_smoothness,
                "lambda_variance": self.training.lambda_variance,
                "lambda_sequence": self.training.lambda_sequence,
                "lambda_attitude": self.training.lambda_attitude,
                "device": self.training.device,
                "seed": self.training.seed,
            },
            "data": {
                "normalize_range": list(self.data.normalize_range),
                "grid_resolution": list(self.data.grid_resolution),
                "use_augmentation": self.data.use_augmentation,
                "noise_std": self.data.noise_std,
            },
        }
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)

        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        data_config = DataConfig(
            normalize_range=tuple(config_dict.get("data", {}).get("normalize_range", (-1.0, 1.0))),
            grid_resolution=tuple(config_dict.get("data", {}).get("grid_resolution", (50.0, 50.0, 10.0))),
            use_augmentation=config_dict.get("data", {}).get("use_augmentation", False),
            noise_std=config_dict.get("data", {}).get("noise_std", 0.01),
        )

        return cls(model=model_config, training=training_config, data=data_config)


# Default configuration instance
default_config = Config()

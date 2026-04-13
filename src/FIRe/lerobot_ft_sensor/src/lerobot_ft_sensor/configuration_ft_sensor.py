from dataclasses import dataclass
import os


@dataclass
class FTSensorConfig:

    config_path: str = os.path.join(os.path.dirname(__file__), "config", "bota_binary.json")
    warmup_s: float = 1.0
    timeout_ms: int = 200

    def __post_init__(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"FT Sensor config file not found: {self.config_path}")

    
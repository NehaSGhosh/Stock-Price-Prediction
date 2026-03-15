import logging
import os
from datetime import datetime

from config.configuration import ConfigurationManager
from stock_price_predictor.utils.common import upload_bytes_to_gcs


class GCSMirroringFileHandler(logging.FileHandler):
    """
    File handler that periodically mirrors the local log file to GCS.
    """

    def __init__(
        self,
        filename: str,
        bucket_name: str,
        gcs_blob_name: str,
        sync_every_n_records: int = 20,
        **kwargs,
    ):
        super().__init__(filename, **kwargs)
        self.bucket_name = bucket_name
        self.gcs_blob_name = gcs_blob_name
        self.sync_every_n_records = max(1, int(sync_every_n_records))
        self._records_since_sync = 0

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self._records_since_sync += 1
        if self._records_since_sync >= self.sync_every_n_records:
            self._sync_to_gcs()
            self._records_since_sync = 0

    def _sync_to_gcs(self) -> None:
        try:
            self.flush()
            if self.stream:
                self.stream.flush()
            with open(self.baseFilename, "rb") as log_file:
                payload = log_file.read()
            upload_bytes_to_gcs(
                data=payload,
                bucket_name=self.bucket_name,
                destination_blob=self.gcs_blob_name,
                content_type="text/plain",
            )
        except Exception:
            # Avoid recursive logger failures if GCS is temporarily unavailable.
            pass

    def close(self) -> None:
        self._sync_to_gcs()
        super().close()


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
run_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
LOG_FILE = os.path.join(LOG_DIR, f"{run_stamp}.log")

config_manager = ConfigurationManager()
gcs_cfg = config_manager.get_gcs_config()
logs_prefix = gcs_cfg.get("logs_prefix", "logs").strip("/")
gcs_log_blob = f"{logs_prefix}/{os.path.basename(LOG_FILE)}"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    handlers=[
        GCSMirroringFileHandler(
            filename=LOG_FILE,
            bucket_name=gcs_cfg["bucket_name"],
            gcs_blob_name=gcs_log_blob,
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

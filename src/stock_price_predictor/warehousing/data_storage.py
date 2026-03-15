import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

from config.configuration import ConfigurationManager
from stock_price_predictor.logger import logging
from stock_price_predictor.utils.common import (
    ensure_dir,
    gcs_blob_exists,
    parse_gcs_uri,
    read_from_gcs,
    upload_to_gcs,
)


class GoldWarehouse:
    EASTERN_TZ = ZoneInfo("America/New_York")

    """
    Gold layer builder and cloud warehouse loader (BigQuery).
    """

    def __init__(
        self,
        project_id: str,
        dataset_id: str,
        table_id: str,
    ):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id

    @staticmethod
    def build_gold_dataset(market_df: pd.DataFrame, news_df: pd.DataFrame) -> pd.DataFrame:
        gold_df = market_df.copy()
        headlines_df = news_df.copy()

        gold_df["date"] = pd.to_datetime(gold_df["date"], errors="coerce")
        gold_df = gold_df.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"]).reset_index(drop=True)
        headlines_df["date"] = pd.to_datetime(headlines_df["date"], errors="coerce")
        headlines_df = headlines_df.dropna(subset=["date", "ticker", "headline"])
        headlines_df["headline"] = headlines_df["headline"].astype(str).str.strip()
        headlines_df = headlines_df[headlines_df["headline"] != ""]

        if not headlines_df.empty:
            daily_headlines_df = (
                headlines_df.groupby(["ticker", "date"], as_index=False)["headline"]
                .agg(lambda series: list(dict.fromkeys(series.tolist())))
            )
            daily_headlines_df = daily_headlines_df.rename(columns={"headline": "headlines_list"})
            gold_df = gold_df.merge(daily_headlines_df, on=["ticker", "date"], how="left")
        else:
            gold_df["headlines_list"] = pd.NA

        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            if col in gold_df.columns:
                gold_df[col] = pd.to_numeric(gold_df[col], errors="coerce")
        gold_df["headlines_list"] = gold_df["headlines_list"].apply(
            lambda value: value if isinstance(value, list) else []
        )

        gold_df = gold_df.dropna(subset=["close"])
        grouped_close = gold_df.groupby("ticker")["close"]
        # Compute next-day pct change directly from shifted close; no persisted next_close column needed.
        next_day_pct_change = ((grouped_close.shift(-1) - gold_df["close"]) / gold_df["close"]) * 100.0
        gold_df["next_day_price_change_pct"] = next_day_pct_change
        has_label = next_day_pct_change.notna()
        gold_df["target_up"] = pd.Series(pd.NA, index=gold_df.index, dtype="Int64")
        gold_df.loc[has_label, "target_up"] = (next_day_pct_change[has_label] > 0).astype(int).astype("Int64")
        gold_df = gold_df.reset_index(drop=True)
        gold_df["date"] = gold_df["date"].dt.date
        return gold_df

    @staticmethod
    def save_gold_csv(
        gold_df: pd.DataFrame,
        output_path: str,
    ) -> str:
        if output_path.startswith("gs://"):
            bucket_name, blob_name = parse_gcs_uri(output_path)
            upload_to_gcs(df=gold_df, bucket_name=bucket_name, destination_blob=blob_name)
            logging.info("Gold CSV saved to GCS: %s (rows=%d)", output_path, len(gold_df))
            return output_path
        ensure_dir(os.path.dirname(output_path))
        gold_df.to_csv(output_path, index=False)
        logging.info("Gold CSV saved at: %s (rows=%d)", output_path, len(gold_df))
        return output_path

    @staticmethod
    def _normalize_gold_dataframe(gold_df: pd.DataFrame) -> pd.DataFrame:
        normalized = gold_df.copy()
        if "headline" in normalized.columns:
            normalized = normalized.drop(columns=["headline"])
        if "date" in normalized.columns:
            normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce").dt.date
        normalized = normalized.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])
        normalized = normalized.drop_duplicates(subset=["ticker", "date"], keep="last").reset_index(drop=True)
        return normalized

    @staticmethod
    def _recompute_gold_targets(gold_df: pd.DataFrame) -> pd.DataFrame:
        recomputed = gold_df.copy()
        if "close" not in recomputed.columns:
            raise ValueError("Gold dataframe is missing required 'close' column.")
        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            if col in recomputed.columns:
                recomputed[col] = pd.to_numeric(recomputed[col], errors="coerce")
        recomputed = recomputed.dropna(subset=["close"])
        grouped_close = recomputed.groupby("ticker")["close"]
        next_day_pct_change = ((grouped_close.shift(-1) - recomputed["close"]) / recomputed["close"]) * 100.0
        recomputed["next_day_price_change_pct"] = next_day_pct_change
        has_label = next_day_pct_change.notna()
        recomputed["target_up"] = pd.Series(pd.NA, index=recomputed.index, dtype="Int64")
        recomputed.loc[has_label, "target_up"] = (next_day_pct_change[has_label] > 0).astype(int).astype("Int64")
        return recomputed.reset_index(drop=True)

    @staticmethod
    def _load_required_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
        cfg_manager = ConfigurationManager()
        gcs_cfg = cfg_manager.get_gcs_config()
        raw_blobs = cfg_manager.get_raw_blob_paths()
        bucket_name = gcs_cfg["bucket_name"]

        if not gcs_blob_exists(bucket_name=bucket_name, blob_name=raw_blobs["market"]):
            raise FileNotFoundError(
                f"Required raw market_data file not found in GCS: gs://{bucket_name}/{raw_blobs['market']}"
            )
        if not gcs_blob_exists(bucket_name=bucket_name, blob_name=raw_blobs["news"]):
            raise FileNotFoundError(
                f"Required raw news_data file not found in GCS: gs://{bucket_name}/{raw_blobs['news']}"
            )

        market_df = read_from_gcs(bucket_name=bucket_name, blob_name=raw_blobs["market"])
        news_df = read_from_gcs(bucket_name=bucket_name, blob_name=raw_blobs["news"])
        if market_df.empty:
            raise ValueError(
                f"Required raw market_data file is empty: gs://{bucket_name}/{raw_blobs['market']}"
            )
        if news_df.empty:
            raise ValueError(f"Required raw news_data file is empty: gs://{bucket_name}/{raw_blobs['news']}")
        return market_df, news_df

    @classmethod
    def _build_gold_window_from_raw(cls, start_date, end_date) -> pd.DataFrame:
        market_df, news_df = cls._load_required_raw_data()
        gold_df = cls.build_gold_dataset(market_df, news_df)
        gold_df["date"] = pd.to_datetime(gold_df["date"], errors="coerce").dt.date
        gold_df = gold_df[(gold_df["date"] >= start_date) & (gold_df["date"] <= end_date)].reset_index(drop=True)
        return cls._normalize_gold_dataframe(gold_df)

    @staticmethod
    def _ensure_google_credentials_env() -> None:
        """
        Normalize GOOGLE_APPLICATION_CREDENTIALS from environment/.env by:
        - loading .env
        - stripping wrapping quotes
        - validating file existence when provided
        """
        load_dotenv()
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip().strip('"').strip("'")
        if not credentials_path:
            logging.warning(
                "GOOGLE_APPLICATION_CREDENTIALS not set in environment/.env; "
                "BigQuery will rely on Application Default Credentials."
            )
            return
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        if not os.path.exists(credentials_path):
            raise FileNotFoundError(f"GOOGLE_APPLICATION_CREDENTIALS file not found: {credentials_path}")
        logging.info("Using BigQuery credentials from GOOGLE_APPLICATION_CREDENTIALS.")

    def load_gold_to_bigquery(self, gold_df: pd.DataFrame, write_mode: str = "truncate") -> str:
        try:
            from google.cloud import bigquery
        except Exception as error:
            raise ImportError("google-cloud-bigquery is required for BigQuery load.") from error

        if not self.project_id:
            raise ValueError("BigQuery project_id is required to load Gold table.")

        self._ensure_google_credentials_env()
        client = bigquery.Client(project=self.project_id)
        dataset_ref = bigquery.DatasetReference(self.project_id, self.dataset_id)
        table_ref = dataset_ref.table(self.table_id)
        client.create_dataset(bigquery.Dataset(dataset_ref), exists_ok=True)

        schema = [
            bigquery.SchemaField("date", "DATE"),
            bigquery.SchemaField("ticker", "STRING"),
            bigquery.SchemaField("headlines_list", "STRING"),
            bigquery.SchemaField("open", "FLOAT"),
            bigquery.SchemaField("high", "FLOAT"),
            bigquery.SchemaField("low", "FLOAT"),
            bigquery.SchemaField("close", "FLOAT"),
            bigquery.SchemaField("adj_close", "FLOAT"),
            bigquery.SchemaField("volume", "FLOAT"),
            bigquery.SchemaField("next_day_price_change_pct", "FLOAT"),
            bigquery.SchemaField("target_up", "INTEGER"),
        ]
        load_df = gold_df.copy()
        load_df["date"] = pd.to_datetime(load_df["date"], errors="coerce").dt.date
        if "headlines_list" in load_df.columns:
            load_df["headlines_list"] = load_df["headlines_list"].apply(
                lambda value: json.dumps(value) if isinstance(value, list) else str(value or "[]")
            )

        mode = (write_mode or "truncate").strip().lower()
        if mode not in {"truncate", "append"}:
            raise ValueError("write_mode must be one of: truncate, append")

        write_disposition = (
            bigquery.WriteDisposition.WRITE_APPEND
            if mode == "append"
            else bigquery.WriteDisposition.WRITE_TRUNCATE
        )
        job_config = bigquery.LoadJobConfig(
            schema=schema,
            write_disposition=write_disposition,
        )
        client.load_table_from_dataframe(load_df, table_ref, job_config=job_config).result()
        full_table_name = f"{self.project_id}.{self.dataset_id}.{self.table_id}"
        logging.info(
            "Gold data loaded to BigQuery table: %s (rows=%d, mode=%s)",
            full_table_name,
            len(load_df),
            mode,
        )
        return full_table_name

    def load_dataframe_to_bigquery(
        self,
        df: pd.DataFrame,
        table_id: str,
        write_mode: str = "truncate",
    ) -> str:
        try:
            from google.cloud import bigquery
        except Exception as error:
            raise ImportError("google-cloud-bigquery is required for BigQuery load.") from error

        if not self.project_id:
            raise ValueError("BigQuery project_id is required to load table.")

        self._ensure_google_credentials_env()
        client = bigquery.Client(project=self.project_id)
        dataset_ref = bigquery.DatasetReference(self.project_id, self.dataset_id)
        table_ref = dataset_ref.table(table_id)
        client.create_dataset(bigquery.Dataset(dataset_ref), exists_ok=True)

        mode = (write_mode or "truncate").strip().lower()
        if mode not in {"truncate", "append"}:
            raise ValueError("write_mode must be one of: truncate, append")
        write_disposition = (
            bigquery.WriteDisposition.WRITE_APPEND
            if mode == "append"
            else bigquery.WriteDisposition.WRITE_TRUNCATE
        )

        load_df = df.copy()
        if "date" in load_df.columns:
            load_df["date"] = pd.to_datetime(load_df["date"], errors="coerce").dt.date
        if "headlines_list" in load_df.columns:
            load_df["headlines_list"] = load_df["headlines_list"].apply(
                lambda value: json.dumps(value) if isinstance(value, list) else str(value or "[]")
            )

        job_config = bigquery.LoadJobConfig(write_disposition=write_disposition)
        client.load_table_from_dataframe(load_df, table_ref, job_config=job_config).result()
        full_table_name = f"{self.project_id}.{self.dataset_id}.{table_id}"
        logging.info(
            "Data loaded to BigQuery table: %s (rows=%d, mode=%s)",
            full_table_name,
            len(load_df),
            mode,
        )
        return full_table_name

    @classmethod
    def refresh_gold_from_raw(cls, lookback_days: int) -> tuple[pd.DataFrame, str]:
        if lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")

        end_date = datetime.now(cls.EASTERN_TZ).date()
        start_date = end_date - timedelta(days=lookback_days - 1)
        gold_df = cls._build_gold_window_from_raw(start_date=start_date, end_date=end_date)

        warehouse = cls.from_config()
        table_name = warehouse.load_gold_to_bigquery(gold_df, write_mode="truncate")
        return gold_df, table_name

    @classmethod
    def append_gold_from_raw(cls, append_days: int) -> tuple[pd.DataFrame, str]:
        if append_days < 1:
            raise ValueError("append_days must be >= 1")

        warehouse = cls.from_config()
        table_full_name = f"{warehouse.project_id}.{warehouse.dataset_id}.{warehouse.table_id}"
        existing_gold_df = warehouse.fetch_all_rows_from_bigquery(warehouse.table_id)
        if existing_gold_df.empty:
            raise FileNotFoundError(f"Append mode requires existing Gold data in BigQuery table: {table_full_name}")

        end_date = datetime.now(cls.EASTERN_TZ).date()
        start_date = end_date - timedelta(days=append_days - 1)
        append_df = cls._build_gold_window_from_raw(start_date=start_date, end_date=end_date)
        append_df = cls._normalize_gold_dataframe(append_df)

        merged_gold_df = pd.concat([existing_gold_df, append_df], ignore_index=True)
        merged_gold_df = cls._normalize_gold_dataframe(merged_gold_df)
        # Recompute forward-looking labels so the day before newly appended data is backfilled.
        merged_gold_df = cls._recompute_gold_targets(merged_gold_df)

        table_name = warehouse.load_gold_to_bigquery(merged_gold_df, write_mode="truncate")
        return merged_gold_df, table_name

    def fetch_gold_from_bigquery(self, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            from google.cloud import bigquery
        except Exception as error:
            raise ImportError("google-cloud-bigquery is required for BigQuery read.") from error

        if not self.project_id:
            raise ValueError("BigQuery project_id is required to read Gold table.")

        self._ensure_google_credentials_env()
        client = bigquery.Client(project=self.project_id)
        table_name = f"`{self.project_id}.{self.dataset_id}.{self.table_id}`"
        query = f"""
        SELECT *
        FROM {table_name}
        WHERE date BETWEEN DATE('{start_date}') AND DATE('{end_date}')
        ORDER BY ticker, date
        """
        result_df = client.query(query).to_dataframe()
        if result_df.empty:
            logging.info("No Gold rows found in BigQuery for window %s to %s.", start_date, end_date)
            return result_df
        result_df["date"] = pd.to_datetime(result_df["date"], errors="coerce").dt.date
        logging.info(
            "Fetched Gold data from BigQuery table %s rows=%d",
            f"{self.project_id}.{self.dataset_id}.{self.table_id}",
            len(result_df),
        )
        return result_df

    def fetch_table_from_bigquery(self, table_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            from google.cloud import bigquery
        except Exception as error:
            raise ImportError("google-cloud-bigquery is required for BigQuery read.") from error

        if not self.project_id:
            raise ValueError("BigQuery project_id is required to read table.")

        self._ensure_google_credentials_env()
        client = bigquery.Client(project=self.project_id)
        table_name = f"`{self.project_id}.{self.dataset_id}.{table_id}`"
        query = f"""
        SELECT *
        FROM {table_name}
        WHERE date BETWEEN DATE('{start_date}') AND DATE('{end_date}')
        ORDER BY ticker, date
        """
        result_df = client.query(query).to_dataframe()
        if "date" in result_df.columns:
            result_df["date"] = pd.to_datetime(result_df["date"], errors="coerce").dt.date
        logging.info("Fetched rows=%d from BigQuery table %s", len(result_df), f"{self.project_id}.{self.dataset_id}.{table_id}")
        return result_df

    def fetch_all_rows_from_bigquery(self, table_id: str) -> pd.DataFrame:
        try:
            from google.cloud import bigquery
        except Exception as error:
            raise ImportError("google-cloud-bigquery is required for BigQuery read.") from error

        if not self.project_id:
            raise ValueError("BigQuery project_id is required to read table.")

        self._ensure_google_credentials_env()
        client = bigquery.Client(project=self.project_id)
        table_name = f"`{self.project_id}.{self.dataset_id}.{table_id}`"
        query = f"""
        SELECT *
        FROM {table_name}
        ORDER BY ticker, date
        """
        result_df = client.query(query).to_dataframe()
        if "date" in result_df.columns:
            result_df["date"] = pd.to_datetime(result_df["date"], errors="coerce").dt.date
        logging.info(
            "Fetched rows=%d from BigQuery table %s",
            len(result_df),
            f"{self.project_id}.{self.dataset_id}.{table_id}",
        )
        return result_df

    @classmethod
    def resolve_gold_for_training(cls) -> tuple[pd.DataFrame, str]:
        cfg_manager = ConfigurationManager()
        ingestion_cfg = cfg_manager.get_data_ingestion_config()
        end_date = datetime.now(cls.EASTERN_TZ).date()
        start_date = end_date - timedelta(days=ingestion_cfg.lookback_days - 1)
        warehouse = cls.from_config()
        logging.info("Fetching Gold data from BigQuery for training.")
        bq_df = warehouse.fetch_gold_from_bigquery(start_date.isoformat(), end_date.isoformat())
        if bq_df.empty:
            raise FileNotFoundError(
                "No matching Gold rows found in BigQuery "
                f"for window {start_date} to {end_date}."
            )

        table_name = f"{warehouse.project_id}.{warehouse.dataset_id}.{warehouse.table_id}"
        return bq_df, table_name

    @classmethod
    def from_env(cls, default_project_id: str) -> "GoldWarehouse":
        _ = default_project_id  # Backward-compatible signature; config.yaml is source of truth.
        return cls.from_config()

    @classmethod
    def from_config(cls) -> "GoldWarehouse":
        cfg = ConfigurationManager().get_gold_warehouse_config()
        return cls(
            project_id=cfg["project_id"].strip(),
            dataset_id=cfg["dataset_id"].strip(),
            table_id=cfg["table_id"].strip(),
        )

import pandas as pd

class DataProcessing:
    @staticmethod
    def merge_news_with_stock(
        news_df: pd.DataFrame,
        stock_df: pd.DataFrame,
    ) -> pd.DataFrame:
        news = news_df.copy()
        stock = stock_df.copy()
        news["date"] = pd.to_datetime(news["date"], errors="coerce").dt.date
        stock["date"] = pd.to_datetime(stock["date"], errors="coerce").dt.date

        required_news_cols = {"ticker", "date"}
        required_stock_cols = {"ticker", "date"}
        missing_news = required_news_cols.difference(news.columns)
        missing_stock = required_stock_cols.difference(stock.columns)
        if missing_news:
            raise ValueError(f"Missing required news columns: {sorted(missing_news)}")
        if missing_stock:
            raise ValueError(f"Missing required stock columns: {sorted(missing_stock)}")

        sentiment_value_cols = [col for col in news.columns if col not in {"ticker", "date"}]
        if not sentiment_value_cols:
            raise ValueError("News dataframe must contain at least one sentiment value column besides ticker/date.")

        merged = stock.merge(
            news[["ticker", "date", *sentiment_value_cols]],
            on=["ticker", "date"],
            how="left",
        )
        for col in sentiment_value_cols:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
        merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
        return merged

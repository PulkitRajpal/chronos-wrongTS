from pathlib import Path
import time

import pandas as pd
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "ENPH",
    "SMCI",
    "CVNA",
    "PLUG",
    "RKLB",
    "IONQ",
]

OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Download the maximum available daily history.
START_DATE = "2010-01-01"

# Important:
# auto_adjust=True makes OHLC prices split/dividend adjusted,
# which is generally better for a long-horizon price forecasting
# experiment because stock splits don't create artificial jumps.
AUTO_ADJUST = True


# ============================================================
# DOWNLOAD ONE TICKER
# ============================================================

def download_ticker(ticker: str) -> None:

    print(f"\nDownloading {ticker}...")

    try:
        df = yf.download(
            ticker,
            start=START_DATE,
            interval="1d",
            auto_adjust=AUTO_ADJUST,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            print(f"  FAILED: no data returned")
            return

        # yfinance can return MultiIndex columns even for one ticker.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.reset_index()

        # Normalize names.
        df.columns = [
            str(c).strip().lower()
            for c in df.columns
        ]

        required = {
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }

        missing = required - set(df.columns)

        if missing:
            print(
                f"  FAILED: missing columns "
                f"{sorted(missing)}"
            )
            print(
                f"  Got columns: {list(df.columns)}"
            )
            return

        # Keep only what the Chronos POC needs.
        df = df[
            [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ].copy()

        # Convert to clean types.
        df["date"] = pd.to_datetime(
            df["date"],
            errors="coerce",
        )

        for col in [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

        # Remove bad rows.
        df = (
            df
            .dropna(
                subset=[
                    "date",
                    "open",
                    "high",
                    "low",
                    "close",
                ]
            )
            .sort_values("date")
            .drop_duplicates("date")
            .reset_index(drop=True)
        )

        output_file = OUT_DIR / f"{ticker}.csv"

        df.to_csv(
            output_file,
            index=False,
        )

        print(
            f"  OK: {len(df):,} rows"
        )

        print(
            f"  Range: "
            f"{df['date'].min().date()} -> "
            f"{df['date'].max().date()}"
        )

        print(
            f"  Saved: {output_file}"
        )

    except Exception as exc:
        print(
            f"  FAILED: "
            f"{type(exc).__name__}: {exc}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DOWNLOADING STOCK DATA")
    print("=" * 70)

    for ticker in TICKERS:

        download_ticker(ticker)

        # Small delay to avoid hammering the endpoint.
        time.sleep(1)

    print("\n")
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
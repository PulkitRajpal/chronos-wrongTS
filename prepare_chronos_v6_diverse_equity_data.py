from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = Path(
    "data/chronos_finance_v6"
)

START_DATE = "2010-01-01"
END_DATE = "2026-01-01"

CONTEXT_LENGTH = 512
PREDICTION_LENGTH = 20
VALIDATION_DAYS = 252

MIN_REQUIRED_ROWS = (
    CONTEXT_LENGTH
    +
    PREDICTION_LENGTH
    +
    VALIDATION_DAYS
)

EVAL_TICKERS = {
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
}

REFERENCE_TICKERS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "VIX": "^VIX",
    "US5Y": "^FVX",
    "US10Y": "^TNX",
}


# ============================================================
# DIVERSE, CURRENT EQUITY UNIVERSE
# ============================================================

SECTOR_TICKERS = {

    "technology": [
        "GOOG", "META", "ORCL", "ADBE", "CRM",
        "INTC", "AMD", "AVGO", "QCOM", "TXN",
        "CSCO", "IBM", "AMAT", "MU", "LRCX",
        "KLAC", "ADI", "MCHP", "MRVL", "NXPI",
        "ON", "ARM", "TER", "APH", "CDNS",
        "SNPS", "FTNT", "PANW", "CRWD", "NET",
        "DDOG", "TEAM", "NOW", "ADSK", "INTU",
        "WDAY", "ZS", "OKTA", "PLTR", "SNOW",
        "SHOP", "SQ", "PYPL", "UBER", "ABNB",
        "DASH", "SPOT", "ROKU", "PINS", "TTD",
        "EA", "TTWO", "APP", "HUBS", "DELL",
        "HPQ", "HPE", "NTAP", "STX", "WDC",
        "AKAM", "GEN", "CHKP",
    ],

    "financials": [
        "JPM", "BAC", "GS", "MS", "C", "WFC",
        "USB", "PNC", "TFC", "STT", "SCHW", "BLK",
        "ICE", "CME", "SPGI", "MCO", "AON", "MMC",
        "AJG", "CB", "PGR", "TRV", "ALL", "MET",
        "PRU", "AIG", "COF", "AXP", "MA", "V",
        "FITB", "KEY", "RF", "CFG", "HBAN", "MTB",
        "ZION", "NTRS", "AMP", "TROW",
    ],

    "healthcare": [
        "JNJ", "PFE", "MRK", "ABBV", "LLY", "UNH",
        "TMO", "DHR", "ABT", "BMY", "AMGN", "GILD",
        "BIIB", "REGN", "VRTX", "ISRG", "MDT",
        "SYK", "BSX", "EW", "ZBH", "ELV", "CI",
        "CVS", "HUM", "MCK", "CAH", "COR", "IQV",
        "HCA", "ZTS", "DXCM", "RMD", "ALGN", "IDXX",
    ],

    "consumer_discretionary": [
        "HD", "LOW", "NKE", "MCD", "SBUX", "TGT",
        "TJX", "BKNG", "MAR", "HLT", "CMG", "YUM",
        "GM", "F", "ORLY", "AZO", "ROST", "DHI",
        "LEN", "PHM", "DRI", "LVS", "WYNN", "MGM",
        "CCL", "RCL", "NCLH", "ETSY", "ULTA", "WSM",
        "BBY", "DECK", "LULU", "RL", "TPR", "EBAY",
        "EXPE", "CHWY",
    ],

    "consumer_staples": [
        "WMT", "COST", "KO", "PEP", "PG", "CL",
        "KMB", "GIS", "MDLZ", "MO", "PM", "KR",
        "KHC", "SYY", "HSY", "K", "EL", "MNST",
        "CHD", "CLX", "TSN", "CAG",
    ],

    "industrials": [
        "BA", "CAT", "GE", "HON", "RTX", "LMT",
        "NOC", "GD", "DE", "ETN", "EMR", "MMM",
        "UPS", "FDX", "CSX", "NSC", "UNP", "WM",
        "RSG", "PCAR", "CMI", "CARR", "OTIS", "TT",
        "PH", "ROK", "FAST", "URI", "DAL", "UAL",
        "LUV", "JCI", "IR", "SWK", "GWW", "ITW",
        "TXT", "AME", "XYL", "PWR",
    ],

    "energy": [
        "XOM", "CVX", "COP", "SLB", "EOG", "OXY",
        "MPC", "PSX", "VLO", "WMB", "KMI", "OKE",
        "HAL", "DVN", "FANG", "BKR", "APA", "CTRA",
        "EQT", "TRGP", "MRO", "CHRD",
    ],

    "materials": [
        "LIN", "APD", "SHW", "ECL", "NEM", "FCX",
        "NUE", "STLD", "DOW", "DD", "VMC", "MLM",
        "ALB", "EMN", "IFF", "CE", "CF", "MOS",
        "PPG", "PKG", "IP", "AVY", "BALL", "AMCR",
        "FMC", "CLF",
    ],

    "utilities": [
        "NEE", "DUK", "SO", "D", "AEP", "EXC",
        "SRE", "XEL", "ED", "PEG", "WEC", "ES",
        "ETR", "FE", "AWK", "DTE", "CMS", "PPL",
        "CNP", "AES", "NI", "EVRG", "LNT",
    ],

    "real_estate": [
        "PLD", "AMT", "EQIX", "CCI", "SPG", "O",
        "PSA", "WELL", "DLR", "VICI", "AVB", "EQR",
        "ESS", "INVH", "EXR", "IRM", "WY", "ARE",
        "MAA", "UDR", "VTR", "SBAC", "HST",
    ],

    "communication": [
        "T", "VZ", "TMUS", "CMCSA", "CHTR", "DIS",
        "WBD", "FOX", "FOXA", "PARA", "LYV", "OMC",
        "NWSA", "NWS",
    ],

    "international_adr": [
        "BABA", "JD", "PDD", "BIDU", "NTES",
        "NIO", "LI", "XPEV", "SONY", "TM", "HMC",
        "SAP", "ASML", "NVO", "NVS", "AZN", "SHEL",
        "BP", "RIO", "BHP", "VALE", "UL", "DEO",
        "RELX", "GSK",
    ],

    "higher_beta_growth": [
        "COIN", "MSTR", "HOOD", "AFRM", "UPST",
        "SOFI", "DKNG", "RIVN", "LCID", "MARA",
        "RIOT", "GME", "AMC", "RBLX", "U", "PATH",
        "CELH", "APP", "HIMS", "TEM", "SOUN", "ACHR",
        "JOBY", "ASTS", "OKLO",
    ],
}


# ============================================================
# YFINANCE HELPERS
# ============================================================

def flatten_yf_columns(
    df: pd.DataFrame,
) -> pd.DataFrame:

    out = df.copy()

    if isinstance(
        out.columns,
        pd.MultiIndex,
    ):

        lvl0 = [
            str(x)
            for x in
            out.columns.get_level_values(0)
        ]

        expected = {
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
        }

        if expected.intersection(
            set(lvl0)
        ):

            out.columns = (
                out.columns
                .get_level_values(0)
            )

        else:

            out.columns = (
                out.columns
                .get_level_values(-1)
            )

    out.columns = [
        str(c)
        for c in out.columns
    ]

    return out


def normalize_index(
    index,
) -> pd.DatetimeIndex:

    idx = pd.to_datetime(
        index,
        errors="coerce",
    )

    try:
        if getattr(
            idx,
            "tz",
            None,
        ) is not None:
            idx = idx.tz_localize(None)
    except Exception:
        pass

    return idx


def clean_market_frame(
    raw: pd.DataFrame,
) -> pd.DataFrame:

    df = flatten_yf_columns(
        raw
    ).copy()

    # THIS IS THE IMPORTANT FIX.
    # yfinance often gives the index the name "Date".
    # If we then create a "Date" column while retaining the index name,
    # pandas may see Date at both levels and `sort_values("Date")` fails.
    df.index.name = None

    if "Close" not in df.columns:
        raise ValueError(
            f"Close column missing; "
            f"columns={list(df.columns)}"
        )

    dates = normalize_index(
        df.index
    )

    out = pd.DataFrame({
        "Date":
            dates,

        "Close":
            pd.to_numeric(
                df["Close"],
                errors="coerce",
            ).to_numpy(),
    })

    if "Volume" in df.columns:
        out["Volume"] = pd.to_numeric(
            df["Volume"],
            errors="coerce",
        ).to_numpy()
    else:
        out["Volume"] = np.nan

    if "Open" in df.columns:
        out["Open"] = pd.to_numeric(
            df["Open"],
            errors="coerce",
        ).to_numpy()
    else:
        out["Open"] = np.nan

    if "High" in df.columns:
        out["High"] = pd.to_numeric(
            df["High"],
            errors="coerce",
        ).to_numpy()
    else:
        out["High"] = np.nan

    if "Low" in df.columns:
        out["Low"] = pd.to_numeric(
            df["Low"],
            errors="coerce",
        ).to_numpy()
    else:
        out["Low"] = np.nan

    # The output starts with a completely clean RangeIndex.
    out = out.reset_index(
        drop=True
    )

    return (
        out
        .dropna(
            subset=[
                "Date",
                "Close",
            ]
        )
        .sort_values(
            by="Date"
        )
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def download_single(
    ticker: str,
) -> pd.DataFrame:

    raw = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if raw is None or raw.empty:
        raise RuntimeError(
            "empty Yahoo Finance response"
        )

    return clean_market_frame(
        raw
    )


def download_batch(
    tickers: list[str],
) -> dict[str, pd.DataFrame]:

    if not tickers:
        return {}

    raw = yf.download(
        tickers=tickers,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    if raw is None or raw.empty:
        return {}

    result = {}

    if isinstance(
        raw.columns,
        pd.MultiIndex,
    ):

        first_level = set(
            str(x)
            for x in
            raw.columns.get_level_values(0)
        )

        # Expected layout: ticker -> field.
        for ticker in tickers:

            if ticker not in first_level:
                continue

            try:

                sub = raw[
                    ticker
                ].copy()

                if sub.empty:
                    continue

                result[
                    ticker
                ] = clean_market_frame(
                    sub
                )

            except Exception:
                continue

    elif len(tickers) == 1:

        try:

            result[
                tickers[0]
            ] = clean_market_frame(
                raw
            )

        except Exception:
            pass

    return result


# ============================================================
# REFERENCE DATA
# ============================================================

def download_reference(
    name: str,
    ticker: str,
) -> pd.DataFrame:

    raw = yf.download(
        ticker,
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if raw is None or raw.empty:
        raise RuntimeError(
            f"empty response for {ticker}"
        )

    df = flatten_yf_columns(
        raw
    ).copy()

    # CRITICAL FIX:
    # Clear the index name before constructing a Date column.
    # Otherwise Date can exist simultaneously as index-level name and column.
    df.index.name = None

    if "Close" not in df.columns:
        raise RuntimeError(
            f"Close missing for {name}/{ticker}; "
            f"columns={list(df.columns)}"
        )

    dates = normalize_index(
        df.index
    )

    out = pd.DataFrame({
        "Date":
            dates,

        "Value":
            pd.to_numeric(
                df["Close"],
                errors="coerce",
            ).to_numpy(),
    })

    # CRITICAL FIX:
    out = out.reset_index(
        drop=True
    )

    out = (
        out
        .dropna(
            subset=[
                "Date",
                "Value",
            ]
        )
        .sort_values(
            by="Date"
        )
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return out


def build_reference_frame() -> pd.DataFrame:

    print()
    print("=" * 80)
    print(
        "DOWNLOADING MARKET / RATE REFERENCES"
    )
    print("=" * 80)

    parts = []

    for name, ticker in (
        REFERENCE_TICKERS.items()
    ):

        print(
            f"Downloading reference "
            f"{name} ({ticker})"
        )

        try:

            ref = download_reference(
                name,
                ticker,
            )

        except Exception as exc:

            raise RuntimeError(
                f"Reference "
                f"{name}/{ticker} failed: "
                f"{exc}"
            ) from exc

        ref = ref.rename(
            columns={
                "Value":
                    name
            }
        )

        parts.append(
            ref
        )

        print(
            f"  {name}: "
            f"{len(ref):,} rows"
        )

    # Start from SPY and outer-merge all other series.
    # This prevents one series' missing market holidays from destroying
    # otherwise valid observations for all equities.
    reference = parts[0].copy()

    for p in parts[1:]:

        reference = reference.merge(
            p,
            on="Date",
            how="outer",
        )

    reference = (
        reference
        .sort_values(
            by="Date"
        )
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    raw_reference_cols = [
        "SPY",
        "QQQ",
        "VIX",
        "US5Y",
        "US10Y",
    ]

    # These are contextual series, not the stock target.
    # Small calendar gaps are filled only for reference data.
    reference[
        raw_reference_cols
    ] = (
        reference[
            raw_reference_cols
        ]
        .ffill(limit=3)
    )

    # Build changes/returns after aligning the levels.
    reference["SPY_RET"] = (
        np.log(
            reference["SPY"]
        ).diff()
    )

    reference["QQQ_RET"] = (
        np.log(
            reference["QQQ"]
        ).diff()
    )

    reference["VIX_RET"] = (
        np.log(
            reference["VIX"]
        ).diff()
    )

    reference["US5Y_CHG"] = (
        reference["US5Y"].diff()
    )

    reference["US10Y_CHG"] = (
        reference["US10Y"].diff()
    )

    result = (
        reference[
            [
                "Date",
                "SPY_RET",
                "QQQ_RET",
                "VIX_RET",
                "US5Y_CHG",
                "US10Y_CHG",
            ]
        ]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=[
                "SPY_RET",
                "QQQ_RET",
                "VIX_RET",
                "US5Y_CHG",
                "US10Y_CHG",
            ]
        )
        .sort_values(
            by="Date"
        )
        .reset_index(drop=True)
    )

    return result


# ============================================================
# FEATURE CONSTRUCTION
# ============================================================

def build_features(
    stock: pd.DataFrame,
    reference: pd.DataFrame,
) -> pd.DataFrame:

    d = stock.copy()

    d["target"] = (
        np.log(
            d["Close"]
        ).diff()
    )

    d["ABS_RETURN"] = (
        d["target"].abs()
    )

    d["REALIZED_VOL_20"] = (
        d["target"]
        .rolling(20)
        .std()
    )

    volume = d["Volume"].clip(
        lower=0
    )

    log_volume = np.log1p(
        volume
    )

    rolling_mean = (
        log_volume
        .rolling(20)
        .mean()
    )

    rolling_std = (
        log_volume
        .rolling(20)
        .std()
    )

    d["VOLUME_Z"] = (
        (
            log_volume
            -
            rolling_mean
        )
        /
        rolling_std.replace(
            0,
            np.nan,
        )
    )

    # Stock trading calendar is the master calendar.
    d = d.merge(
        reference,
        on="Date",
        how="left",
    )

    # Only reference covariates can be forward-filled.
    reference_feature_cols = [
        "SPY_RET",
        "QQQ_RET",
        "VIX_RET",
        "US5Y_CHG",
        "US10Y_CHG",
    ]

    d[
        reference_feature_cols
    ] = (
        d[
            reference_feature_cols
        ]
        .ffill(limit=3)
    )

    required = [
        "target",
        *reference_feature_cols,
        "ABS_RETURN",
        "REALIZED_VOL_20",
        "VOLUME_Z",
    ]

    d = (
        d
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna(
            subset=required
        )
        .sort_values(
            by="Date"
        )
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    values = d[
        required
    ].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(
        values
    ).all():

        raise ValueError(
            "non-finite feature values remain"
        )

    return d[
        [
            "Date",
            "target",
            "SPY_RET",
            "QQQ_RET",
            "VIX_RET",
            "US5Y_CHG",
            "US10Y_CHG",
            "ABS_RETURN",
            "REALIZED_VOL_20",
            "VOLUME_Z",
        ]
    ].copy()


# ============================================================
# SAVE TRAIN / VALIDATION
# ============================================================

def save_series(
    ticker: str,
    features: pd.DataFrame,
):

    if len(features) < (
        MIN_REQUIRED_ROWS
    ):

        return None

    split_idx = (
        len(features)
        -
        VALIDATION_DAYS
    )

    train = features.iloc[
        :split_idx
    ].copy()

    validation_start = max(
        0,
        split_idx
        -
        CONTEXT_LENGTH,
    )

    validation = features.iloc[
        validation_start:
    ].copy()

    if len(train) < (
        CONTEXT_LENGTH
        +
        PREDICTION_LENGTH
    ):

        return None

    if len(validation) < (
        CONTEXT_LENGTH
        +
        PREDICTION_LENGTH
    ):

        return None

    ticker_dir = (
        OUTPUT_DIR
        /
        ticker
    )

    ticker_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train.to_parquet(
        ticker_dir
        /
        "train.parquet",
        index=False,
    )

    validation.to_parquet(
        ticker_dir
        /
        "validation.parquet",
        index=False,
    )

    return {
        "ticker":
            ticker,
        "total_rows":
            len(features),
        "train_rows":
            len(train),
        "validation_rows":
            len(validation),
        "first_date":
            str(
                features[
                    "Date"
                ].iloc[0].date()
            ),
        "last_date":
            str(
                features[
                    "Date"
                ].iloc[-1].date()
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    universe = sorted(
        (
            set(
                ticker
                for sector
                in SECTOR_TICKERS.values()
                for ticker in sector
            )
            -
            EVAL_TICKERS
        )
    )

    print("=" * 80)
    print(
        "CHRONOS-2 V6 ROBUST DIVERSE EQUITY DATA"
    )
    print("=" * 80)

    print(
        f"Candidate equities: "
        f"{len(universe)}"
    )

    print(
        f"Minimum required rows: "
        f"{MIN_REQUIRED_ROWS}"
    )

    print(
        f"Context length: "
        f"{CONTEXT_LENGTH}"
    )

    print(
        f"Prediction length: "
        f"{PREDICTION_LENGTH}"
    )

    print(
        f"Validation days: "
        f"{VALIDATION_DAYS}"
    )

    print()
    print(
        "Held-out stocks excluded:"
    )

    print(
        "  "
        +
        ", ".join(
            sorted(
                EVAL_TICKERS
            )
        )
    )

    # --------------------------------------------------------
    # References
    # --------------------------------------------------------

    reference = build_reference_frame()

    print()
    print(
        f"Reference feature rows: "
        f"{len(reference):,}"
    )

    # --------------------------------------------------------
    # Batch equity download
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(
        "BATCH DOWNLOADING EQUITIES"
    )
    print("=" * 80)

    downloaded = download_batch(
        universe
    )

    print(
        f"Successful batch downloads: "
        f"{len(downloaded)}"
    )

    # --------------------------------------------------------
    # Individual fallback.
    # --------------------------------------------------------

    missing = [
        ticker
        for ticker in universe
        if ticker not in downloaded
    ]

    failures = []

    if missing:

        print()
        print(
            f"Retrying {len(missing)} "
            "tickers individually..."
        )

    for ticker in tqdm(
        missing,
        desc="Individual retries",
    ):

        try:

            downloaded[
                ticker
            ] = download_single(
                ticker
            )

        except Exception as exc:

            failures.append({
                "ticker":
                    ticker,
                "stage":
                    "download",
                "error":
                    str(exc),
            })

        time.sleep(
            0.05
        )

    # --------------------------------------------------------
    # Feature build
    # --------------------------------------------------------

    successful = []
    insufficient = []

    print()
    print("=" * 80)
    print(
        "BUILDING EQUITY FEATURES"
    )
    print("=" * 80)

    for ticker in tqdm(
        universe,
        desc="Building V6 series",
    ):

        if ticker not in downloaded:
            continue

        try:

            features = build_features(
                downloaded[
                    ticker
                ],
                reference,
            )

            result = save_series(
                ticker,
                features,
            )

            if result is None:

                insufficient.append({
                    "ticker":
                        ticker,
                    "usable_rows":
                        len(features),
                    "reason":
                        (
                            "insufficient history "
                            f"(need {MIN_REQUIRED_ROWS})"
                        ),
                })

                continue

            successful.append(
                result
            )

        except Exception as exc:

            failures.append({
                "ticker":
                    ticker,
                "stage":
                    "feature_build",
                "error":
                    str(exc),
            })

    # --------------------------------------------------------
    # Leakage check
    # --------------------------------------------------------

    produced = {
        item[
            "ticker"
        ]
        for item in successful
    }

    leakage = (
        produced
        &
        EVAL_TICKERS
    )

    if leakage:

        raise RuntimeError(
            "HELD-OUT LEAKAGE DETECTED: "
            f"{sorted(leakage)}"
        )

    # --------------------------------------------------------
    # Sector coverage
    # --------------------------------------------------------

    sector_coverage = {}

    for sector, tickers in (
        SECTOR_TICKERS.items()
    ):

        clean_sector = (
            set(tickers)
            -
            EVAL_TICKERS
        )

        sector_coverage[
            sector
        ] = len(
            clean_sector
            &
            produced
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "version":
            "V6",

        "start_date":
            START_DATE,

        "end_date":
            END_DATE,

        "context_length":
            CONTEXT_LENGTH,

        "prediction_length":
            PREDICTION_LENGTH,

        "validation_days":
            VALIDATION_DAYS,

        "minimum_required_rows":
            MIN_REQUIRED_ROWS,

        "target":
            "daily_log_return",

        "past_only_covariates": [
            "SPY_RET",
            "QQQ_RET",
            "VIX_RET",
            "US5Y_CHG",
            "US10Y_CHG",
            "ABS_RETURN",
            "REALIZED_VOL_20",
            "VOLUME_Z",
        ],

        "held_out_stocks":
            sorted(
                EVAL_TICKERS
            ),

        "candidate_equities":
            len(universe),

        "usable_equities":
            len(successful),

        "sector_coverage":
            sector_coverage,

        "series":
            successful,

        "insufficient_history":
            insufficient,

        "failures":
            failures,
    }

    with open(
        OUTPUT_DIR
        /
        "metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_train = sum(
        x["train_rows"]
        for x in successful
    )

    total_validation = sum(
        x["validation_rows"]
        for x in successful
    )

    print()
    print("=" * 80)
    print(
        "V6 DATA PREPARATION COMPLETE"
    )
    print("=" * 80)

    print(
        f"Candidate equities: "
        f"{len(universe)}"
    )

    print(
        f"Usable equities: "
        f"{len(successful)}"
    )

    print(
        f"Insufficient history: "
        f"{len(insufficient)}"
    )

    print(
        f"Download/build failures: "
        f"{len(failures)}"
    )

    print(
        f"Training rows: "
        f"{total_train:,}"
    )

    print(
        f"Validation rows: "
        f"{total_validation:,}"
    )

    print()
    print(
        "Sector coverage:"
    )

    for sector, count in sorted(
        sector_coverage.items()
    ):

        print(
            f"  {sector:<24} "
            f"{count:>3}"
        )

    print()
    print(
        "Output:"
    )

    print(
        OUTPUT_DIR.resolve()
    )

    if len(successful) < 150:

        print()
        print(
            "WARNING: fewer than 150 "
            "usable equities survived."
        )

        print(
            "Review metadata.json before training."
        )


if __name__ == "__main__":
    main()

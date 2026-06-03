# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip
# --- Do not remove these libs ---
from functools import reduce
from typing import Any, Callable, Dict, List

import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
from freqtrade.strategy import (IStrategy, Trade, CategoricalParameter,
                                DecimalParameter, IntParameter, BooleanParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class AIAgentStrategy(IStrategy):
    """
    AI Agent Strategy v1.0 — Multi-indicator strategy with adaptive risk.

    Автор: Hermes AI Agent
    Назначение: Готовая стратегия для Freqtrade с бэктестами.
    Рынок: Spot, USDT пары
    Таймфрейм: 5m

    Сигналы на покупку:
      1. EMA50 выше EMA200 (восходящий тренд)
      2. RSI(14) между 30 и 50 (не перекуплен)
      3. MACD line выше Signal line
      4. Объём выше среднего за 20 свечей

    Сигналы на продажу:
      1. RSI > 70 (перекуплен)
      2. MACD line ниже Signal line
      3. Take-profit по фиксированному %
      4. Stop-loss
    """

    # Strategy interface version
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI — агрессивный выход
    minimal_roi = {
        "0": 0.08,    # 8% — сразу
        "15": 0.04,   # 4% через 15 мин
        "60": 0.02,   # 2% через час
        "240": 0.01,  # 1% через 4 часа
        "1440": 0     # 0% через сутки
    }

    # Stop-loss
    stoploss = -0.05  # 5%

    # Trailing stop
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    # Optional order type mapping
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }

    # Timeframe
    timeframe = "5m"

    # --- Hyperoptable parameters ---

    # Buy params
    buy_rsi_lower = IntParameter(25, 45, default=32, space="buy")
    buy_ema_short = IntParameter(30, 70, default=50, space="buy")
    buy_ema_long = IntParameter(150, 250, default=200, space="buy")
    buy_volume_factor = DecimalParameter(1.0, 3.0, default=1.5, decimals=1, space="buy")

    # Sell params
    sell_rsi_upper = IntParameter(65, 85, default=72, space="sell")
    sell_roi_factor = DecimalParameter(0.02, 0.10, default=0.06, decimals=2, space="sell")

    # --- Indicator Definitions ---

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """

        # EMA
        dataframe["ema_short"] = ta.EMA(dataframe, timeperiod=self.buy_ema_short.value)
        dataframe["ema_long"] = ta.EMA(dataframe, timeperiod=self.buy_ema_long.value)

        # RSI
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)

        # MACD
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["macdhist"] = macd["macdhist"]

        # Volume SMA
        dataframe["volume_sma"] = ta.SMA(dataframe["volume"], timeperiod=20)
        dataframe["volume_mean"] = dataframe["volume"].rolling(window=20).mean()

        # ATR for volatility
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)

        # Bollinger Bands
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lowerband"] = bollinger["lower"]
        dataframe["bb_middleband"] = bollinger["mid"]
        dataframe["bb_upperband"] = bollinger["upper"]

        # Additional
        dataframe["close"] = dataframe["close"]
        dataframe["volume"] = dataframe["volume"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal
        """
        conditions = []

        # GUARD: EMA50 > EMA200 (восходящий тренд)
        conditions.append(dataframe["ema_short"] > dataframe["ema_long"])

        # GUARD: RSI между нижней границей и 50
        conditions.append(dataframe["rsi"] > self.buy_rsi_lower.value)
        conditions.append(dataframe["rsi"] < 50)

        # GUARD: MACD line > Signal line
        conditions.append(dataframe["macd"] > dataframe["macdsignal"])

        # GUARD: Объём выше среднего
        conditions.append(dataframe["volume"] > dataframe["volume_mean"] * self.buy_volume_factor.value)

        # GUARD: Цена выше нижней полосы Боллинджера
        conditions.append(dataframe["close"] > dataframe["bb_lowerband"])

        # Combine all conditions
        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                "enter_long"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal
        """
        conditions = []

        # EXIT: RSI > верхней границы (перекуплен)
        conditions.append(dataframe["rsi"] > self.sell_rsi_upper.value)

        # EXIT: MACD line < Signal line
        conditions.append(dataframe["macd"] < dataframe["macdsignal"])

        # EXIT: Цена выше верхней полосы Боллинджера
        conditions.append(dataframe["close"] > dataframe["bb_upperband"])

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                "exit_long"] = 1

        return dataframe

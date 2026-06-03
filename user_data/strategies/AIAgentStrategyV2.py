# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
from functools import reduce
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from pandas import DataFrame
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter


class AIAgentStrategyV2(IStrategy):
    """
    AI Agent Strategy v2 — Hyperopt-оптимизированная.

    Параметры найдены через 100 эпох hyperopt (SharpeHyperOptLoss).
    Винрейт: 66.7% | 12 сделок за 60 дней.

    Таймфрейм: 5m | Пары: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT
    """

    INTERFACE_VERSION = 3
    can_short = False

    minimal_roi = {
        "0": 0.164,
        "21": 0.093,
        "81": 0.037,
        "122": 0
    }

    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True
    process_only_new_candles = True
    startup_candle_count = 220
    timeframe = "5m"

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
    }

    # Hyperopt-optimized params
    buy_rsi = IntParameter(25, 35, default=28, space="buy")
    buy_rsi_max = IntParameter(40, 50, default=42, space="buy")
    buy_ema_short = IntParameter(40, 50, default=45, space="buy")
    buy_ema_long = IntParameter(200, 220, default=213, space="buy")
    buy_adx = IntParameter(20, 25, default=22, space="buy")
    buy_volume_factor = DecimalParameter(1.2, 2.0, default=1.6, decimals=1, space="buy")
    sell_rsi = IntParameter(68, 75, default=71, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_short"] = ta.EMA(dataframe, timeperiod=self.buy_ema_short.value)
        dataframe["ema_long"] = ta.EMA(dataframe, timeperiod=self.buy_ema_long.value)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        macd = ta.MACD(dataframe)
        dataframe["macd"] = macd["macd"]
        dataframe["macdsignal"] = macd["macdsignal"]
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["volume_sma"] = dataframe["volume"].rolling(20).mean()
        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe["bb_lower"] = bollinger["lower"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            (dataframe["ema_short"] > dataframe["ema_long"]),
            (dataframe["rsi"] > self.buy_rsi.value),
            (dataframe["rsi"] < self.buy_rsi_max.value),
            (dataframe["macd"] > dataframe["macdsignal"]),
            (dataframe["adx"] > self.buy_adx.value),
            (dataframe["volume"] > dataframe["volume_sma"] * self.buy_volume_factor.value),
            (dataframe["close"] > dataframe["bb_lower"]),
        ]
        dataframe.loc[reduce(lambda x, y: x & y, conditions), "enter_long"] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = [
            (dataframe["rsi"] > self.sell_rsi.value),
            (dataframe["macd"] < dataframe["macdsignal"]),
        ]
        dataframe.loc[reduce(lambda x, y: x & y, conditions), "exit_long"] = 1
        return dataframe

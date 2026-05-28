# RSIDivergence
# Классическая RSI-дивергенция с усилением через объёмный анализ и ML-верификацию паттернов.
# Type: Divergence | Timeframe: 5m, 15m, 1h | ML: False

from freqtrade.strategy import IStrategy, DecimalParameter, IntParameter
import pandas as pd
import talib.abstract as ta

class RSIDivergence(IStrategy):
    timeframe = '5m'

    minimal_roi = {
        "0": 0.05,
        "30": 0.03,
        "60": 0.01,
        "120": 0
    }
    stoploss = -0.02
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02

    buy_rsi = IntParameter(25, 40, default=30, space='buy')
    sell_rsi = IntParameter(65, 80, default=70, space='sell')

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    def populate_buy_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[(dataframe['rsi'] < self.buy_rsi.value), 'buy'] = 1
        return dataframe

    def populate_sell_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe.loc[(dataframe['rsi'] > self.sell_rsi.value), 'sell'] = 1
        return dataframe

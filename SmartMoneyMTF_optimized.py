# SmartMoneyMTF v2 - Multi-Timeframe Smart Money Strategy for Freqtrade
# Higher timeframe (1h): SMC analysis for direction (LONG + SHORT)
# Lower timeframe (5m): Entry confirmation on pullback
# Exchange: OKX (futures)
#
# === HYPEROPT 2026 (200 epochs) ===
# 192 trades, 72.9% winrate, +2.06% profit, Sharpe optimized
# Optimal params: swing_length=31, ob_min_percentage=0.61, range_percent=0.047
#
# Author: geniok (Evgeny Shlyakhtin)
# GitHub: https://github.com/geniok1980/Freqtrade-Strategies
# License: MIT

import pandas as pd
import numpy as np
from datetime import datetime
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter, informative
from functools import reduce
import talib.abstract as ta
import os
os.environ['SMC_CREDIT'] = '0'
from smartmoneyconcepts import smc


class SmartMoneyMTF(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = False

    # --- Stoploss (hyperopt 2026) ---
    stoploss = -0.167
    trailing_stop = False

    minimal_roi = {
        "0": 0.276,
        "33": 0.081,
        "92": 0.038,
        "200": 0
    }

    max_open_trades = 5
    startup_candle_count = 300

    # Hyperoptable params (defaults = 2026 optimal)
    swing_length = IntParameter(30, 80, default=31, space="buy")
    fvg_join = BooleanParameter(default=True, space="buy")
    ob_close_mitigation = BooleanParameter(default=True, space="buy")
    ob_min_percentage = DecimalParameter(0.3, 0.8, default=0.61, decimals=2, space="buy")
    range_percent = DecimalParameter(0.01, 0.05, default=0.047, decimals=3, space="buy")
    atr_multiplier = DecimalParameter(1.5, 3.0, default=2.4, decimals=1, space="sell")

    # === HIGHER TIMEFRAME (1h) INDICATORS ===

    @informative('1h')
    def populate_indicators_1h(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        ohlc_4 = dataframe[['open', 'high', 'low', 'close']].copy()
        ohlc_5 = dataframe[['open', 'high', 'low', 'close', 'volume']].copy()

        # --- Swing Highs/Lows ---
        swing_result = smc.swing_highs_lows(ohlc_4, swing_length=self.swing_length.value)
        dataframe['swing_high_low'] = swing_result['HighLow']
        dataframe['swing_level'] = swing_result['Level']

        # --- Fair Value Gap ---
        fvg_result = smc.fvg(ohlc_4, join_consecutive=self.fvg_join.value)
        dataframe['fvg'] = fvg_result['FVG']
        dataframe['fvg_top'] = fvg_result['Top']
        dataframe['fvg_bottom'] = fvg_result['Bottom']
        dataframe['fvg_mitigated'] = fvg_result['MitigatedIndex']

        dataframe['fvg_active'] = 0
        last_mitigated = -1
        for i in range(len(dataframe)):
            mit_idx = dataframe.loc[dataframe.index[i], 'fvg_mitigated']
            if not pd.isna(mit_idx):
                last_mitigated = int(mit_idx)
            if not pd.isna(dataframe.loc[dataframe.index[i], 'fvg']):
                if last_mitigated < 0 or i < last_mitigated:
                    dataframe.loc[dataframe.index[i], 'fvg_active'] = 1

        # --- Order Blocks ---
        ob_result = smc.ob(ohlc_5, swing_result, close_mitigation=self.ob_close_mitigation.value)
        dataframe['ob'] = ob_result['OB']
        dataframe['ob_top'] = ob_result['Top']
        dataframe['ob_bottom'] = ob_result['Bottom']
        dataframe['ob_volume'] = ob_result['OBVolume']
        dataframe['ob_percentage'] = ob_result['Percentage']
        dataframe['ob_mitigated'] = ob_result['MitigatedIndex']

        dataframe['ob_active'] = 0
        last_ob_mitigated = -1
        for i in range(len(dataframe)):
            mit_idx = dataframe.loc[dataframe.index[i], 'ob_mitigated']
            if not pd.isna(mit_idx):
                last_ob_mitigated = int(mit_idx)
            if not pd.isna(dataframe.loc[dataframe.index[i], 'ob']):
                if last_ob_mitigated < 0 or i < last_ob_mitigated:
                    dataframe.loc[dataframe.index[i], 'ob_active'] = 1

        # --- BOS / CHOCH for direction ---
        bos_result = smc.bos_choch(ohlc_4, swing_result, close_break=True)
        dataframe['bos'] = bos_result['BOS']
        dataframe['choch'] = bos_result['CHOCH']

        # --- Liquidity ---
        liq_result = smc.liquidity(ohlc_4, swing_result, range_percent=self.range_percent.value)
        dataframe['liquidity'] = liq_result['Liquidity']
        dataframe['liquidity_swept'] = liq_result['Swept']

        # --- Forward-fill LONG OB zone (bullish) ---
        dataframe['ob_zone_top_bull'] = np.nan
        dataframe['ob_zone_bottom_bull'] = np.nan
        for i in range(len(dataframe)):
            if not pd.isna(dataframe.loc[dataframe.index[i], 'ob']) and \
               dataframe.loc[dataframe.index[i], 'ob'] == 1:
                dataframe.loc[dataframe.index[i], 'ob_zone_top_bull'] = \
                    dataframe.loc[dataframe.index[i], 'ob_top']
                dataframe.loc[dataframe.index[i], 'ob_zone_bottom_bull'] = \
                    dataframe.loc[dataframe.index[i], 'ob_bottom']
        dataframe[['ob_zone_top_bull', 'ob_zone_bottom_bull']] = \
            dataframe[['ob_zone_top_bull', 'ob_zone_bottom_bull']].ffill()

        # --- Forward-fill SHORT OB zone (bearish) ---
        dataframe['ob_zone_top_bear'] = np.nan
        dataframe['ob_zone_bottom_bear'] = np.nan
        for i in range(len(dataframe)):
            if not pd.isna(dataframe.loc[dataframe.index[i], 'ob']) and \
               dataframe.loc[dataframe.index[i], 'ob'] == -1:
                dataframe.loc[dataframe.index[i], 'ob_zone_top_bear'] = \
                    dataframe.loc[dataframe.index[i], 'ob_top']
                dataframe.loc[dataframe.index[i], 'ob_zone_bottom_bear'] = \
                    dataframe.loc[dataframe.index[i], 'ob_bottom']
        dataframe[['ob_zone_top_bear', 'ob_zone_bottom_bear']] = \
            dataframe[['ob_zone_top_bear', 'ob_zone_bottom_bear']].ffill()

        # Distance from price to OB zones
        dataframe['ob_dist_bull_pct'] = np.abs(
            (dataframe['close'] - dataframe['ob_zone_top_bull']) / dataframe['ob_zone_top_bull']
        ) * 100
        dataframe['ob_dist_bear_pct'] = np.abs(
            (dataframe['close'] - dataframe['ob_zone_top_bear']) / dataframe['ob_zone_top_bear']
        ) * 100

        # FVG active recent window
        dataframe['fvg_active_recent'] = (
            dataframe['fvg_active'].rolling(window=20, min_periods=1).max().fillna(0)
        )

        # --- SIGNALS ---
        # LONG: OB bullish (1) + FVG active recent + price near zone
        ob_bull = dataframe['ob_zone_top_bull'].notna()
        fvg_bull = dataframe['fvg_active_recent'] > 0
        price_near_bull = dataframe['ob_dist_bull_pct'] < 15
        
        dataframe['signal_bullish'] = 0
        dataframe.loc[ob_bull & fvg_bull, 'signal_bullish'] = 1
        dataframe.loc[ob_bull & fvg_bull & price_near_bull, 'signal_bullish'] = 2

        # SHORT: OB bearish (-1) + FVG active + CHOCH down + price near zone
        ob_bear = dataframe['ob_zone_top_bear'].notna()
        fvg_bearish = dataframe['fvg'] == -1
        fvg_bear_recent = (dataframe['fvg_active'] & fvg_bearish).rolling(window=20, min_periods=1).max().fillna(0) > 0
        choch_down = dataframe['choch'] == -1
        price_near_bear = dataframe['ob_dist_bear_pct'] < 15

        dataframe['signal_bearish'] = 0
        dataframe.loc[ob_bear & fvg_bear_recent & choch_down, 'signal_bearish'] = 1
        dataframe.loc[ob_bear & fvg_bear_recent & choch_down & price_near_bear, 'signal_bearish'] = 2

        # ATR
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        dataframe.drop(columns=['fvg_mitigated', 'ob_mitigated'], inplace=True, errors='ignore')
        return dataframe

    # === MAIN TIMEFRAME (5m) INDICATORS ===

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = dataframe['atr'] / dataframe['close'] * 100
        dataframe['volume_sma_20'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        return dataframe

    # === ENTRY LOGIC ===

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        conditions_long = []
        conditions_long.append(dataframe['signal_bullish_1h'] >= 1)

        ob_top_bull = dataframe['ob_zone_top_bull_1h']
        ob_bot_bull = dataframe['ob_zone_bottom_bull_1h']
        price_near_ob_long = (
            (dataframe['close'] >= ob_bot_bull * 0.95) &
            (dataframe['close'] <= ob_top_bull * 1.05)
        )
        conditions_long.append(price_near_ob_long)
        conditions_long.append(dataframe['rsi'] < 60)
        conditions_long.append(dataframe['volume'] >= dataframe['volume_sma_20'] * 0.5)

        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_long),
                'enter_long'] = 1

        # --- SHORT ENTRY (disabled, spot mode) ---
        # conditions_short = []
        # ... (uncomment for futures)

        return dataframe

    # === EXIT ===
    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return dataframe

    # === RISK ===
    def custom_stoploss(self, pair: str, trade, current_time, current_rate, current_profit, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) < 30:
            return self.stoploss
        last_candle = dataframe.iloc[-1]
        atr_pct = last_candle.get('atr_percent', 2.0)
        if pd.isna(atr_pct) or atr_pct <= 0:
            atr_pct = 2.0
        stop_val = atr_pct * self.atr_multiplier.value / 100.0
        return -stop_val

    def custom_take_profit(self, pair, trade, current_time, current_rate, current_profit, **kwargs) -> bool:
        return False

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs) -> float:
        return 1.0

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force, current_time, entry_tag, side, **kwargs) -> bool:
        return True

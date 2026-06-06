# SmartMoneyMTF (Optimized) - Multi-Timeframe Smart Money Strategy for Freqtrade
# ============================================================================
# Higher timeframe (1h): SMC analysis (Order Blocks + Fair Value Gap)
# Lower timeframe (5m): Precise entry on pullback with volume confirmation
#
# Exchange: OKX (also works on any exchange with USDT pairs)
# Pair: BTC/USDT, ETH/USDT, SOL/USDT, and other top pairs
#
# === HYPEROPT RESULTS ===
# Best result: 98 trades, 72.4% winrate, +21.47 USDT (2.15%), Sharpe optimized
# Parameters optimized via SharpeHyperOptLossDaily, 100 epochs
#
# Author: geniok (Evgeny Shlyakhtin)
# GitHub: https://github.com/geniok1980/Freqtrade-Strategies
# License: MIT
# ============================================================================

import pandas as pd
import numpy as np
from datetime import datetime
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter, informative
from functools import reduce
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

import os
os.environ['SMC_CREDIT'] = '0'

from smartmoneyconcepts import smc


class SmartMoneyMTF(IStrategy):
    """
    Multi-Timeframe Smart Money Concepts Strategy for Freqtrade.

    Higher Timeframe (1h):
    - Bullish Order Block zone (forward-filled)
    - Active Fair Value Gap (FVG)
    - Combined signal for directional bias

    Lower Timeframe (5m):
    - Pullback / retracement into the 1h OB zone
    - Volume confirmation
    - RSI-based oversold condition

    Risk Management:
    - ATR-based dynamic stoploss (optimized: 2.5x ATR)
    - ROI table optimized for quick exits
    """

    # --- Strategy Interface Settings ---
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = False

    # --- Stoploss (optimized by hyperopt) ---
    stoploss = -0.069
    trailing_stop = False

    # --- ROI (optimized by hyperopt) ---
    minimal_roi = {
        "0": 0.155,
        "32": 0.065,
        "79": 0.018,
        "172": 0
    }

    # --- Optimal positions ---
    max_open_trades = 3
    startup_candle_count = 300

    # --- Hyperopt-optimized Parameters ---
    swing_length = IntParameter(30, 80, default=34, space="buy")
    fvg_join = BooleanParameter(default=True, space="buy")
    ob_close_mitigation = BooleanParameter(default=True, space="buy")
    ob_min_percentage = DecimalParameter(0.3, 0.8, default=0.3, decimals=2, space="buy")
    range_percent = DecimalParameter(0.01, 0.05, default=0.014, decimals=3, space="buy")
    atr_multiplier = DecimalParameter(1.5, 3.0, default=2.5, decimals=1, space="sell")

    # === HIGHER TIMEFRAME (1h) INDICATORS ===

    @informative('1h')
    def populate_indicators_1h(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Calculate SMC indicators on the 1h timeframe for directional bias.
        Columns get `_1h` suffix in the main 5m dataframe.
        """
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

        # Track active FVG (not mitigated yet)
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

        # Track active OB
        dataframe['ob_active'] = 0
        last_ob_mitigated = -1
        for i in range(len(dataframe)):
            mit_idx = dataframe.loc[dataframe.index[i], 'ob_mitigated']
            if not pd.isna(mit_idx):
                last_ob_mitigated = int(mit_idx)
            if not pd.isna(dataframe.loc[dataframe.index[i], 'ob']):
                if last_ob_mitigated < 0 or i < last_ob_mitigated:
                    dataframe.loc[dataframe.index[i], 'ob_active'] = 1

        # --- Forward-fill the bullish OB zone ---
        dataframe['ob_zone_top'] = np.nan
        dataframe['ob_zone_bottom'] = np.nan

        for i in range(len(dataframe)):
            if not pd.isna(dataframe.loc[dataframe.index[i], 'ob']) and \
               dataframe.loc[dataframe.index[i], 'ob'] == 1:
                dataframe.loc[dataframe.index[i], 'ob_zone_top'] = \
                    dataframe.loc[dataframe.index[i], 'ob_top']
                dataframe.loc[dataframe.index[i], 'ob_zone_bottom'] = \
                    dataframe.loc[dataframe.index[i], 'ob_bottom']

        dataframe[['ob_zone_top', 'ob_zone_bottom']] = \
            dataframe[['ob_zone_top', 'ob_zone_bottom']].ffill()

        # Distance from current price to OB zone
        dataframe['ob_distance_pct'] = np.abs(
            (dataframe['close'] - dataframe['ob_zone_top']) / dataframe['ob_zone_top']
        ) * 100

        # FVG active in recent 20 candles window
        dataframe['fvg_active_recent'] = (
            dataframe['fvg_active']
            .rolling(window=20, min_periods=1)
            .max()
            .fillna(0)
        )

        # Combined signal
        dataframe['signal_bullish'] = 0
        ob_zone_exists = dataframe['ob_zone_top'].notna()
        fvg_recent = dataframe['fvg_active_recent'] > 0
        price_near_zone = dataframe['ob_distance_pct'] < 15

        signal_moderate = ob_zone_exists & fvg_recent
        signal_strong = ob_zone_exists & fvg_recent & price_near_zone

        dataframe.loc[signal_moderate, 'signal_bullish'] = 1
        dataframe.loc[signal_strong, 'signal_bullish'] = 2

        # --- ATR on 1h ---
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)

        # Cleanup
        dataframe.drop(columns=['fvg_mitigated', 'ob_mitigated'],
                       inplace=True, errors='ignore')
        return dataframe

    # === MAIN TIMEFRAME (5m) INDICATORS ===

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Calculate 5m indicators for precise entry timing.
        """
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_percent'] = dataframe['atr'] / dataframe['close'] * 100
        dataframe['volume_sma_20'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['close'] > dataframe['open']
        return dataframe

    # === ENTRY LOGIC ===

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Multi-timeframe entry logic.
        Entry: 1h bullish signal (OB+FVG) + 5m pullback into OB zone + volume + RSI.
        """
        conditions_long = []

        # 1h signal: bullish setup (OB zone + recent FVG)
        conditions_long.append(dataframe['signal_bullish_1h'] >= 1)

        # 5m entry: price is near the 1h OB zone (within 5%)
        ob_top = dataframe['ob_zone_top_1h']
        ob_bot = dataframe['ob_zone_bottom_1h']

        price_near_ob = (
            (dataframe['close'] >= ob_bot * 0.95) &
            (dataframe['close'] <= ob_top * 1.05)
        )
        conditions_long.append(price_near_ob)

        # RSI < 60 (not overbought, ideally retracing)
        conditions_long.append(dataframe['rsi'] < 60)

        # Volume confirmation
        conditions_long.append(dataframe['volume'] >= dataframe['volume_sma_20'] * 0.5)

        if conditions_long:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions_long),
                'enter_long'] = 1

        return dataframe

    # === EXIT LOGIC ===

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return dataframe

    # === RISK MANAGEMENT ===

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) < 30:
            return self.stoploss

        last_candle = dataframe.iloc[-1]
        atr_pct = last_candle.get('atr_percent', 2.0)
        if pd.isna(atr_pct) or atr_pct <= 0:
            atr_pct = 2.0

        multiplier = self.atr_multiplier.value
        stop_val = atr_pct * multiplier / 100.0
        return -stop_val

    def custom_take_profit(self, pair: str, trade: 'Trade', current_time: datetime,
                           current_rate: float, current_profit: float, **kwargs) -> bool:
        return False

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        return 1.0

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str | None,
                            side: str, **kwargs) -> bool:
        return True

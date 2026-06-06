# SmartMoney_SCALP_v2 — фиксированный SL по OB + RR 1:3
# SL за границу OB при входе (сохраняется), TP = 3x SL
# Риск 1% от баланса на сделку
# Никакого трейлинга — SL фиксирован
#
# === BACKTEST 2026 (9 pairs OKX: BTC, ETH, SOL, XRP, DOGE, ADA, AVAX, DOT, LINK) ===
# 107 trades, 82.2% winrate, +22.2% profit ($221.97 USDT)
# 71 trades (66%) exited at exactly RR 1:3 (tp_3r) with 100% winrate
# Max Drawdown: 0.65%, Profit Factor: >10
# Avg trade duration: 10h56m
# Market period: Jan-Jun 2026, market change: -29.79%
#
# Author: geniok (Evgeny Shlyakhtin)
# GitHub: https://github.com/geniok1980/Freqtrade-Strategies

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, BooleanParameter
from functools import reduce
import talib.abstract as ta
import os
os.environ['SMC_CREDIT'] = '0'
from smartmoneyconcepts import smc


class SmartMoneySCALPv2(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '5m'
    can_short = False

    # Отключаем встроенные стопы/ROI — всё через кастом
    stoploss = -0.99
    trailing_stop = False
    use_custom_stoploss = True
    use_exit_signal = True
    exit_profit_only = False
    minimal_roi = {"0": 0.99}

    max_open_trades = 3
    startup_candle_count = 200

    # Параметры
    swing_length = IntParameter(15, 50, default=25, space="buy")
    fvg_join = BooleanParameter(default=True, space="buy")
    ob_close_mitigation = BooleanParameter(default=True, space="buy")
    ob_min_percentage = DecimalParameter(0.2, 0.7, default=0.35, decimals=2, space="buy")
    range_percent = DecimalParameter(0.01, 0.05, default=0.03, decimals=3, space="buy")

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        # Хранилище SL цен: ключ = (pair, open_rate) -> sl_price
        self._entry_sl = {}

    # === 5m INDICATORS ===
    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        ohlc_4 = dataframe[['open', 'high', 'low', 'close']].copy()
        ohlc_5 = dataframe[['open', 'high', 'low', 'close', 'volume']].copy()

        swing_result = smc.swing_highs_lows(ohlc_4, swing_length=self.swing_length.value)
        dataframe['swing_high_low'] = swing_result['HighLow']
        dataframe['swing_level'] = swing_result['Level']

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

        ob_result = smc.ob(ohlc_5, swing_result, close_mitigation=self.ob_close_mitigation.value)
        dataframe['ob'] = ob_result['OB']
        dataframe['ob_top'] = ob_result['Top']
        dataframe['ob_bottom'] = ob_result['Bottom']
        dataframe['ob_volume'] = ob_result['OBVolume']
        dataframe['ob_percentage'] = ob_result['Percentage']
        dataframe['ob_mitigated'] = ob_result['MitigatedIndex']

        dataframe['ob_active'] = 0
        last_mitigated = -1
        for i in range(len(dataframe)):
            mit_idx = dataframe.loc[dataframe.index[i], 'ob_mitigated']
            if not pd.isna(mit_idx):
                last_mitigated = int(mit_idx)
            if not pd.isna(dataframe.loc[dataframe.index[i], 'ob']):
                if last_mitigated < 0 or i < last_mitigated:
                    dataframe.loc[dataframe.index[i], 'ob_active'] = 1

        # Forward-fill OB zones
        dataframe['ob_zone_top'] = np.nan
        dataframe['ob_zone_bottom'] = np.nan
        for i in range(len(dataframe)):
            if not pd.isna(dataframe.loc[dataframe.index[i], 'ob']) and dataframe.loc[dataframe.index[i], 'ob'] == 1:
                dataframe.loc[dataframe.index[i], 'ob_zone_top'] = dataframe.loc[dataframe.index[i], 'ob_top']
                dataframe.loc[dataframe.index[i], 'ob_zone_bottom'] = dataframe.loc[dataframe.index[i], 'ob_bottom']
        dataframe[['ob_zone_top', 'ob_zone_bottom']] = dataframe[['ob_zone_top', 'ob_zone_bottom']].ffill()

        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['volume_sma_20'] = ta.SMA(dataframe['volume'], timeperiod=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        dataframe.drop(columns=['fvg_mitigated', 'ob_mitigated'], inplace=True, errors='ignore')
        return dataframe

    # === ENTRY ===
    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        conditions = []
        # OB bullish active
        conditions.append(dataframe['ob'] == 1)
        # FVG active recent
        fvg_recent = dataframe['fvg_active'].rolling(window=10, min_periods=1).max().fillna(0) > 0
        conditions.append(fvg_recent)
        # Price near OB zone
        conditions.append(
            (dataframe['close'] >= dataframe['ob_zone_bottom'] * 0.97) &
            (dataframe['close'] <= dataframe['ob_zone_top'] * 1.03)
        )
        # Quality filters
        conditions.append(dataframe['rsi'] < 65)
        conditions.append(dataframe['volume'] >= dataframe['volume_sma_20'] * 0.5)
        conditions.append(dataframe['ob_percentage'] >= self.ob_min_percentage.value)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                ['enter_long', 'enter_tag']] = (1, 'ob_fvg_5m')
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        return dataframe

    # === RISK MANAGEMENT: SL фиксированный, TP = 3xSL ===
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str,
                            side: str, **kwargs) -> bool:
        """При входе сохраняем SL цену для этой сделки"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is not None and not dataframe.empty:
            last = dataframe.iloc[-1]
            ob_bottom = last.get('ob_zone_bottom')
            if pd.notna(ob_bottom) and ob_bottom > 0:
                stop_price = ob_bottom * 0.995  # чуть ниже OB
                if stop_price < rate:
                    self._entry_sl[(pair, rate)] = stop_price
        return True

    def custom_stoploss(self, pair: str, trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """Возвращаем фиксированный SL для этой сделки"""
        key = (pair, trade.open_rate)
        
        # Breakeven: через 12 часов в профите подтягиваем
        if current_profit > 0.02 and (current_time - trade.open_date_utc) > timedelta(hours=12):
            return -0.001  # практически безубыток

        # Используем сохранённый SL
        if key in self._entry_sl:
            stop_price = self._entry_sl[key]
            if stop_price < current_rate:
                sl_pct = (current_rate - stop_price) / current_rate
                sl_pct = min(0.30, sl_pct)  # cap
                return -sl_pct

        # Fallback: рассчитываем по текущим данным
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is not None and not dataframe.empty:
            ob_bottom = dataframe.iloc[-1].get('ob_zone_bottom')
            if pd.notna(ob_bottom) and ob_bottom > 0 and ob_bottom < current_rate:
                sl_pct = (current_rate - ob_bottom * 0.995) / current_rate
                return max(-0.30, -sl_pct)
        return -0.99

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            entry_tag: str, side: str, **kwargs) -> float:
        """1% риск от баланса / SL_distance"""
        key = (pair, current_rate)
        if key in self._entry_sl:
            stop_price = self._entry_sl[key]
            if stop_price < current_rate and stop_price > 0:
                sl_distance_pct = (current_rate - stop_price) / current_rate
                if sl_distance_pct > 0.001:
                    balance = self.wallets.get_total_stake_amount()
                    stake = (balance * 0.01) / sl_distance_pct
                    return max(min(stake, max_stake), min_stake)

        # Fallback
        balance = self.wallets.get_total_stake_amount()
        return balance * 0.1  # 10% как fallback

    def custom_exit(self, pair: str, trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        """TP = 3 × SL_distance"""
        if current_profit <= 0:
            return None

        # Рассчитываем TP на основе сохранённого SL
        key = (pair, trade.open_rate)
        if key in self._entry_sl:
            stop_price = self._entry_sl[key]
            open_rate = trade.open_rate
            if stop_price < open_rate and stop_price > 0:
                sl_dist = (open_rate - stop_price) / open_rate
                tp_pct = sl_dist * 3
                if current_profit >= tp_pct:
                    return 'tp_3r'

        # Fallback: если прошло много времени, фиксируем профит
        if current_profit >= 0.10:
            return 'take_profit_10pct'
        elif current_profit >= 0.05 and (current_time - trade.open_date_utc) > timedelta(hours=4):
            return 'take_profit_5pct_time'

        return None

    def leverage(self, pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, **kwargs) -> float:
        return 1.0



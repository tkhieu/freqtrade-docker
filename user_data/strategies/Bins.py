from datetime import datetime
from functools import reduce

from freqtrade.persistence import Trade
from freqtrade.strategy import IntParameter, DecimalParameter, stoploss_from_open, CategoricalParameter, informative
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
import freqtrade.vendor.qtpylib.indicators as qtpylib

import talib.abstract as ta
import numpy  # noqa


class Bins(IStrategy):
    """

        strategy sponsored by user BinH from slack

    """

    minimal_roi = {
        "0": 1
    }

    buy_params = {
        "buy_adx1": 62,
        "buy_adx2": 29,
        "buy_adx3": 33,
        "buy_adx4": 88,
        "buy_emarsi1": 29,
        "buy_emarsi2": 30,
        "buy_emarsi3": 22,
        "buy_emarsi4": 57
    }

    sell_params = {
        "adx2": 21,
        "emarsi1": 30,
        "emarsi2": 71,
        "emarsi3": 72,
        "leverage_num": 3,
        "sell_1": False,
        "sell_2": True,
        "sell_3": True,
        "sell_4": True,
        "sell_5": False,
        "pHSL": -0.75,
        "pPF_1": 0.03,
        "pPF_2": 0.09,
        "pSL_1": 0.028,
        "pSL_2": 0.08
    }

    stoploss = -0.99
    timeframe = '5m'

    process_only_new_candles = True
    startup_candle_count = 240

    # default False
    use_custom_stoploss = True

    can_short = True

    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': False,

        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }

    # buy params
    buy_optimize = True
    buy_adx1 = IntParameter(low=10, high=100, default=25, space='buy', optimize=buy_optimize)
    buy_emarsi1 = IntParameter(low=10, high=100, default=20, space='buy', optimize=buy_optimize)
    buy_adx2 = IntParameter(low=20, high=100, default=30, space='buy', optimize=buy_optimize)
    buy_emarsi2 = IntParameter(low=20, high=100, default=20, space='buy', optimize=buy_optimize)
    buy_adx3 = IntParameter(low=10, high=100, default=35, space='buy', optimize=buy_optimize)
    buy_emarsi3 = IntParameter(low=10, high=100, default=20, space='buy', optimize=buy_optimize)
    buy_adx4 = IntParameter(low=20, high=100, default=30, space='buy', optimize=buy_optimize)
    buy_emarsi4 = IntParameter(low=20, high=100, default=25, space='buy', optimize=buy_optimize)

    # trailing stoploss
    trailing_optimize = False
    pHSL = DecimalParameter(-0.990, -0.040, default=-0.08, decimals=3, space='sell', optimize=trailing_optimize)
    pPF_1 = DecimalParameter(0.008, 0.100, default=0.016, decimals=3, space='sell', optimize=trailing_optimize)
    pSL_1 = DecimalParameter(0.008, 0.100, default=0.011, decimals=3, space='sell', optimize=trailing_optimize)
    pPF_2 = DecimalParameter(0.040, 0.200, default=0.080, decimals=3, space='sell', optimize=trailing_optimize)
    pSL_2 = DecimalParameter(0.040, 0.200, default=0.040, decimals=3, space='sell', optimize=trailing_optimize)

    # sell params
    sell_optimize = True
    adx2 = IntParameter(low=10, high=100, default=30, space='sell', optimize=sell_optimize)
    emarsi1 = IntParameter(low=10, high=100, default=75, space='sell', optimize=sell_optimize)
    emarsi2 = IntParameter(low=20, high=100, default=80, space='sell', optimize=sell_optimize)
    emarsi3 = IntParameter(low=20, high=100, default=75, space='sell', optimize=sell_optimize)

    sell2_optimize = True
    sell_1 = CategoricalParameter([True, False], default=True, space="sell", optimize=sell2_optimize)
    sell_2 = CategoricalParameter([True, False], default=True, space="sell", optimize=sell2_optimize)
    sell_3 = CategoricalParameter([True, False], default=True, space="sell", optimize=sell2_optimize)
    sell_4 = CategoricalParameter([True, False], default=True, space="sell", optimize=sell2_optimize)
    sell_5 = CategoricalParameter([True, False], default=True, space="sell", optimize=sell2_optimize)

    leverage_optimize = False
    leverage_num = IntParameter(low=1, high=20, default=1, space='sell', optimize=leverage_optimize)

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:

        # hard stoploss profit
        HSL = self.pHSL.value
        PF_1 = self.pPF_1.value
        SL_1 = self.pSL_1.value
        PF_2 = self.pPF_2.value
        SL_2 = self.pSL_2.value

        # For profits between PF_1 and PF_2 the stoploss (sl_profit) used is linearly interpolated
        # between the values of SL_1 and SL_2. For all profits above PL_2 the sl_profit value
        # rises linearly with current profit, for profits below PF_1 the hard stoploss profit is used.

        if current_profit > PF_2:
            sl_profit = SL_2 + (current_profit - PF_2)
        elif current_profit > PF_1:
            sl_profit = SL_1 + ((current_profit - PF_1) * (SL_2 - SL_1) / (PF_2 - PF_1))
        else:
            sl_profit = HSL

        if self.can_short:
            if (-1 + ((1 - sl_profit) / (1 - current_profit))) <= 0:
                return 1
        else:
            if (1 - ((1 + sl_profit) / (1 + current_profit))) <= 0:
                return 1

        return stoploss_from_open(sl_profit, current_profit, is_short=trade.is_short)

    @informative('1h', 'BTC/{stake}', fmt='{base}_{column}_{timeframe}')
    def populate_indicators_btc_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['ema_fast'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_slow'] = ta.EMA(dataframe, timeperiod=25)
        dataframe['uptrend'] = (dataframe['ema_fast'] > dataframe['ema_slow']).astype('int')
        dataframe['downtrend'] = (dataframe['ema_fast'] < dataframe['ema_slow']).astype('int')

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi'] = numpy.nan_to_num(ta.RSI(dataframe, timeperiod=5))
        rsiframe = DataFrame(dataframe['rsi']).rename(columns={'rsi': 'close'})
        dataframe['emarsi'] = numpy.nan_to_num(ta.EMA(rsiframe, timeperiod=5))
        dataframe['adx'] = numpy.nan_to_num(ta.ADX(dataframe))
        dataframe['minusdi'] = numpy.nan_to_num(ta.MINUS_DI(dataframe))
        minusdiframe = DataFrame(dataframe['minusdi']).rename(columns={'minusdi': 'close'})
        dataframe['minusdiema'] = numpy.nan_to_num(ta.EMA(minusdiframe, timeperiod=25))
        dataframe['plusdi'] = numpy.nan_to_num(ta.PLUS_DI(dataframe))
        plusdiframe = DataFrame(dataframe['plusdi']).rename(columns={'plusdi': 'close'})
        dataframe['plusdiema'] = numpy.nan_to_num(ta.EMA(plusdiframe, timeperiod=5))
        dataframe['lowsma'] = numpy.nan_to_num(ta.EMA(dataframe, timeperiod=60))
        dataframe['highsma'] = numpy.nan_to_num(ta.EMA(dataframe, timeperiod=120))
        dataframe['fastsma'] = numpy.nan_to_num(ta.SMA(dataframe, timeperiod=120))
        dataframe['slowsma'] = numpy.nan_to_num(ta.SMA(dataframe, timeperiod=240))
        dataframe['bigup'] = dataframe['fastsma'].gt(dataframe['slowsma']) & (
                (dataframe['fastsma'] - dataframe['slowsma']) > dataframe['close'] / 300)
        dataframe['bigdown'] = ~dataframe['bigup']
        dataframe['trend'] = dataframe['fastsma'] - dataframe['slowsma']
        dataframe['preparechangetrend'] = dataframe['trend'].gt(dataframe['trend'].shift())
        dataframe['preparechangetrendconfirm'] = dataframe['preparechangetrend'] & dataframe['trend'].shift().gt(
            dataframe['trend'].shift(2))
        dataframe['continueup'] = dataframe['slowsma'].gt(dataframe['slowsma'].shift()) & dataframe[
            'slowsma'].shift().gt(dataframe['slowsma'].shift(2))
        dataframe['delta'] = dataframe['fastsma'] - dataframe['fastsma'].shift()
        dataframe['slowingdown'] = dataframe['delta'].lt(dataframe['delta'].shift())

        dataframe['short'] = (dataframe['btc_downtrend_1h'] == 1)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''

        buy_1 = (
                dataframe['slowsma'].gt(0) & (dataframe['short']) &
                dataframe['close'].lt(dataframe['highsma']) &
                dataframe['close'].lt(dataframe['lowsma']) &
                dataframe['minusdi'].gt(dataframe['minusdiema']) &
                dataframe['rsi'].ge(dataframe['rsi'].shift()) &
                ~dataframe['preparechangetrend'] &
                ~dataframe['continueup'] &
                dataframe['adx'].gt(self.buy_adx1.value) &
                dataframe['bigdown'] &
                dataframe['emarsi'].le(self.buy_emarsi1.value)
        )

        buy_2 = (
                dataframe['slowsma'].gt(0) & (dataframe['short']) &
                dataframe['close'].lt(dataframe['highsma']) &
                dataframe['close'].lt(dataframe['lowsma']) &
                dataframe['minusdi'].gt(dataframe['minusdiema']) &
                dataframe['rsi'].ge(dataframe['rsi'].shift()) &
                ~dataframe['preparechangetrend'] &
                dataframe['continueup'] &
                dataframe['adx'].gt(self.buy_adx2.value) &
                dataframe['bigdown'] &
                dataframe['emarsi'].le(self.buy_emarsi2.value)
        )

        buy_3 = (
                dataframe['slowsma'].gt(0) & (dataframe['short']) &
                dataframe['close'].lt(dataframe['highsma']) &
                dataframe['close'].lt(dataframe['lowsma']) &
                dataframe['minusdi'].gt(dataframe['minusdiema']) &
                dataframe['rsi'].ge(dataframe['rsi'].shift()) &
                ~dataframe['continueup'] &
                dataframe['adx'].gt(self.buy_adx3.value) &
                dataframe['bigup'] &
                dataframe['emarsi'].le(self.buy_emarsi3.value)
        )

        buy_4 = (
                dataframe['slowsma'].gt(0) & (dataframe['short']) &
                dataframe['close'].lt(dataframe['highsma']) &
                dataframe['close'].lt(dataframe['lowsma']) &
                dataframe['minusdi'].gt(dataframe['minusdiema']) &
                dataframe['rsi'].ge(dataframe['rsi'].shift()) &
                dataframe['continueup'] &
                dataframe['adx'].gt(self.buy_adx4.value) &
                dataframe['bigup'] &
                dataframe['emarsi'].le(self.buy_emarsi4.value)
        )

        conditions.append(buy_1)
        dataframe.loc[buy_1, 'enter_tag'] += 'buy_1'

        conditions.append(buy_2)
        dataframe.loc[buy_2, 'enter_tag'] += 'buy_2'

        conditions.append(buy_3)
        dataframe.loc[buy_3, 'enter_tag'] += 'buy_3'

        conditions.append(buy_4)
        dataframe.loc[buy_4, 'enter_tag'] += 'buy_4'

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_short'] = 1

        dataframe.loc[(), ['enter_long', 'enter_tag']] = (0, 'long_in')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[(), ['exit_short', 'exit_tag']] = (0, 'short_out')
        dataframe.loc[(), ['exit_long', 'exit_tag']] = (0, 'long_out')
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1].squeeze()

        if current_profit >= self.pPF_1.value:
            return None

        if self.sell_1.value:
            if (
                    (~last_candle['preparechangetrendconfirm'])
                    and (~last_candle['continueup'])
                    and (last_candle['close'] > last_candle['lowsma'] or last_candle['close'] > last_candle['highsma'])
                    and (last_candle['highsma'] > 0)
                    and (last_candle['bigdown'])
            ):
                return "sell_1"

        if self.sell_2.value:
            if (
                    (~last_candle['preparechangetrendconfirm'])
                    and (~last_candle['continueup'])
                    and (last_candle['close'] > last_candle['highsma'])
                    and (last_candle['highsma'] > 0)
                    and (last_candle['emarsi'] > self.emarsi1.value or last_candle['close'] > last_candle['slowsma'])
                    and (last_candle['bigdown'])
            ):
                return "sell_2"

        if self.sell_3.value:
            if (
                    (~last_candle['preparechangetrendconfirm'])
                    and (last_candle['close'] > last_candle['highsma'])
                    and (last_candle['highsma'] > 0)
                    and (last_candle['adx'] > self.adx2.value)
                    and (last_candle['emarsi'] >= self.emarsi2.value)
                    and (last_candle['bigup'])
            ):
                return "sell_3"

        if self.sell_4.value:
            if (
                    (last_candle['preparechangetrendconfirm'])
                    and (~last_candle['continueup'])
                    and (last_candle['slowingdown'])
                    and (last_candle['emarsi'] >= self.emarsi3.value)
                    and (last_candle['slowsma'] > 0)
            ):
                return "sell_4"

        if self.sell_5.value:
            if (
                    (last_candle['preparechangetrendconfirm'])
                    and (last_candle['minusdi'] < last_candle['plusdi'])
                    and (last_candle['close'] > last_candle['lowsma'])
                    and (last_candle['slowsma'] > 0)
            ):
                return "sell_5"

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, side: str,
                 **kwargs) -> float:

        return self.leverage_num.value

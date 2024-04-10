# %%
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import collections
from collections import defaultdict
import string
import copy
import pandas as pd
import numpy as np
import math

empty_dict = {'STARFRUIT' : 0, 'AMETHYSTS' : 0}
empty_dict_cache = {'STARFRUIT' : np.array([]), 'AMETHYSTS' : np.array([])}


import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any

class Logger:
    def __init__(self) -> None:
        self.logs = ""

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        print(json.dumps([
            self.compress_state(state),
            self.compress_orders(orders),
            conversions,
            trader_data,
            self.logs,
        ], cls=ProsperityEncoder, separators=(",", ":")))

        self.logs = ""

    def compress_state(self, state: TradingState) -> list[Any]:
        return [
            state.timestamp,
            state.traderData,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

logger = Logger()


class Trader:
    
    position = copy.deepcopy(empty_dict)
    POSITION_LIMIT = {'STARFRUIT' : 20, 'AMETHYSTS' : 20}
    spread_cache = copy.deepcopy(empty_dict_cache)
    bid_cache = copy.deepcopy(empty_dict_cache)
    ask_cache = copy.deepcopy(empty_dict_cache)
    mid_cache = copy.deepcopy(empty_dict_cache)
    skew_cache = copy.deepcopy(empty_dict_cache)

    def values_extract(self, order_dict, buy=0):
        tot_vol = 0
        best_val = -1
        mxvol = -1

        for ask, vol in order_dict.items():
            if (buy == 0):
                vol *= -1
            tot_vol += vol
            if tot_vol > mxvol:
                mxvol = vol
                best_val = ask
        
        return tot_vol, best_val
    
    def values_extract_depth(self, order_dict):

      price = np.array([])
      vol = np.array([])

      for i, j in order_dict.items():
          price = np.append(price, abs(i))
          vol = np.append(vol, abs(j))
      
      return price, vol
    
    def replace_outlier(self, mid):
        median = np.median(mid)
        quartile_1, quartile_3 = np.percentile(mid, [25, 75])
        iqr = quartile_3 - quartile_1
        lower_bound = median - iqr
        upper_bound = median + iqr
        return np.where((mid < lower_bound) | (mid > upper_bound), median, mid)

    def AMETHYSTS_bs(self, best_bid, best_ask):
        
        if best_ask <= 10000:       
            return 'buy'
        elif best_bid >= 10000:
            return 'sell'
        else:
            return 'MM'
        
    def STARFRUIT_bs(self, best_bid, best_ask, mid_price_cache):

        product = 'STARFRUIT'
        mid_prices = mid_price_cache[product]

        prev_mid_price = mid_prices[-1] if len(mid_prices) > 1 else 0
        bid_spike = best_bid > prev_mid_price
        ask_spike = best_ask < prev_mid_price
        price_spread = best_ask - best_bid

        if ask_spike and price_spread <= 3:
            return 'buy'
        elif bid_spike and price_spread <= 3:
            return 'sell'
        else: 
            return 'MM'
    
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:

        # print("traderData: " + state.traderData)
        # print("Observations: " + str(state.observations))
        conversions = 0
        trader_data = ""
                          
        for key, val in state.position.items():
          self.position[key] = val

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            
          # get order data
          order_depth: OrderDepth = state.order_depths[product]

          # Initialize the list of Orders to be sent as an empty list
          orders: List[Order] = []        

          best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
          best_ask2, best_ask_amount2 = list(order_depth.sell_orders.items())[1] if len(list(order_depth.sell_orders.items())) > 1 else [0,0]
          best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
          best_bid2, best_bid_amount2 = list(order_depth.buy_orders.items())[1] if len(list(order_depth.buy_orders.items())) > 1 else [0,0]

          
          ###############################
          ## strategy engineering here ##
          ###############################
          price_spread = best_ask - best_bid
        
          # mid price info
          mid_price = (best_bid + best_ask) / 2
         
          # skew info
          skew = np.log(best_ask) - np.log(best_bid) * 100

          ################
          ## Conditions ##
          ################

          curr_pos = self.position[product]
          pos_limit = self.POSITION_LIMIT[product]
          weight = 1

          if product == 'STARFRUIT':
              cond = self.STARFRUIT_bs(best_bid, best_ask, self.mid_cache)
              buy_price, sell_price = best_ask, best_bid
              if cond == 'MM':
                  buy_price, sell_price = best_bid + 1, best_ask - 1 

          if product == 'AMETHYSTS':
              cond = self.AMETHYSTS_bs(best_bid, best_ask)
              buy_price, sell_price = best_ask, best_bid
              if cond == 'MM':
                buy_price, sell_price = best_bid + 1, best_ask - 1  # i quote the bid, CP wants to sell at bid, i am long

          logger.print(product, 'current pos', curr_pos)

          buy_amount = min(pos_limit, min(best_bid_amount, pos_limit - curr_pos))
          sell_amount = min(pos_limit, abs(min(abs(best_ask_amount), -pos_limit - curr_pos)))

              
   
            ### BUY ### 
          if cond == 'buy':
            curr_buy_amount = math.ceil(buy_amount * weight)
            curr_sell_amount = -math.floor(buy_amount * (1 - weight)) if abs(-math.floor(buy_amount * (1 - weight)) + curr_pos ) <= pos_limit else 0

            orders.append(Order(product, buy_price, curr_buy_amount))
            orders.append(Order(product, sell_price, curr_sell_amount))
            logger.print('BUYYYY')
            logger.print(f'buying {curr_buy_amount} {product} @ {best_ask+1}')
            logger.print(f'selling {curr_sell_amount} {product} @ {best_ask+2}')
                

            ### SELL ###
          if cond == 'sell':
            curr_sell_amount = -math.ceil(sell_amount * weight)
            curr_buy_amount = math.floor(sell_amount * (1 - weight)) if abs(math.floor(sell_amount * (1 - weight)) + curr_pos ) <= pos_limit else 0

            orders.append(Order(product, sell_price, curr_sell_amount))
            orders.append(Order(product, buy_price, curr_buy_amount))
            logger.print('SELLLLL')
            logger.print(f'selling {curr_sell_amount} {product} @ {best_bid-1}')
            logger.print(f'buying {curr_buy_amount} {product} @ {best_bid-2}')

          if cond == 'MM':
            orders.append(Order(product, sell_price, -sell_amount))
            orders.append(Order(product, buy_price, buy_amount))
            logger.print('MM')
            logger.print(f'quoting bid {buy_price}{product} {buy_amount}')
            logger.print(f'quoting ask {sell_price} {product} {-sell_amount}')

              

          window = 10
          caches = [self.spread_cache, self.ask_cache, self.bid_cache, self.mid_cache, self.skew_cache]
          # Appending Cache
          self.spread_cache[product] = np.append(self.spread_cache[product], price_spread)
          self.ask_cache[product] = np.append(self.ask_cache[product], best_ask)
          self.bid_cache[product] = np.append(self.bid_cache[product], best_bid)
          self.mid_cache[product] = np.append(self.mid_cache[product], mid_price)
          self.skew_cache[product] = np.append(self.skew_cache[product], skew)

          # Controlling size of cache
          if len(self.spread_cache[product]) > window:
            for i in caches:
                i[product] = i[product][-window:]  
              
          
          result[product] = orders
        
				# Sample conversion request. Check more details below. 
        conversions = 0

        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
# import backtrader as bt
import yfinance as yf
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import random
import math
#sktime, prophet, kats, darts, autots, tsfresh
#implied vol vs historical vol

import warnings
warnings.filterwarnings("ignore")

class Summary:
    def __init__(self, underlying, triple, start, end):
        self.UNDERLYING = underlying
        self.TRIPLE = triple
        self.STARTING_VALUE_UNDER = 3000
        self.STARTING_VALUE_TRIP = -1000
        self.TICKERS = [underlying, triple, "^VIX"]
        self.cash = 0

        self.data = yf.download(self.TICKERS,start=start, end = end, group_by="ticker", auto_adjust=False)
        self.dat_under = self.data[underlying]['Close']
        self.dat_trip = self.data[triple]['Close']

        dates = [d.strftime('%Y-%m-%d') for d in self.data.index.date]
        self.dfr = pd.DataFrame(data = {"VIX": self.data["^VIX"]['Close']})
        # self.dfr = pd.DataFrame(data = {f"{underlying}_Close": self.dat_under})

        self.shares_under = self.STARTING_VALUE_UNDER / self.dat_under[0]
        self.shares_trip = self.STARTING_VALUE_TRIP / self.dat_trip[0]
        
        
        self.dfr.index = dates
        self.dfr[f"{underlying}_Close"] = self.dat_under
        self.dfr[f"{underlying}_per_change"] = 1
        self.dfr[f"{underlying}_cumulative"] = 1
        self.dfr[f"{triple}_Close"] = self.dat_trip
        self.dfr[f"{triple}_per_change"] = 1
        self.dfr[f"{triple}_cumulative"] = 1
        # self.dfr[f"{triple}_log_per_change"] = 1
        # self.dfr[f"{triple}_daily_volatility"] = 1
        self.dfr[f"ideal_per_change"] = 1
        self.dfr[f"ideal_cumulative"] = 1
        self.dfr[f"shares_{underlying}"] = 0
        self.dfr[f"shares_{triple}"] = 0
        self.dfr["P/L"] = 0

        # self.daily_log_returns = self.daily_log_returns(self.dat_trip)


    def daily_log_returns(self, price: pd.Series) -> pd.Series:
        """Log returns r_t = ln(P_t / P_{t-1})."""
        r = np.log(price / price.shift(1))
        return r.dropna()

    def summary_stats(self):
        for i in range(1, len(self.dat_under)):
            self.dfr[f"{self.UNDERLYING}_per_change"][i] = self.dat_under[i] / self.dat_under[i-1] - 1
            self.dfr[f"{self.TRIPLE}_per_change"][i] = self.dat_trip[i] / self.dat_trip[i-1] - 1
            self.dfr["ideal_per_change"][i] = (self.dat_under[i] / self.dat_under[i-1] - 1) *3

            self.dfr[f"{self.UNDERLYING}_cumulative"][i] = self.dfr[f"{self.UNDERLYING}_cumulative"][i-1]*(1 + self.dfr[f"{self.UNDERLYING}_per_change"][i])
            self.dfr[f"{self.TRIPLE}_cumulative"][i] = self.dfr[f"{self.TRIPLE}_cumulative"][i-1]*(1 + self.dfr[f"{self.TRIPLE}_per_change"][i])
            self.dfr["ideal_cumulative"][i] = self.dfr["ideal_cumulative"][i-1]*(1 + self.dfr["ideal_per_change"][i])

            # self.dfr[f"{self.TRIPLE}_log_per_change"][i] *= np.log(self.dfr[f"{self.UNDERLYING}_per_change"][i] - 1)
            # var_d = self.daily_log_returns.rolling(window=20).var(ddof=1)
            # self.dfr[f"{self.TRIPLE}_daily_volatility"][i] = np.sqrt(var_d)

        self.dfr[f"{self.UNDERLYING}_per_change"] *= 100
        self.dfr[f"{self.TRIPLE}_per_change"] *= 100
        self.dfr["ideal_per_change"] *= 100
        self.dfr[f"{self.UNDERLYING}_cumulative"] *= 100
        self.dfr[f"{self.TRIPLE}_cumulative"] *= 100
        self.dfr["ideal_cumulative"] *= 100
        # self.dfr[f"{self.TRIPLE}_log_per_change"] *= 100

    def calc_returns(self):
        quart_under = self.data[self.UNDERLYING]['Adj Close'].resample('ME').last().pct_change()*3
        quart_trip = self.data[self.TRIPLE]['Adj Close'].resample('ME').last().pct_change()
        plt.plot(quart_under, label = f"{self.UNDERLYING} quarterly returns, beta adjusted", linewidth=0.85)
        plt.plot(quart_trip, label = f"{self.TRIPLE} quarterly returns", linewidth = 0.85)
        plt.legend()
        plt.grid(True)
        plt.show()
        quart_under = round(quart_under, 3)
        quart_trip = round(quart_trip, 3)

        d = pd.DataFrame(data = {(f"{self.UNDERLYING} quarterly returns (%)"): quart_under, f"{self.TRIPLE} quarterly returns (%)": quart_trip})
    
    
    def init_portfolio(self):
        self.dfr[f"portfolio_{self.UNDERLYING}_long"]= self.STARTING_VALUE_UNDER
        self.dfr[f"portfolio_{self.TRIPLE}_short"]= self.STARTING_VALUE_TRIP
        self.dfr["portfolio_total"]= self.STARTING_VALUE_TRIP + self.STARTING_VALUE_UNDER
        self.dfr["cash_used"] = 0
        self.dfr["beta_exposure"] = 0
        self.dfr["beta_exposure_per"] = 0
        self.dfr["cumulative_P/L"] = 0
        


    def plot_portfolio(self):
        self.dfr = round(self.dfr, 2)
        self.dfr.to_csv('new.csv')
        plt.plot(self.dfr[f"portfolio_{self.UNDERLYING}_long"], label = f"portfolio_{self.UNDERLYING}_long", linewidth=0.85)
        plt.plot(self.dfr[f"portfolio_{self.TRIPLE}_short"], label = f"portfolio_{self.TRIPLE}_short", linewidth=0.85)
        plt.plot(self.dfr['portfolio_total'], label = "portfolio_total", linewidth=0.85)
        plt.plot(self.dfr['beta_exposure'], label = "beta_exposure", linewidth=0.85)
        plt.plot(self.dfr["cumulative_P/L"], label = "cumulative_P/L", linewidth=.85)

        plt.legend()
        plt.grid(True)
        plt.savefig("new.png")
        plt.show()


    def update_portfolio(self, i):      
        self.dfr[f"portfolio_{self.UNDERLYING}_long"][i] = self.shares_under * self.dfr[f"{self.UNDERLYING}_Close"][i]
        self.dfr[f"portfolio_{self.TRIPLE}_short"][i] = self.shares_trip * self.dfr[f"{self.TRIPLE}_Close"][i]
        self.dfr["portfolio_total"][i] = self.dfr[f"portfolio_{self.UNDERLYING}_long"][i] + self.dfr[f"portfolio_{self.TRIPLE}_short"][i]
        self.dfr["beta_exposure"][i] = (self.shares_under * self.dfr[f"{self.UNDERLYING}_Close"][i]) + (self.shares_trip * self.dfr[f"{self.TRIPLE}_Close"][i] * 3)
        self.dfr["beta_exposure_per"][i] = self.dfr["beta_exposure"][i] / self.dfr[f"portfolio_{self.UNDERLYING}_long"][i]
        
        #num shares * share price * -3 + num shares * share price


    def no_trades(self):
        for i in range(1, len(self.dat_under)):
            self.dfr[f"portfolio_{self.UNDERLYING}_long"][i] = self.shares_under * self.dfr[f"{self.UNDERLYING}_Close"][i]
            self.dfr[f"portfolio_{self.TRIPLE}_short"][i] = self.shares_trip * self.dfr[f"{self.TRIPLE}_Close"][i]
            self.dfr["portfolio_total"][i] = self.dfr[f"portfolio_{self.UNDERLYING}_long"][i] + self.dfr[f"portfolio_{self.TRIPLE}_short"][i]
            self.dfr["beta_exposure"][i] = (self.shares_under * self.dfr[f"{self.UNDERLYING}_Close"][i]) + (self.shares_trip * self.dfr[f"{self.TRIPLE}_Close"][i] * 3)

    def find_optimal_exposure_indicator(self):
        x = np.zeros(5)
        y = np.zeros(5)
        counter = 0
        start = .01
        end = .05
        step = .01
        i = start
        while (i <= end):
            self.summary_stats()
            self.init_portfolio()
            x[counter] = i
            y[counter] = self.beta_norm_strategy(i)
            counter += 1

            i += step

        # for j in range(5):
        #     print(x[j])
        #     print(y[j])
        plt.plot(x,y)
        plt.show()
        
        

    def beta_norm_strategy(self, exposure_indicator):
        for i in range(1, len(self.dat_under)):
            # print(f"__________DAY {i}______________")

            self.update_portfolio(i)
            
            if (self.dfr["VIX"][i] <= 15):
                exposure_indicator = .01
            elif (self.dfr["VIX"][i] > 15 and self.dfr["VIX"][i] <= 20):
                exposure_indicator = .03
            elif (self.dfr["VIX"][i] > 20 and self.dfr["VIX"][i] <= 30):
                exposure_indicator = .06
            else:
                exposure_indicator = .1
            if (self.dfr['beta_exposure'][i] > (exposure_indicator * self.dfr[f"portfolio_{self.UNDERLYING}_long"][i])):
                # print(f"beta exposure: {self.dfr['beta_exposure'][i]}")
                beta = self.dfr['beta_exposure'][i]
                amt = beta/3/self.dfr[f"{self.TRIPLE}_Close"][i]
                self.shares_trip -= amt 
                # print(f"selling {amt} shares of SPY.")
                self.cash -= amt * self.dfr[f"{self.TRIPLE}_Close"][i]
                self.update_portfolio(i)



            elif (self.dfr['beta_exposure'][i] < -1 * (exposure_indicator * self.dfr[f"portfolio_{self.UNDERLYING}_long"][i])):
                # print(f"beta exposure: {self.dfr['beta_exposure'][i]}")
                beta = self.dfr['beta_exposure'][i]
                amt = -1*beta/3/self.dfr[f"{self.TRIPLE}_Close"][i]
                self.shares_trip += amt
                # print(f"buying {amt} shares of SPXL.")
                self.cash += amt * self.dfr[f"{self.TRIPLE}_Close"][i]
                self.update_portfolio(i)



            self.dfr["cash_used"][i] = self.cash

            cash_change = self.dfr["cash_used"][i] - self.dfr["cash_used"][i-1]
            self.dfr["P/L"][i] = self.dfr["portfolio_total"][i] - self.dfr["portfolio_total"][i-1] - cash_change
            self.dfr["cumulative_P/L"][i] = self.dfr["cumulative_P/L"][i-1] + self.dfr["P/L"][i]
            # print(f"beta exposure: {self.dfr['beta_exposure'][i]}")
            
            self.dfr[f"shares_{self.UNDERLYING}"][i] = self.shares_under
            self.dfr[f"shares_{self.TRIPLE}"][i] = self.shares_trip
        return(self.dfr["P/L"].sum())
    
    # def calculate_pl(self):
    #     for i in range(1, len(self.dat_under)):


        
#boyl vs ung (boyl is triple natural gas)
#TQQQ vs QQQ
#UDOW vs DIA
#DUSL vs XLI
#XBI vs LABU
#FAS vs XLF
#ERX vs XLE
#TNA vs IWM

#for loop for each etf and its triple
#generate random time intervals eg. 5 yrs, 20 yrs, 50 yrs
#generate a random start point since inception to present-random time interval
#run the main function, extract gap and p/l
#






























# x = np.zeros(29)
# y = np.zeros(29)
# counter = 0
# start = .05
# end = .3
# step = .01
# i = start
# df_yr = pd.DataFrame()

# while (i <= end):
#     spy = Summary(underlying="SPY", triple="SPXU")
#     spy.summary_stats()
#     spy.init_portfolio()
#     x[counter] = i
#     y[counter] = spy.beta_norm_strategy(i)
#     print(y[counter])
#     counter += 1

#     i = round(step+i, 2)


#     # df_yr["SOXL_close"] = spy.dfr["SOXL_Close"]
#     df_yr[f"{i}"] = spy.dfr["cumulative_P/L"]

#     # df_yr[f"{i} shares SOXL"]   = spy.dfr[f"shares_SOXL"]
#     plt.plot(spy.dfr["cumulative_P/L"], label = f"Adjust: {float(i)}", linewidth=.85)
#     # plt.plot(spy.dfr[f"shares_{spy.TRIPLE}"]*10, label = f"{i} shares SOXL", linewidth = .85)
#     # plt.plot(df_yr["SOXL_close"], label = "SOXL_close", linewidth = .85)

# # plt.plot(x,y)
# # df_yr["diff"] = df_yr.iloc[:, 1] - df_yr.iloc[:, 3]
# df_yr = round(df_yr, 2)
# df_yr.to_csv("test2.csv")


# plt.legend()
# plt.savefig("new.png")
# plt.show()





start = "2010-06-09"
end = "2024-07-04"

spy = Summary(underlying="SOXX", triple="SOXL", start = start, end = end)
spy.summary_stats()
spy.init_portfolio()

print(spy.beta_norm_strategy(.001))
spy.plot_portfolio()
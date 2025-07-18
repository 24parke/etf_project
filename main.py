# import backtrader as bt
import yfinance as yf
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
class Summary:
    def __init__(self, underlying, triple):
        self.UNDERLYING = underlying
        self.TRIPLE = triple
        self.STARTING_VALUE_UNDER = 3000
        self.STARTING_VALUE_TRIP = -1000
        self.TICKERS = [underlying, triple]

        self.data = yf.download(self.TICKERS, period="8y", group_by="ticker", auto_adjust=False)
        self.dat_under = self.data[underlying]['Close']
        self.dat_trip = self.data[triple]['Close']

        dates = [d.strftime('%Y-%m-%d') for d in self.data.index.date]
        self.dfr = pd.DataFrame(data = {f"{underlying}_Close": self.dat_under})

        self.shares_under = self.STARTING_VALUE_UNDER / self.dat_under[0]
        self.shares_trip = self.STARTING_VALUE_TRIP / self.dat_trip[0]
        

        self.dfr.index = dates
        self.dfr[f"{underlying}_per_change"] = 1
        self.dfr[f"{underlying}_cumulative"] = 1
        self.dfr[f"{triple}_Close"] = self.dat_trip
        self.dfr[f"{triple}_per_change"] = 1
        self.dfr[f"{triple}_cumulative"] = 1
        self.dfr[f"ideal_per_change"] = 1
        self.dfr[f"ideal_cumulative"] = 1
        self.dfr["shares_under"] = 0
        self.dfr["shares_trip"] = 0


    def summary_stats(self):
        for i in range(1, len(self.dat_under)):
            self.dfr[f"{self.UNDERLYING}_per_change"][i] = self.dat_under[i] / self.dat_under[i-1] - 1
            self.dfr[f"{self.TRIPLE}_per_change"][i] = self.dat_trip[i] / self.dat_trip[i-1] - 1
            self.dfr["ideal_per_change"][i] = (self.dat_under[i] / self.dat_under[i-1] - 1) *3

            self.dfr[f"{self.UNDERLYING}_cumulative"][i] = self.dfr[f"{self.UNDERLYING}_cumulative"][i-1]*(1 + self.dfr[f"{self.UNDERLYING}_per_change"][i])
            self.dfr[f"{self.TRIPLE}_cumulative"][i] = self.dfr[f"{self.TRIPLE}_cumulative"][i-1]*(1 + self.dfr[f"{self.TRIPLE}_per_change"][i])
            self.dfr["ideal_cumulative"][i] = self.dfr["ideal_cumulative"][i-1]*(1 + self.dfr["ideal_per_change"][i])


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
        self.dfr[f"portfolio_{self.UNDERLYING}_long"]= 1
        self.dfr[f"portfolio_{self.TRIPLE}_short"]= 0
        self.dfr["portfolio_total"]=0
        self.dfr["beta_exposure"]=0


    def plot_portfolio(self):
        self.dfr = round(self.dfr, 2)
        self.dfr.to_csv('new.csv')
        plt.plot(self.dfr[f"portfolio_{self.UNDERLYING}_long"], label = f"portfolio_{self.UNDERLYING}_long", linewidth=0.85)
        plt.plot(self.dfr[f"portfolio_{self.TRIPLE}_short"], label = f"portfolio_{self.TRIPLE}_short", linewidth=0.85)
        plt.plot(self.dfr['portfolio_total'], label = "portfolio_total", linewidth=0.85)
        plt.plot(self.dfr['beta_exposure'], label = "beta_exposure", linewidth=0.85)
        plt.legend()
        plt.grid(True)
        plt.savefig("new.png")
        plt.show()


    def calc_portfolio(self, i):      
        self.dfr[f"portfolio_{self.UNDERLYING}_long"][i] = self.shares_under * self.dfr[f"{self.UNDERLYING}_Close"][i]
        self.dfr[f"portfolio_{self.TRIPLE}_short"][i] = self.shares_trip * self.dfr[f"{self.TRIPLE}_Close"][i]
        self.dfr["portfolio_total"][i] = self.dfr[f"portfolio_{self.UNDERLYING}_long"][i] + self.dfr[f"portfolio_{self.TRIPLE}_short"][i]
        self.dfr["beta_exposure"][i] = (self.shares_under * self.dfr[f"{self.UNDERLYING}_Close"][i]) + (self.shares_trip * self.dfr[f"{self.TRIPLE}_Close"][i] * 3)
        #num shares * share price * -3 + num shares * share price
        

    def beta_norm_strategy(self):
        for i in range(1, len(self.dat_under)):
            print(f"__________DAY {i}______________")

            self.calc_portfolio(i)
            if (self.dfr['beta_exposure'][i] > 150):
                beta = self.dfr['beta_exposure'][i]
                amt = beta/3/self.dfr[f"{self.TRIPLE}_Close"][i]
                self.shares_trip -= amt / 2
                print(f"selling {amt/2} shares of SPY.")
                # bamt = beta/3/self.dfr[f"{self.UNDERLYING}_Close"][i]
                self.shares_under -= amt/2
                print(f"selling {amt/2} shares of SPXL.")
                self.calc_portfolio(i)

            elif (self.dfr['beta_exposure'][i] < -150):
                beta = self.dfr['beta_exposure'][i]
                amt = -1*beta/self.dfr[f"{self.UNDERLYING}_Close"][i]
                self.shares_trip += amt/2
                print(f"buying {amt/2} shares of SPXL.")
                # bamt = -1*beta/3/self.dfr[f"{self.TRIPLE}_Close"][i]
                self.shares_under += amt/2
                print(f"buying {amt/2} shares of SPY.")
                self.calc_portfolio(i)
            else:
                print(f"beta exposure: {self.dfr['beta_exposure'][i]}")
            
            self.dfr["shares_under"][i] = self.shares_under
            self.dfr["shares_trip"][i] = self.shares_trip

            

spy = Summary(underlying="SPY", triple="SPXL")
spy.summary_stats()
spy.calc_returns()
spy.init_portfolio()
spy.beta_norm_strategy()
spy.plot_portfolio()
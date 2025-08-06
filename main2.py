import yfinance as yf
import pandas as pd
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
import random
from dateutil.relativedelta import relativedelta
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression



# --- 1) Modify Summary to accept start/end dates ---
class Summary:
    def __init__(self, underlying, triple, start, end):
        self.UNDERLYING = underlying
        self.TRIPLE = triple
        self.STARTING_VALUE_UNDER = 3000
        self.STARTING_VALUE_TRIP = -1000
        self.TICKERS = [underlying, triple]
        self.cash = 0

        self.data = yf.download(self.TICKERS,start=start, end = end, group_by="ticker", auto_adjust=False)
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
        self.dfr[f"shares_{underlying}"] = 0
        self.dfr[f"shares_{triple}"] = 0
        self.dfr["P/L"] = 0



    def summary_stats(self):
        for i in range(1, len(self.dat_under)):
            self.dfr[f"{self.UNDERLYING}_per_change"][i] = self.dat_under[i] / self.dat_under[i-1] - 1
            self.dfr[f"{self.TRIPLE}_per_change"][i] = self.dat_trip[i] / self.dat_trip[i-1] - 1
            self.dfr["ideal_per_change"][i] = (self.dat_under[i] / self.dat_under[i-1] - 1) *3

            self.dfr[f"{self.UNDERLYING}_cumulative"][i] = self.dfr[f"{self.UNDERLYING}_cumulative"][i-1]*(1 + self.dfr[f"{self.UNDERLYING}_per_change"][i])
            self.dfr[f"{self.TRIPLE}_cumulative"][i] = self.dfr[f"{self.TRIPLE}_cumulative"][i-1]*(1 + self.dfr[f"{self.TRIPLE}_per_change"][i])
            self.dfr["ideal_cumulative"][i] = self.dfr["ideal_cumulative"][i-1]*(1 + self.dfr["ideal_per_change"][i])

        self.dfr[f"{self.UNDERLYING}_per_change"] *= 100
        self.dfr[f"{self.TRIPLE}_per_change"] *= 100
        self.dfr["ideal_per_change"] *= 100
        self.dfr[f"{self.UNDERLYING}_cumulative"] *= 100
        self.dfr[f"{self.TRIPLE}_cumulative"] *= 100
        self.dfr["ideal_cumulative"] *= 100

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
    


# --- 2) Set up your ETF/triple list and interval lengths ---
pairs = [
    ("SPY","SPXL"),
    # ("UNG","BOYL"),
    ("SOXX", "SOXL"),
    ("IWM", "TNA"),
    ("QQQ","TQQQ"),
    ("DIA","UDOW"),
    # ("DUSL","XLI"),
    ("XBI","LABU"),
    ("XLF","FAS")
    # ("ERX","XLE"),

]
# interval_years = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
n_simulations = 35  # per pair
pos_gap = 0
pos_pl = 0
pgppl = 0

results = []

# --- 3) Simulation loop ---
for under, trip in pairs:
    # download once to get date bounds
    full = yf.download(
            [under, trip],
            period="max",
            group_by="ticker",
            auto_adjust=False
        )

    # 2) extract just the Close series for each ticker
    #    this gives you a DataFrame with columns ["SPY","SPXL"]
    closes = full.xs("Close", axis=1, level=1)

    # 3) for each column, find the first valid (non-NaN) index
    first_valid = closes.apply(lambda col: col.first_valid_index())

    # 4) the “youngest” inception is the later (max) of those two
    first_date = first_valid.max()
    last_date = full.index.max()

    for _ in range(n_simulations):
        # maximum = (last_date - first_date).years
        # years = random.randint(4, maximum)


        # # years = random.choice(interval_years)
        # # ensure there's room for that many years
        # max_offset_days = (last_date - first_date).days - int(years * 365)

        # if max_offset_days <= 0:

        #     continue  # skip if you can't fit a window of that size



        # # random start
        # start_offset = random.randint(0, max_offset_days)
        # start_date = first_date + pd.Timedelta(days=start_offset)
        # end_date   = start_date + pd.Timedelta(days=int(years*365))



        # 1) total span in days
        diff_days = (last_date - first_date).days

        # 2) maximum whole years you can fit
        max_years = diff_days // 365
        if max_years < 6:
            raise ValueError("Not enough history to sample a 4-year window")

        # 3) pick your random window length
        years = random.randint(6, max_years)

        # 4) now compute how many days that truly is, calendar-aware
        #    by adding N years to first_date
        #    then taking the difference in days
        candidate_end = first_date + relativedelta(years=years)
        delta_days    = (candidate_end - first_date).days

        # 5) ensure you still have room for the offset
        max_offset_days = diff_days - delta_days
        offset_days     = random.randint(0, max_offset_days)

        start_date = first_date + pd.Timedelta(days=offset_days)
        end_date   = start_date + relativedelta(years=years)

        # instantiate & run
        summ = Summary(under, trip,
                       start=start_date.strftime("%Y-%m-%d"),
                       end  =end_date.strftime("%Y-%m-%d"))
        summ.summary_stats()
        summ.init_portfolio()
        pl = summ.beta_norm_strategy(exposure_indicator=0.0001)
        # summ.plot_portfolio()

        # extract gap between final ideal vs. triple cumulative
        # final_ideal = summ.dfr.loc[:, 'ideal_cumulative']
        # final_trip = summ.dfr.loc[:, f"{trip}_cumulative"]
        final_ideal = summ.dfr["ideal_cumulative"].iloc[-1]
        final_trip  = summ.dfr[f"{trip}_cumulative"].iloc[-1]
        print(final_ideal)
        print(final_trip)
        gap = final_ideal - final_trip
        if (gap > 0 and pl > 0):
            pgppl += 1
        elif (pl > 0):
            pos_pl += 1
        elif (gap > 0):
            pos_gap += 1


        results.append({
            "underlying": under,
            "triple":     trip,
            "years":      years,
            "start":      start_date,
            "gap":        gap,
            "P/L":        pl
        })

# --- 4) Aggregate and compute correlation ---
df_res = pd.DataFrame(results)
df_res.plot.scatter(x="gap", y = "P/L")
plt.xlabel('Gap')
plt.ylabel('P/L')
plt.grid(True)
plt.show()
plt.savefig("n.png")
corr = df_res["gap"].corr(df_res["P/L"])
print(f"Correlation between gap and P/L over all sims: {corr:.3f}")


df_res.plot.scatter(x="years", y = "gap")
plt.xlabel('years')
plt.ylabel('gap')
plt.grid(True)
plt.show()
corr = df_res["years"].corr(df_res["gap"])
print(f"Correlation between time and gap over all sims: {corr:.3f}")


print(f"number of positive gaps: {pos_gap}")
print(f"number of positive p/ls: {pos_pl}")
print(f"number of positive gaps with positive p/ls: {pgppl}")

df_res.to_csv("n.csv")



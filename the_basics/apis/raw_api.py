from yahoofinancials import YahooFinancials
import seaborn as sns
import pandas as pd
from pandas.io.json import json_normalize

ticker = 'AAPL'
yahoo_financials = YahooFinancials(ticker)

yahoo_financials

all_statement_data_qt =  yahoo_financials.get_financial_stmts('quarterly', ['income', 'cash', 'balance'])

statements = json_normalize(all_statement_data_qt)

stat = statements.loc[0][0][0]

datos = pd.DataFrame.from_dict(stat)

datos.loc['totalCurrentAssets']

tickers = ['NKE','ADDYY','PUMSY']

ticker = 'NKE'

def comparable_roe(tickers):
    all_roes = []
    for t in tickers:
    
        yahoo_financials = YahooFinancials(t)
        b_s = yahoo_financials.get_financial_stmts('quarterly', 'balance')
        b_s = json_normalize(b_s)
        i_s = yahoo_financials.get_financial_stmts('quarterly', 'income')
        i_s = json_normalize(i_s)
        
        roes = []
        for i in range(4):
        
            balance = pd.DataFrame.from_dict(b_s.loc[0][0][i])
            income_s = pd.DataFrame.from_dict(i_s.loc[0][0][i])
        
            roe = income_s.loc['netIncome'] / balance.loc['totalStockholderEquity']
            roes.append(float(roe))
        
        dates = list(pd.DataFrame.from_dict(i_s.loc[0][0]).columns)
        frame = pd.DataFrame({'ticker':[t]*4,'roe':roes, 'date':dates})
        
        all_roes.append(frame)
    
    frame = pd.concat(all_roes)   
    frame['date']=pd.to_datetime(frame['date']) 
       
    ax = sns.lineplot(x="date", y="roe",
                      hue="ticker",
                      data=frame)

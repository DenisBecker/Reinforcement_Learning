from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import pandas as pd


# List to be created before running the environment 
def Make_Result_Lists():
    
    global Time_List 
    Time_List= [env.time]

    global Return_List
    Return_List = [state[0]]
    
    global Price_List
    Price_List = [1]
    
    global State_List
    State_List = [state]
    
    global Stock_Investment_List    
    Stock_Investment_List = [0]
    
    global Bank_Account_List
    Bank_Account_List = [InitialBank]
    
    global Action_List
    Action_List = []
    
    global Done_List 
    Done_List = [False]
    
    global Reward_List
    Reward_List = [None]

    
def Append_Result_Lists():    
    Return_List.append(state[0])
    Price_List.append(Price_List[-1]*(1+state[0]))
    Time_List.append(env.time)
    State_List.append(state)
    Action_List.append(action)
    Stock_Investment_List.append(env.StockValue_Out)
    Bank_Account_List.append(env.BankValue_Out)
    Done_List.append(done)
    Reward_List.append(reward)    

def Make_Result_df():

    Result_Dict = {'Time': Time_List,
                   'Return': Return_List,
                   'Price': Price_List,
                   'State': State_List,
                   'Action': Action_List,
                   'Stock Investment': Stock_Investment_List,
                   'Bank Account': Bank_Account_List,
                   'Done': Done_List,
                   'Reward': Reward_List} 

    global Result_df
    Result_df = pd.DataFrame(Result_Dict)



class InOut(Env):

# This environment is based on returns, i.e.:
# The first column in the data frame must be returns and not prices
# States are defined by means of all cells (columns) in a particular row (where rows 
# represent points in time)


    def __init__(self,
                 df,
                 initial_investment = 100,
                 interest_rate = 0.00,
                 transaction_cost_pct=0.00):
        
           
        # In the beginning we might just check whether the input data is correct
        assert df.ndim == 2, "Only 2D data frames are supported."
        assert df.empty == False, "The Data Frame seems empty."
        
        # Put the input data into instance-variables
        self.data = df.values  # This are the data in the whole dataframe
        self.ret = df.iloc[:, 0].values # This is the first column in the data frame
        
        self.initial_investment = initial_investment
        self.interest_rate = interest_rate
        self.transaction_cost_pct = transaction_cost_pct
        
        # In what follows we create the observation space
        # Here we use the minimum and maximimum values in the data frame
        # Here we increase the max by a little (e.g. 0.01 for returns and 1 for prices) such that a new bin does not start exactly on the max value
        # Otherwise the Q-Learning algorithm gives an error
               
        max_values = np.array(df.max())
        max_values = max_values + 0.01     #
        min_values = np.array(df.min())
        
        self.observation_space = Box(low=min_values, high=max_values, dtype=np.float32)
        
        # Action Space
        
        self.action_space = Discrete(2)  # in (long in asset) and out (money on bank account)        
        
        self.time = None

       
    def reset(self):
        
        self.time = 0
        self.StockValue_Out = 0
        self.BankValue_Out = self.initial_investment
        self.reward = 0
        self.Wealth_Out = self.BankValue_Out + self.StockValue_Out
        
        self.state = self.data[self.time,:]
        self.state = np.float32(self.state)
       
        return self.state
    
    
    def render(self):
        # Implement visualization if needed
        pass   
    
    def step(self, action):
        # The following actions apply
        # 0 means sell or stay out
        # 1 means buy or hold
        
        
        self.BankValue_In = self.BankValue_Out
        self.StockValue_In = self.StockValue_Out
        self.Wealth_In = self.Wealth_Out
        
        self.StockReturn = self.data[self.time + 1 ,0]  # The stock return achieved in point t+1 
                                                        # when buying the stock at point t
  
          
        if action == 1: # Long in Asset
            if self.StockValue_In == 0:  # Meaning that all money was on bank account before decision
                self.StockValue_Out = self.BankValue_In / (1 + self.transaction_cost_pct) * (1 + self.StockReturn)
                self.BankValue_Out = 0
        
            else:
                self.StockValue_Out = self.StockValue_In * (1 + self.StockReturn)
                self.BankValue_Out = 0
                
        elif action == 0: # Money in Bank Account
            if self.StockValue_In == 0:
                self.BankValue_Out = self.BankValue_In * (1 + self.interest_rate)
                self.Stock_Value = 0
                           
            else:
                self.BankValue_Out = self.StockValue_In * (1 - self.transaction_cost_pct) * (1 + self.interest_rate)
                self.StockValue_Out = 0
                

        self.Wealth_Out = self.BankValue_Out + self.StockValue_Out
        self.time = self.time + 1
        
        self.state = self.data[self.time,:]
        self.state = np.float32(self.state)
        
        self.reward = float(self.Wealth_Out - self.Wealth_In)
        
              
        if self.time == len(self.ret) - 1:
            done = True
        else:
            done = False
        
        
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, self.reward, done, info

    
    
class ContinuousTrading(Env):

# This environment is based on returns, i.e.:
# The first column in the data frame must be returns and not prices
# States are defined by means of all cells (columns) in a particular row (where rows 
# represent points in time)


    def __init__(self,
                 df,
                 initial_investment = 100,
                 interest_rate = 0.00,
                 transaction_cost_pct=0.00):
        
           
        # In the beginning we might just check whether the input data is correct
        assert df.ndim == 2, "Only 2D data frames are supported."
        assert df.empty == False, "The Data Frame seems empty."
        
        # Put the input data into instance-variables
        self.data = df.values  # This are the data in the whole dataframe
        self.ret = df.iloc[:, 0].values # This is the first column in the data frame
        
        self.initial_investment = initial_investment
        self.interest_rate = interest_rate
        self.transaction_cost_pct = transaction_cost_pct
        
        # In what follows we create the observation space
        # Here we use the minimum and maximimum values in the data frame
        # Here we increase the max by a little (e.g. 0.01 for returns and 1 for prices) such that a new bin does not start exactly on the max value
        # Otherwise the Q-Learning algorithm gives an error
               
        max_values = np.array(df.max())
        max_values = max_values + 0.01     #
        min_values = np.array(df.min())
     
        
        self.observation_space = Box(low=min_values, high=max_values, dtype=np.float32)
        
        # Action Space
        
        self.action_space = Box(low = np.array([0]), high = np.array([1]), dtype=np.float32) # How much of total asset weal
        
        self.time = None
        
        
    def reset(self):
        
        self.time = 0
        self.StockValue_Out = 0
        self.BankValue_Out = self.initial_investment
        self.reward = 0
        self.Wealth_Out = self.BankValue_Out + self.StockValue_Out
        
        self.state = self.data[self.time,:]
        self.state = np.float32(self.state)
       
        return self.state
    
    
    def render(self):
        # Implement visualization if needed
        pass   
    
    
    def step(self, action):
        # The following actions apply
        # 0 means sell or stay out
        # 1 means buy or hold
        
        
        self.BankValue_In = self.BankValue_Out
        self.StockValue_In = self.StockValue_Out
        self.Wealth_In = self.Wealth_Out
        
        self.StockReturn = self.data[self.time + 1 ,0]  # The stock return achieved in point t+1 
                                                        # when buying the stock at point t
            
            
        if action == 0:
            # We sell all crypto currency
            self.delta_Stock = -self.StockValue_In 
        else:    
            # The change in stock investment (which can be positive or negative):
            self.delta_Stock = self.BankValue_In - (( 1/action - 1) * self.StockValue_In) / (1/action + self.transaction_cost_pct)
                    
        
        self.StockValue_Out = (self.StockValue_In + self.delta_Stock) * (1 + self.StockReturn)
        self.BankValue_Out = (self.BankValue_In - self.delta_Stock * (1 + self.transaction_cost_pct)) * (1 + self.interest_rate)
        
        self.Wealth_Out = self.BankValue_Out + self.StockValue_Out
        self.time = self.time + 1
        
        self.state = self.data[self.time,:]
        self.state = np.float32(self.state)
        
        self.reward = self.Wealth_Out - self.Wealth_In
        self.reward = float(self.reward)
              
        if self.time == len(self.ret) - 1:
            done = True
        else:
            done = False
        
        
        # Set placeholder for info
        info = {}
        
        # Return step information
        return self.state, self.reward, done, info           

   
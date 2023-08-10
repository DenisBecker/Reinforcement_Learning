import matplotlib.pyplot as plt


def Plot_Results(Result_df):
    
    # I use None because Zero-Values would be plotted

    Result_df['In_Return'] = None
    Result_df['Out_Return'] = None
    Result_df['In_Price'] = None
    Result_df['Out_Price'] = None
    
    # Now we add columns where buy/in or sell/out decsions are registered in the dataframe
    # They need to be plotted either on the returns or prices
    # In order to use this for continuous decisions we define in as >=0.5 and out as <0.5
    # Originally it was ==1 or ==0
        
    Result_df.loc[Result_df['Action'] < 0.5, 'Out_Return'] = 1 * Result_df['Return']
    Result_df.loc[Result_df['Action'] >= 0.5, 'In_Return'] = 1 * Result_df['Return']
    Result_df.loc[Result_df['Action'] < 0.5, 'Out_Price'] = 1 * Result_df['Price']
    Result_df.loc[Result_df['Action'] >= 0.5, 'In_Price'] = 1 * Result_df['Price']

    
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 8))
    fig.suptitle('Trading Decisions')
    
    axs[0].plot(Result_df.Time, Result_df.Return, label = 'Time Series', linestyle = '--', linewidth=0.5)

    axs[0].scatter(Result_df.Time, Result_df.Out_Return, marker = '.',
                   label = 'Out (Money in Bank)', color = 'red')
    axs[0].scatter(Result_df.Time, Result_df.In_Return, marker = '.',
                   label = 'In (Long in Asset)', color = 'blue')
    

    
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Return')
    axs[0].axhline(y=0, color='black', linestyle='--')

    
    
    axs[1].plot(Result_df.Time, Result_df.Price, 
                label = 'Time Series', linestyle = '--', linewidth=0.5)

    axs[1].scatter(Result_df.Time, Result_df.Out_Price, marker = '.', 
                   label = 'Out (Money in Bank)', color = 'red')
    
    axs[1].scatter(Result_df.Time, Result_df.In_Price, marker = '.', 
                   label = 'In (Long in Asset)', color = 'blue')

    
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Price (Normalized)')
    axs[1].legend(facecolor='white',  # background color
                  framealpha=1,       # transperancy
                  loc = 'center left',
                  bbox_to_anchor = (1, 0.5),
                  fancybox = True,
                  shadow = True) # location of legend in plot
    
    axs[2].plot(Result_df.Time, Result_df['Total Reward'], 
                label = 'Total Reward', linestyle = '--', linewidth=0.5)
    axs[2].plot(Result_df.Time, Result_df['Buy Hold Total Reward'], 
                label = 'Total Reward Buy Hold', linestyle = '--', linewidth=0.5)
    axs[2].legend(facecolor='white',  # background color
                  framealpha=1,       # transperancy
                  loc = 'center left',
                  bbox_to_anchor = (1, 0.5),
                  fancybox = True,
                  shadow = True) # location of legend in plot

    

    
    plt.show()
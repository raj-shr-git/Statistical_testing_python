# Importing Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import stemgraphic as stem
from statsmodels import api as sm_api

class pre_anova_vis:
    """
    Description: This class consists of different functions for plotting the distribution of data before running the ANOVA.
    """
    def __init__(self,data,cols,dot_scale=0.1):
        """
        Description: This function is created for initializing the variables and it will be called every time when a class object is instantiated.  
        
        Input: It accepts below input parameters:
            1. ``data`` : (Pandas DataFrame)
                        It is the dataframe having the samples or populations observations
            2. ``cols`` : (Python List)
                        List of Groups or Treatments or columns for which distributions to be plotted
            3. ``dot_scale`` : Int or Float
                        Value of scale used in Dot plot. By default scale is 0.1
        
        Child-Functions: 
            ``plot_dist`` : This function is created for plotting the histogram of every treatment or column.
            ``dot_plot``  : This function is created for plotting the DOT Plot of every treatment or column.
            ``qq_plot``   : This function is created for creating the qunatile-quantile plot of every treatment or column.
            ``plot_box``  : This function is created for plotting the box-whisker plot of every treatment or column.
            ``all_plots`` : This function is created for creating all the plots(histogram, qq/pp/prob plots, Dot and Box plots) in one go. 
        """
        self.data = data
        self.cols = cols
        self.dot_scale = dot_scale
        # User-defined labels styles
        self.label_font_style = {'size':17, 'color': 'green', 'family': 'calibri'}
        self.title_font_style = {'size':19, 'color': 'purple', 'family': 'calibri'}
        # Figure length and width
        self.figstyle = (5,5)
        return None
    
    def plot_dist(self):
        """
        Description: This function is created for plotting the histogram of every treatment or column.
        Return: Plot the histogram.
        """
        # Below is the data distribution plot code
        print("\n")
        for col in self.cols:
            with plt.style.context('seaborn'):
                self.data[col].plot(kind='hist',histtype='bar',density=True,color='coral',figsize=self.figstyle)
                self.data[col].plot.density(color='black')
                plt.grid('ggplot2')
                plt.xlabel(col,fontdict=self.label_font_style)
                plt.ylabel('Freq',fontdict=self.label_font_style)
                plt.title('Data Distribution of {}'.format(col),fontdict=self.title_font_style)
            plt.show()
        return None
    
    def dot_plot(self):
        """
        Description: This function is created for plotting the DOT Plot of every treatment or column.
        Return: Plot the DOT Plot.
        """
        # Below is the Dot plot code
        for col in self.cols:
            print('\n###### Dot Plot of {} ######'.format(col))
            stem.stem_dot(self.data[col],flip_axes=True,asc=True,scale=self.dot_scale)
        return None
    
    def qq_plot(self):
        """
        Description: This function is created for creating the qunatile-quantile plot of every treatment or column.
        Return: Plot the qq-plot.
        """
        print("\n")
        with plt.style.context('classic'):
            for col in self.cols:
                prob_plt = sm_api.ProbPlot(self.data[~self.data[col].isna()][col])
                # Below is the quantile-quantile plot code
                prob_plt.qqplot(line='r')
                plt.xlabel(col,fontdict=self.label_font_style)
                plt.ylabel('Quantiles',fontdict=self.label_font_style)
                plt.title('QQ Plot of {}'.format(col),fontdict=self.title_font_style)
                plt.show()
                # Below is the percentile-percentile plot code    
                prob_plt.ppplot(line='r')
                plt.xlabel(col,fontdict=self.label_font_style)
                plt.ylabel('Probabilities',fontdict=self.label_font_style)
                plt.title('PP Plot of {}'.format(col),fontdict=self.title_font_style)
                plt.show()
                # Below is the Probability plot code    
                prob_plt.probplot(line='r')
                plt.xlabel(col,fontdict=self.label_font_style)
                plt.ylabel('Quantiles',fontdict=self.label_font_style)
                plt.title('Probability Plot of {}'.format(col),fontdict=self.title_font_style)
                plt.xticks(rotation=75)
            plt.show()
        return None
            
    def plot_box(self):
        """
        Description: This function is created for plotting the box-whisker plot of every treatment or column.
        Return: Plot the box-whisker plot.
        """
        # Below is the box-plot creation code
        print("\n")
        with plt.style.context('seaborn'):
            for col in self.cols:
                self.data[col].plot(kind='box',style='inferno',figsize=self.figstyle,label='')
                plt.xlabel(col,fontdict=self.label_font_style)
                plt.ylabel('Freq',fontdict=self.label_font_style)
                plt.title('Box-Plot of {}'.format(col),fontdict=self.title_font_style)
                plt.show()
        return None
            
    def all_plots(self):
        """
        Description: This function is created for creating all the plots(histogram, qq/pp/prob plots, Dot and Box plots) in one go. 
        """
        self.plot_dist()
        self.dot_plot()
        self.qq_plot()
        self.plot_box()
        return None
        
def marginal_row_mean_plot(df,grand_mean,row1=False,row2=False,row3=False,row4=False,row5=False,row6=False):
    """
    Description: This function is created for plotting the marginal mean graph of a dataset having at most 6 groups or columns.
    
    Input parameter:
        1. df : DataFrame having treatment or group data
        2. grand_mean : Overall mean of groups or str
        3. row1 : Row or Block 1 or str
        4. row2 : Row or Block 2 or str
        5. row3 : Row or Block 3 or str
        6. row4 : Row or Block 4 or str
        7. row5 : Row or Block 5 or str
        8. row6 : Row or Block 6 or str
    
    Output: Generate the Marginal Mean Graph
    
    Work-in-progress :: These two marginal mean plot functions to be combined in one as a generic function.
    """
    plt.figure(figsize=(10,7))
    with plt.style.context("classic"):
        plt.axhline(grand_mean,linestyle='--',color='black',label='Grand Mean')
        if row1 != False:
            plt.plot(df[row1][0],marker='>',ls='',ms=12,color='pink',label='Row/Block 1 mean')
        if row2 != False:
            plt.plot(df[row2][1],marker='>',ls='',ms=12,color='gray',label='Row/Block 2 mean')
        if row3 != False:
            plt.plot(df[row3][2],marker='>',ls='',ms=12,color='yellow',label='Row/Block 3 mean')
        if row4 != False:
            plt.plot(df[row4][3],marker='>',ls='',ms=12,color='skyblue',label='Row/Block 4 mean')
        if row5 != False:
            plt.plot(df[row5][4],marker='>',ls='',ms=12,color='lightgray',label='Row/Block 5 mean')
        if row6 != False:
            plt.plot(df[row6][5],marker='>',ls='',ms=12,color='orange',label='Row/Block 6 mean')
        plt.xticks(rotation=25)
        plt.title('Marginal Mean Graph of Blocks or Rows',fontdict={'size':20, 'family':'calibri', 'color':'coral', 'style': 'italic'})
    plt.legend()
    return None

def marginal_mean_plot(df,grand_mean,grp1,grp2,grp3=False,grp4=False,grp5=False,grp6=False,
                       row_graph_flg=False,row1=False,row2=False,row3=False,row4=False,row5=False,row6=False):
    """
    Description: This function is created for plotting the marginal mean graph of a dataset having at most 6 groups/columns and blocks/rows.
    
    Input parameter:
        1. df : DataFrame having treatment or group data
        2. grand_mean : Overall mean of groups or str
        3. grp1 : Column or Treatment 1 or str
        4. grp2 : Column or Treatment 2 or str
        5. grp3 : Column or Treatment 3 or str
        6. grp4 : Column or Treatment 4 or str
        7. grp5 : Column or Treatment 5 or str
        8. grp6 : Column or Treatment 6 or str
    
    Output: Generate the Marginal Mean Graphs
    """
    plt.figure(figsize=(10,7))
    with plt.style.context("classic"):
        plt.axhline(grand_mean,linestyle='--',color='black',label='Grand Mean')
        plt.plot(np.mean(df[grp1]),marker='*',ls='',ms=12,color='red',label='Grp({}) mean'.format(grp1))
        plt.plot(np.mean(df[grp2]),marker='*',ls='',ms=12,color='green',label='Grp({}) mean'.format(grp2))
        if grp3 != False:
            plt.plot(np.mean(df[grp3]),marker='*',ls='',ms=12,color='blue',label='Grp({}) mean'.format(grp3))
        if grp4 != False:
            plt.plot(np.mean(df[grp4]),marker='*',ls='',ms=12,color='brown',label='Grp({}) mean'.format(grp4))
        if grp5 != False:
            plt.plot(np.mean(df[grp5]),marker='*',ls='',ms=12,color='purple',label='Grp({}) mean'.format(grp5))
        if grp6 != False:
            plt.plot(np.mean(df[grp6]),marker='*',ls='',ms=12,color='coral',label='Grp({}) mean'.format(grp6))
        plt.xticks(rotation=25)
        plt.title('Marginal Mean Graph',fontdict={'size':22, 'family':'calibri', 'color':'coral', 'style': 'italic'})
    plt.legend()
    
    if row_graph_flg!=False:
        marginal_row_mean_plot(df=df,grand_mean=grand_mean,row1=row1,row2=row2,row3=row3,row4=row4,row5=row5,row6=row6)
    return None
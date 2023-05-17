
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import numpy as np


def reading_data(filedata):
    """
        Reads a CSV file and extracts the required data.

        Parameters:
            file_path (str): Path to the CSV file.

        Returns:
            tuple: A tuple containing two pandas DataFrames - dataframe_name and countries.
    """

    # read the CSV file and skip the first 4 rows of metadata
    dataframe = pd.read_csv(filedata, skiprows=4)
    countries = dataframe.drop(
        columns=['Country Code', 'Indicator Code', 'Unnamed: 66'], inplace=True)
    countries = dataframe.set_index('Country Name').T
    dataframe_name = dataframe.set_index('Country Name').reset_index()
    return dataframe_name, countries


def choosing_attribute(indicators, details):
    '''
    function for choosing an choosing_attribute
    '''
    details = details[details['Indicator Name'].isin([indicators])]
    return details


def choose_country(countries, details):
    """
    Extracts data for a specific country from a given DataFrame.

    Parameters:
        country (str): The name of the country to extract data for.
        dataframe (pd.DataFrame): The DataFrame containing the data.

    Returns:
        pd.DataFrame: A new DataFrame containing the extracted data for the specified country.
    """
    details = details[details['Country Name'].isin([countries])]
    details = details.set_index("Indicator Name")
    details = details.drop("Country Name", axis=1)
    # Transposing the dataframe
    details = details.T
    return details

def bar_chart(countries, indicator):
    """
    Plot the specified indicator for the given list of countries as a stacked bar chart.
    
    Select the indicator and data of your choice
    group the data according to the country and the respective years
    """
    selected_data = dataframe_name.loc[(dataframe_name['Country Name'].isin(countries)) &
                                (dataframe_name['Indicator Name'] == indicator) &
                                (dataframe_name[['1970', '1980', '1990', '2000', '2010', '2020']].notnull().all(axis=1)), :]

    # Grouping the selected data
    selected_data_grouped = selected_data.groupby(
        'Country Name')[['1970', '1980', '1990', '2000', '2010', '2020']].agg(list)

    fig, ax = plt.subplots()

    # Initialize variables for plotting
    width = 0.5
    colors = ['#3288bd', '#66c2a5', '#abdda4', '#e6f598', '#fee08b', '#fdae61']
    years = ['1970', '1980', '1990', '2000', '2010', '2020']
    x_coords = np.arange(len(countries))
    bottom = np.zeros(len(countries))

    # Plot the stacked bar chart
    for i, year in enumerate(years):
        data = selected_data_grouped[year].apply(lambda x: x[0])
        ax.bar(x_coords, data, width, label=year,
               color=colors[i], alpha=1, edgecolor='black', bottom=bottom)
        bottom += data

    ax.set_xticks(x_coords)
    ax.set_xticklabels(countries, rotation=90)
    ax.set_ylabel(indicator)
    ax.set_title(indicator, fontsize="10")
    ax.legend(fontsize="7", loc="upper right")

    # Calculate the skewness of the selected data
    skew = selected_data_grouped.skew(axis=0, skipna=True)
    skew.head()

    # Display the plot
    plt.show()

# Function to plot a multi-line graph
def line_chart(indicator, countries, df):
    years = [str(year) for year in range(1990, 2020, 5)]
    data = df.loc[df['Country Name'].isin(countries) & df['Indicator Name'].isin([indicator]), years].T
    kurt = data.kurtosis(axis=0, skipna=True)
    print(kurt.head())

    # Specify the colors using the 'spectral' colormap
    colors = ['#3288bd', '#66c2a5', '#abdda4', '#e6f598', '#fee08b', '#fdae61']

    # Plot the lines with the specified colors
    for i in range(len(countries)):
        plt.plot(data.index, data.iloc[:, i], linestyle='-', color=colors[i], linewidth=2.5)

    plt.legend(countries, fontsize="7", loc="upper right")
    plt.xlabel('Year')
    plt.ylabel(indicator)
    plt.show()


    
#function to plot a heat map
def heatmap(country, data, cols, cmap='viridis'):  
    subset = choose_country(country, data)
    columns = subset[cols]
    corr = columns.corr()
    sb.heatmap(corr, annot=True, cmap=cmap)
    plt.title(f"Correlation matrix for {country}")
    plt.show()


        
#reading the data 
dataframe_name, countries = reading_data(r"D:\python core\wbdata.csv")

#describe()
print(dataframe_name.describe())
print(countries.describe())

#seleted countries for bar graph
countries1 = ['Germany', 'Italy', 'United Kingdom',
              'France', 'Ireland', 'Austria']
indicator1 = 'Population growth (annual %)'
indicator2 = 'Arable land (% of land area)'
bar_chart(countries1, indicator1)
bar_chart(countries1, indicator2)

#plotting a line graph
line_chart('Forest area (% of land area)', [
               'Germany', 'Italy', 'United Kingdom', 'France', 'Ireland', 'Austria'], dataframe_name)

line_chart('Total greenhouse gas emissions (% change from 1990)', [
               'Germany', 'Italy', 'United Kingdom', 'France', 'Ireland', 'Austria'], dataframe_name)

#plotting a heat map    
heatmap('France', dataframe_name, ['Electricity production from oil sources (% of total)',
                                    'Electricity production from natural gas sources (% of total)',
                                    'Electricity production from hydroelectric sources (% of total)',
                                    'Electricity production from coal sources (% of total)'], cmap='Dark2')

heatmap('Germany', dataframe_name, ['Urban population (% of total population)',
                                 'Agriculture, forestry, and fishing, value added (% of GDP)',
                                 'Total greenhouse gas emissions (% change from 1990)',
                                 'Agricultural land (% of land area)',
                                 'Population growth (annual %)'], cmap='Spectral')

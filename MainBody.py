import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import statsmodels.api as sm


variableDictionary = {

    'opc_pm25':'PM2.5 from Optical Particle Counter (ug/m3)',
    'opc_pm10':'PM10 from Optical Particle Counter (ug/m3)',
    'pm25_env':'PM2.5 from Nephelometer (ug/m3)',
    'pm10_env':'PM10 from Nephelometer (ug/m3)',
    'timestamp_iso':'Time (UTC)'
    }
labelDictionary ={}

class quantAirTools:

    dataLibrary = {}
    deviceLibrary = {}
    dataJoin = pd.DataFrame()
    resampledLibrary = {}
   
 
    #Returns a cleaned and joined csv datasets
    def QAcleanToCSV(outputfile,startTime,endTime,*datasets):
 
        #Changing start and end times to datetime format
        startTime = pd.to_datetime(startTime, utc=True)
        endTime = pd.to_datetime(endTime, utc=True)
        #list to store data from each device
        dataList = []
        resampledList = []


        for dataset in datasets:
            #reading data from csv, pulling device information from top of file
            device = pd.read_csv(dataset,delimiter=',',engine='python',on_bad_lines='skip')
            # Skip first 3 rows (deviceModel, deviceID, deviceSN), then row 0 becomes the header
            data = pd.read_csv(dataset,delimiter=',',engine='python',skiprows=[0,1,2],header=0)
            #removes columns not being used
            data = data[['timestamp_iso','opc_pm25','pm25_env','opc_pm10','pm10_env','flag']]
            #updating timestamps to datetime to allow potting over time
            data.timestamp_iso = pd.to_datetime(data.timestamp_iso,yearfirst=True)
             #Adding boolean mask to filter data to only include sampling period
            timeMask = (data.timestamp_iso >= startTime) & (data.timestamp_iso <= endTime)
            data = data[timeMask]
            data = data.sort_values(by='timestamp_iso')
            dataResampled = data.resample('1min',on='timestamp_iso').mean()
            #adding a suffix to indicate the file from which each dataset comes from
            #adding suffix to columns for differentiation of source
            data = data.add_suffix(f'_{device.iat[1,1]}')
            dataResampled = dataResampled.add_suffix(f'_{device.iat[1,1]}')
            #adding data to data list
            dataList.append(data)
            resampledList.append(dataResampled)
            quantAirTools.dataLibrary[dataset] = [data, device]

        resampleJoin = pd.concat(resampledList, axis=1)
        dataJoin = pd.concat(dataList, axis=1)
        dataJoin.to_csv(outputfile)
        resampleJoin.to_csv(f'Resampled{outputfile}')
            
        return outputfile
    
    #adding plotting tool
    def QAPlotter(inputFile,startTime,endTime,var,dataset_labels=None):

        # Read the merged CSV file created by QAcleanToCSV
        # This file has columns like: timestamp_isoTalbotColocateO1.csv, opc_pm25TalbotColocateO1.csv, etc.
        df = pd.read_csv(inputFile)

        # Convert start and end times from strings to datetime objects for comparison
        # Input format: '2025-10-7 17:23:00'
        # Add UTC timezone to match the timezone-aware timestamps in the CSV
        startTime = pd.to_datetime(startTime, utc=True)
        endTime = pd.to_datetime(endTime, utc=True)

        # Get the first timestamp column to use as x-axis
        # After merging, there are multiple timestamp columns (one per device)
        # All should have the same values, so we just use the first one
        timestamp_col = df.filter(regex='timestamp').iloc[:, 0]

        # Convert timestamp column to datetime format so we can filter by time range
        # Handles ISO format like: '2025-10-07T17:23:46Z' or '2025-10-07 17:23:46+00:00'
        timestamp_col = pd.to_datetime(timestamp_col)

        # Create a boolean mask: True where timestamp is within our range, False otherwise
        # This gives us an array of True/False values, one for each row
        mask = (timestamp_col >= startTime) & (timestamp_col <= endTime)

        # Apply the mask to get only the timestamps within our time range
        xdata = timestamp_col[mask]

        # Filter columns that contain the variable name
        # e.g., var='pm25' will match: opc_pm25TalbotColocateO1.csv, pm25_envTalbotColocateO1.csv, etc.
        # This captures data from all devices for the specified variable
        ydata = df.filter(regex=var)

        # Apply the same time mask to get only the sensor data within our time range
        ydata_filtered = ydata[mask]

        # Plot each matching column as a separate line
        # If var='pm25' and you have 4 devices, you'll get 8 lines:
        # 4 devices × 2 columns each (opc_pm25 and pm25_env)
        for column in ydata_filtered.columns:
            
            parts = column.split('_MOD')
            device_name = parts[1]
            # Create the label based on whether custom labels were provided
            if dataset_labels=='custom':
                device_name = f'MOD{device_name}'
                label = labelDictionary[device_name]
                clean_label = label
            else:
                # Auto-generate label from column name
                # Extract the sensor type and dataset filename from the column name
                # Column format: sensor_typedataset.csv (e.g., opc_pm25TalbotColocateO1.csv)
                
                clean_label = f'MOD{device_name}'

            plt.plot(xdata, ydata_filtered[column], label=clean_label)

            variable = variableDictionary[f'{var}']

        # Add labels to make the plot readable
        plt.xlabel('Time (UTC)')
        plt.ylabel(f'{variable}')
        plt.title(f'{variable} readings from {startTime.strftime("%Y-%m-%d %H:%M")} to {endTime.strftime("%Y-%m-%d %H:%M")}')
        tickformatter = mp.dates.DateFormatter("%H:%M")
        # Add legend to identify which line is which device/sensor
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.gca().xaxis.set_major_formatter(tickformatter)
        # Rotate x-axis labels 45 degrees so datetime strings don't overlap
        plt.xticks(rotation=45)

        # Adjust layout so labels and legend don't get cut off
        plt.tight_layout()

        # Display the plot
        plt.show()

    def QASetLabels(sensor,location):       
        labelDictionary[f'{sensor}'] = f'{location}'
        return labelDictionary
    
    def QAAutocorrelation(dataset,var):

        dataset = pd.read_csv(dataset)
        dataset = dataset[f'{var}'].dropna()
        

        data = dataset.to_list()
        

        test = sm.tsa.stattools.adfuller(data)
        result = pd.Series(test[0:4], index=['ADF Test Statistic','p-value','Lags Used','Observations Used'])
        print(result)

        return(result)
    
    def QANormality(dataset,var):

         dataset = pd.read_csv(dataset)
         dataset = dataset[f'{var}'].dropna()
         data = dataset.to_list()

         test = sm.stats.diagnostic.kstest_normal(data, pvalmethod='table')

         result = pd.Series(test[0:2], index=('KS Test Statistic','p-value'))

         print(result)

         return(result)

    def QAMannWhitney(dataset,var1,var2):

        dataset = pd.read_csv(dataset)
        
        var1 = dataset[f'{var1}'].dropna()
        var2 = dataset[f'{var2}'].dropna()
        
        test = sm.stats.nonparametric.rank_compare_2indep(var1, var2, use_t = True)
        result = pd.Series(test[0:2])



quantAirTools.QAcleanToCSV('CandleTest.csv','2025-11-01 01:00:00','2025-11-01 04:00:00','CandleTest00378-1101.csv','CandleTest00384-1101.csv')
quantAirTools.QASetLabels('MOD-PM-00384','Location 1')
quantAirTools.QASetLabels('MOD-PM-00378','Location 2')
quantAirTools.QAPlotter('ResampledCandleTest.csv','2025-11-01 01:00:00','2025-11-01 04:00:00','opc_pm25')

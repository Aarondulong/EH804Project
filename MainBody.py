import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import statsmodels.api as sm

#creating dictionaries for labeling purposes
variableDictionary = {

    'opc_pm25':'PM2.5 from Optical Particle Counter (ug/m3)',
    'opc_pm10':'PM10 from Optical Particle Counter (ug/m3)',
    'timestamp_iso':'Time (UTC)'
    }
labelDictionary ={}
locationDictionary={}
treatmentDictionary={}

#Tools for analyzing data from QuantAir Modulair-PM sensors
class quantAirTools:

#Creating libraries for data joining purposes
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
            data = data[['timestamp_iso','sample_rh','sample_temp','opc_pm25','opc_pm10']]
            #updating timestamps to datetime to allow potting over time
            data.timestamp_iso = pd.to_datetime(data.timestamp_iso,yearfirst=True)
             #Adding boolean mask to filter data to only include sampling period
            timeMask = (data.timestamp_iso >= startTime) & (data.timestamp_iso <= endTime)
            data = data[timeMask]
            #sorting values chronologically to ensure timestamp numeric variable will line up
            data = data.sort_values(by='timestamp_iso')
            #Resamples data to 1 minute intervals to reduce noise for graphics
            dataResampled = data.set_index('timestamp_iso').resample('1min').mean()
            #resetting the index to allow access to 'timestamp-iso' variable
            dataResampled = dataResampled.reset_index()
            #adding timestamp index variable
            dataResampled['timeStamp'] = range(1, len(dataResampled)+1)
            #adding separate date and time columns
            dataResampled['Time'] = dataResampled['timestamp_iso'].dt.time
            dataResampled['Date'] = dataResampled['timestamp_iso'].dt.date
            
            #adding instrument ID from device info, split info to include number only
            instrumentID = device.iat[1,1]
            idParts = instrumentID.split('00')
            instrumentID = idParts[1]
            dataResampled['instrumentID'] = instrumentID

            #adding treatment if set via QASetTreatments
            dictKey = str(dataResampled.Date[1])
            if dictKey in treatmentDictionary.keys():
                dataResampled['treatment'] = treatmentDictionary[dictKey]
            else:
                dataResampled['treatment'] = 'Unknown'

            #adding Session if set in QASetLabels
            if dictKey in labelDictionary.keys():
                dataResampled['Session'] = labelDictionary[dictKey]
            else:
                dataResampled['Session'] = 'Unknown'
            
            #setting location labels based on instrument ID as outlines in QASetLocation
            dataResampled['proximityToXRoad'] = locationDictionary[f'{instrumentID}']

            #reorganizing columns for readability
            dataResampled = dataResampled[['timeStamp','instrumentID','Date','Time','treatment','Session','proximityToXRoad','sample_rh','sample_temp','opc_pm25','opc_pm10']]
            #renaming vars to fit codebook guidelines
            cleanAxis = ['Time_Stamp','Instrument_ID','Date','Time','Condition','Session','Prox_to_xroad','Relative_Humidity','Temperature','PM25','PM10']
            dataResampled = dataResampled.set_axis(cleanAxis,axis=1)
            dataList.append(data)
            resampledList.append(dataResampled)
        #joining datasets if more than one is passed (May be removed: see QAJoinCleaned)
        if len(dataList) > 1:
            #adding a suffix to indicate the file from which each dataset comes from
            data = data.add_suffix(f'_{device.iat[1,1]}')
            dataResampled = dataResampled.add_suffix(f'_{device.iat[1,1]}')
            #joining data to make a wideformat table
            resampleJoin = pd.concat(resampledList, axis=1)
            dataJoin = pd.concat(dataList, axis=1)
            dataJoin.to_csv(outputfile)
            resampleJoin.to_csv(f'Resampled{outputfile}')
        #returning cleaned data if only one is passed
        else:
            data.to_csv(outputfile)
            dataResampled.to_csv(f'Resampled{outputfile}',index=False)
        return outputfile
    
    #adding plotting tool (Being Reworked!)
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
    #used to set location based on sensor (3-digit ID)
    def QASetLocation(sensor,location):
        locationDictionary[f'{sensor}'] = f'{location}'
        return locationDictionary
    #used to set labels for which Session data is from pass date in YYYY-MM-DD format as key
    def QASetLabels(sensor,label):       
        labelDictionary[f'{sensor}'] = f'{label}'
        return labelDictionary
    #used to set treatment labels based on date
    def QASetTreatments(date,treatment):
        treatmentDictionary[f'{date}'] = f'{treatment}'
        return treatmentDictionary
    #used to check for autocorrelation (IN PROGRESS)
    def QAAutocorrelation(dataset,var):

        dataset = pd.read_csv(dataset)
        dataset = dataset[f'{var}'].dropna()
        

        data = dataset.to_list()
        

        test = sm.tsa.stattools.adfuller(data)
        result = pd.Series(test[0:4], index=['ADF Test Statistic','p-value','Lags Used','Observations Used'])
        print(result)

        return(result)
    #used to check normality (IN PROGRESS)
    def QANormality(dataset,var):

         dataset = pd.read_csv(dataset)
         dataset = dataset[f'{var}'].dropna()
         data = dataset.to_list()

         test = sm.stats.diagnostic.kstest_normal(data, pvalmethod='table')

         result = pd.Series(test[0:2], index=('KS Test Statistic','p-value'))

         print(result)

         return(result)
    #runs a Mann Whitney U test of selected variables (IN PROGRESS)
    def QAMannWhitney(dataset,var1,var2):

        dataset = pd.read_csv(dataset)
        
        var1 = dataset[f'{var1}'].dropna()
        var2 = dataset[f'{var2}'].dropna()
        
        test = sm.stats.nonparametric.rank_compare_2indep(var1, var2, use_t = True)
        result = pd.Series(test[0:2])
    #Joins cleaned data in long table format, returns csv of completed table
    def QAJoinCleaned(outputfile,*datasets):
    
        dataList=[]

        for dataset in datasets:
            data = pd.read_csv(dataset,delimiter=',',engine='python',header=0)
            dataList.append(data)
        output = pd.concat(dataList,axis=0)
        output.to_csv(f'{outputfile}',index=False)

#tools to clean and analyze purpleair data
class purpleAirTools:
    #cleans purpleair data
    def PAclean(outputfile,startTime,endTime,dataset):  
        #reading in data and filtering to timestamp and IAQ
        data = pd.read_csv(dataset,delimiter=',',engine='python')
        data = data[['UTCDateTime','gas']]
        #converting timestamp column to datetime variable
        data['UTCDateTime'] = pd.to_datetime(data['UTCDateTime'], utc=True, yearfirst=True)
        #splitting datetime variale into date and time
        data['Date'] = data['UTCDateTime'].dt.date            
        data['Time'] = data['UTCDateTime'].dt.time
        #using a mask to crop data to testing period
        timeMask = (data['UTCDateTime'] >= startTime) & (data['UTCDateTime'] <= endTime)
        data = data[timeMask]
        #sorting data chronologically to ensure timestamp index lines up properly
        data = data.sort_values(by='UTCDateTime')
        data['Time_Stamp'] = range(1, len(data) + 1)
        #adding instrument ID (only one used so single definition is ok)
        data['Instrument_ID'] = 'PurpleAir_Zen'
        #adding location (stayed in only one location)
        data['Prox_to_xroad'] = 'Away'
        #resetting index to ensure no columns were moved to index
        data = data.reset_index()
        #using dictionary keys to set treatment and Session labels
        dictKey = str((data.Date[1]))
        print(dictKey)
        if dictKey in treatmentDictionary.keys():
            data['treatment'] = treatmentDictionary[dictKey]
        else:
            data['treatment'] = 'Unknown'
        if dictKey in labelDictionary.keys():
            data['Session'] = labelDictionary[dictKey]
        else:
            data['Session'] = 'Unknown'
        #reorganizing columns for readability 
        data = data[['Time_Stamp','Instrument_ID','Prox_to_xroad','Date','Time','treatment','Session','gas']]
        #renaming columns to line up with codebook
        cleanedAxis = ['Time_Stamp','Instrument_ID','Prox_to_xroad','Date','Time','Condition','Session','IAQ']
        data = data.set_axis(cleanedAxis,axis=1)
        #removes whitespace from IAQ column
        data['IAQ'] = data['IAQ'].str.replace(' ','')
        #changes to float
        data['IAQ'] = pd.to_numeric(data['IAQ'], errors='coerce')
        #returns cleaned data
        data.to_csv(f'{outputfile}', index=False)

    #joins cleaned purpleair datesets in long table format
    def PAJoin(outputfile,*datasets):
        dataList = []
        for dataset in datasets:
            data = pd.read_csv(dataset)
            dataList.append(data)
        JoinedData = pd.concat(dataList, axis=0)
        JoinedData.to_csv(f'{outputfile}',index=False)

#setting location of each sensor for proximity to cross road variable
quantAirTools.QASetLocation('378', 'Near')
quantAirTools.QASetLocation('384','Away')

#setting the treatment based on the date
quantAirTools.QASetTreatments(date='2025-10-25',treatment='Control')
quantAirTools.QASetTreatments(date='2025-10-26',treatment='Intervention')
quantAirTools.QASetTreatments(date='2025-11-01',treatment='Control')
quantAirTools.QASetTreatments(date='2025-11-02',treatment='Intervention')

#setting which sampling period data originates from based on date
quantAirTools.QASetLabels('2025-10-25','Session 1')
quantAirTools.QASetLabels('2025-10-26','Session 1')
quantAirTools.QASetLabels('2025-11-01','Session 2')
quantAirTools.QASetLabels('2025-11-02','Session 2')


#cleaning modulair data
quantAirTools.QAcleanToCSV('NearControlSession1.csv','2025-10-25 19:00:00', '2025-10-25 20:59:59', 'MOD-PM-00378_10252025.csv')
quantAirTools.QAcleanToCSV('NearInterventionSession1.csv','2025-10-26 19:00:00', '2025-10-26 20:59:59', 'MOD-PM-00378_10262025.csv')
quantAirTools.QAcleanToCSV('AwayControlSession1.csv','2025-10-25 19:00:00', '2025-10-25 20:59:59', 'MOD-PM-00384_10252025.csv')
quantAirTools.QAcleanToCSV('AwayInterventionSession1.csv','2025-10-26 19:00:00', '2025-10-26 20:59:59', 'MOD-PM-00384_10262025.csv')
quantAirTools.QAcleanToCSV('NearControlSession2.csv','2025-11-01 19:00:00','2025-11-01 20:59:59','MOD-PM-00378_11012025.csv')
quantAirTools.QAcleanToCSV('NearInterventionSession2.csv','2025-11-02 20:00:00','2025-11-02 21:59:59','MOD-PM-00378_11022025.csv')
quantAirTools.QAcleanToCSV('AwayControlSession2.csv','2025-11-01 19:00:00','2025-11-01 20:59:59','MOD-PM-00384_11012025.csv')
quantAirTools.QAcleanToCSV('AwayInterventionSession2.csv','2025-11-02 20:00:00','2025-11-02 21:59:59','MOD-PM-00384_11022025.csv')

#joining modulair data
quantAirTools.QAJoinCleaned('ModulairMaster.csv',
                            'ResampledNearControlSession1.csv',
                            'ResampledNearInterventionSession1.csv',
                            'ResampledAwayControlSession1.csv',
                            'ResampledAwayInterventionSession1.csv',
                            'ResampledNearControlSession2.csv',
                            'ResampledNearInterventionSession2.csv',
                            'ResampledAwayControlSession2.csv',
                            'ResampledAwayInterventionSession2.csv')

#cleaning purpleair data
purpleAirTools.PAclean('ControlSession1.csv','2025-10-25 19:00:00', '2025-10-25 20:59:59','PA_20251025.csv')
purpleAirTools.PAclean('InterventionSession1.csv','2025-10-26 19:00:00', '2025-10-26 20:59:59','PA_20251026.csv')
purpleAirTools.PAclean('ControlSession2.csv','2025-11-01 19:00:00', '2025-11-01 20:59:59','PA_20251101.csv')
purpleAirTools.PAclean('InterventionSession2.csv','2025-11-02 20:00:00', '2025-11-02 21:59:59','PA_20251102.csv')

#joining cleaned purpleair data into master list
purpleAirTools.PAJoin('PurpleAirMaster.csv','ControlSession1.csv','InterventionSession1.csv','ControlSession2.csv','InterventionSession2.csv')

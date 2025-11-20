import pandas as pd
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

class QAPlotter:

    variableDictionary = {

    'PM25_ug/m3':'PM2.5 from Optical Particle Counter (ug/m3)',
    'PM10_ug/m3':'PM10 from Optical Particle Counter (ug/m3)',
    'Time_Stamp':'Time (UTC)'
    }

    #adding plotting tool (Being Reworked!)
    def QAPlotter(inputFile,startTimeStamp,endTimeStamp,var,prox_to_xroad,dataset_labels=None):

        # Read the merged CSV file created by QAcleanToCSV
        # This file has columns like: timestamp_isoTalbotColocateO1.csv, opc_pm25TalbotColocateO1.csv, etc.
        df = pd.read_csv(inputFile)

        mask = (df['Time_Stamp'] >= startTimeStamp) & (df['Time_Stamp'] <= startTimeStamp)
        df = df[mask]

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

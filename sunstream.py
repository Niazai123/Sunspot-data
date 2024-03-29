import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from scipy import signal
from scipy.signal import welch
import pywt
import plotly.graph_objects as go
from scipy.signal.windows import hann
mydata = pd.read_csv('sunspot_data.csv')
st.title('Sunspot Data Analysis')
st.write('This Streamlit app allows you to explore sunspot data.')
st.header('Sunspot Data set')
mydata['Date'] = pd.to_datetime(mydata[['Year','Month','Day']])
st.write(mydata)
st.sidebar.image('solar+flares.jpg', caption='Solar Flares, Sunspots, and the Solar Cycle', use_column_width=True)
st.sidebar.title("Sunspots Project Overview")
st.sidebar.write("""
The sunspots project aims to analyze and understand the periodic patterns and variations in sunspot activity over time. Sunspots, dark patches on the Sun's surface, are indicative of magnetic activity and play a crucial role in solar dynamics. By examining historical sunspot data, collected over centuries, we seek to uncover long-term cycles, trends, and irregularities in sunspot activity.

**Key Objectives:**
1. Periodic Pattern Identification
2. Trend Analysis
3. Anomaly Detection
4. Data Visualization

**Approach:**
1. Data Acquisition
2. Data Preprocessing
3. Analysis Techniques
4. Interpretation and Conclusion

**Expected Outcomes:**
The sunspots project aims to contribute to our understanding of solar activity by providing insights into the periodic behavior, trends, and anomalies in sunspot data. The analysis results will enhance our knowledge of solar physics, potentially leading to advancements in space weather forecasting and solar climate research. Additionally, the project will demonstrate the application of data analysis techniques in studying complex natural phenomena and encourage further exploration in solar science.
""")
st.title('Is there a noticeable trend in the number of sunspots over time?')
st.write("So, In the bar chart given below, it is shown that the maximum number of sunspot counts had been observed on Aug 26, 1870. And the next trend of sunspots observed in Dec 25,1957.")

if not pd.api.types.is_datetime64_any_dtype(mydata['Date']):
    mydata['Date'] = pd.to_datetime(mydata['Date'])

# Group data by date and calculate the sum of sunspots for each date
daily_sunspots = mydata.groupby('Date')['Number of Sunspots'].sum().reset_index()

# Plot the line chart
st.line_chart(daily_sunspots.set_index('Date'))
st.title('Are there any long-term cycles or periodicities in the data?')




sunspot_data = mydata

# Option for applying different analysis techniques
st.header("Applying Different Analysis Techniques")
##analysis_option = st.selectbox("Select analysis technique:", ["Descriptive Statistics", "Histogram", "Box Plot", "Fourier Transform", "Wavelet Transform", "Spectral Analysis"])
tabs = ["Descriptive Statistics", "Histogram", "Box Plot", "Fourier Transform", "Wavelet Transform", "Spectral Analysis"]

analysis_option = st.radio("Select analysis technique:", tabs)
if analysis_option == "Descriptive Statistics":
    # Descriptive statistics
    st.write("Descriptive Statistics:")
    st.write(sunspot_data.describe())
elif analysis_option == "Histogram":
    # Histogram
    st.write("Histogram:")
    explanation_histogram_text = """
**Explanation of Histogram Results:**

In the histogram displaying the number of sunspots, the x-axis represents the bins or intervals of the number of sunspots, while the y-axis represents the frequency or count of observations falling within each bin.

If the histogram is showing frequencies exceeding 20000 on the y-axis for the bin range of 0 - 25 number of sunspots, it indicates that there are a large number of observations in your data falling within this range. Specifically, more than 20000 data points or observations have been recorded where the number of sunspots falls within the interval of 0 to 25.

This high frequency count suggests that sunspot activity levels within this range are relatively common or prevalent in our dataset, contributing to the overall distribution of sunspot occurrences.
"""
    st.markdown(explanation_histogram_text)
    plt.figure(figsize=(10, 6))
    plt.hist(sunspot_data["Number of Sunspots"], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title("Distribution of Sunspot Counts")
    plt.xlabel("Number of Sunspots")
    plt.ylabel("Frequency")
    plt.grid(True)
    st.pyplot(plt)
elif analysis_option == "Box Plot":
    # Box plot
    st.write("Box Plot:")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=sunspot_data["Number of Sunspots"], color='skyblue')
    plt.title("Box Plot of Sunspot Counts")
    plt.xlabel("Number of Sunspots")
    plt.grid(True)
    st.pyplot(plt)
elif analysis_option == "Fourier Transform":
  explanation_fourier_text = """
**Explanation of 0 Hz Component in Fourier Transformation:**

In Fourier transformation, the 0 Hz component represents the DC (Direct Current) component of the signal. This component indicates the average value or the mean level of the signal over the entire time domain. In the context of sunspot data, the presence of a peak at 0 Hz suggests the existence of a constant or steady-state component in the sunspot activity.

The 0 Hz component provides essential information about the baseline or average level of the signal, which can be significant for understanding the overall behavior or trends in the data. However, it does not convey specific periodic patterns or variations present in the signal.

Therefore, while the 0 Hz component does not directly represent periodic oscillations or cycles in the sunspot activity, it serves as a fundamental component in Fourier transformation, aiding in the characterization of the signal's baseline or average behavior.
"""
  st.markdown(explanation_fourier_text)
  mydata_df = pd.DataFrame(mydata)

# Sort the DataFrame by date if it's not already sorted
  mydata_df = mydata_df.sort_values(by='Date')

# Ensure even spacing and fill in missing values if any
  date_range = pd.date_range(start=mydata_df['Date'].min(), end=mydata_df['Date'].max(), freq='D')
  mydata_df = mydata_df.set_index('Date').reindex(date_range).fillna(0).reset_index()

# Compute the FFT
  sunspot_fft = np.fft.fft(mydata_df['Number of Sunspots'])
  frequencies = np.fft.fftfreq(len(mydata_df))

# Create Plotly figure
  fig = go.Figure()

# Add trace for Fourier Transform
  fig.add_trace(go.Scatter(x=frequencies, y=np.abs(sunspot_fft),
                         mode='lines', name='Fourier Transform'))

# Update layout
  fig.update_layout(title="Fourier Transform of Sunspot Counts",
                  xaxis_title="Frequency",
                  yaxis_title="Amplitude",
                  template='plotly_white')

# Display the plot in Streamlit
  st.plotly_chart(fig)
elif analysis_option == "Wavelet Transform":
    # Perform Wavelet Transform and display results
    st.write("The Wavelet Transform allows for the analysis of localized changes and long-term trends in the sunspot data. The presence of a long-cycle periodic pattern, such as the 11-year solar cycle, is evident as a concentrated region of energy in the Wavelet spectrum. This periodic pattern arises from the cyclical variation of solar magnetic activity, which influences the formation and distribution of sunspots over extended periods.")
    sunspot_counts = sunspot_data["Number of Sunspots"].values
    
    # Example of Continuous Wavelet Transform (CWT) using Morlet wavelet
    widths = np.arange(1, 31)
    cwt_matrix = signal.cwt(sunspot_counts, signal.morlet, widths)
    
    # Plot CWT matrix
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(cwt_matrix), extent=[0, len(sunspot_counts), 1, 31], cmap='jet', aspect='auto')
    plt.colorbar(label='Magnitude')
    plt.title("Continuous Wavelet Transform of Sunspot Counts")
    plt.xlabel("Time")
    plt.ylabel("Scale")
    st.pyplot(plt)
else:
    # Perform Spectral Analysis and display results
    explanation_text = """
**Spectral Analysis Results:**

The spectral analysis reveals distinctive peaks in the frequency domain, providing insights into the periodic patterns present in the sunspot data. Notably, we observe two prominent peaks:

1. **Peak at 0 Hz:** This peak represents the presence of a significant low-frequency component in the data, suggesting long-term variations or trends in sunspot activity. Frequencies close to 0 Hz correspond to cycles that span over extended periods, potentially reflecting long-term solar activity patterns.

2. **Peak at 0.03 Hz:** The presence of a peak at approximately 0.03 Hz indicates the existence of a periodic component with a cycle duration of approximately 33 years. This periodicity suggests the occurrence of a recurring pattern in sunspot activity over this timescale.

Following these peaks, the graph shows a decline in power as the frequency increases, indicating a decrease in the strength of periodic patterns at higher frequencies. This decline reflects the diminishing influence of shorter-term fluctuations or noise in the sunspot data as we move towards higher frequencies.

These spectral analysis results provide valuable insights into the underlying periodic patterns and trends in sunspot activity, aiding in the understanding of solar dynamics and their potential impacts on Earth's climate and space weather.
"""
    st.markdown(explanation_text)

 


    sunspot_data = pd.DataFrame(sunspot_data)
    sunspot_data = sunspot_data.sort_values(by='Date')

# Compute the power spectral density using Welch's method
    freq, psd = welch(sunspot_data['Number of Sunspots'], nperseg=256)

# Create the Plotly figure
    fig = go.Figure()

# Add the power spectral density trace
    fig.add_trace(go.Scatter(x=freq, y=psd, mode='lines', name='Power Spectral Density'))

# Update layout
    fig.update_layout(title='Power Spectral Density of Sunspots (Welch\'s Method)',
                  xaxis_title='Frequency',
                  yaxis_title='Power/Frequency (dB/Hz)',
                  template='plotly_white')

# Display the plot in Streamlit
    st.plotly_chart(fig)


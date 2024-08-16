# Time Series Feature Extraction

This project is designed to extract key features from time series data. The extracted features provide insights into various structural changes, trends, and patterns within the data. The project focuses on detecting and analyzing specific characteristics such as plateaus, mean shifts, variance shifts, noise, trends, and seasonality.

## Features

The following features are extracted by the project:

1. **Plateau Detection**
   - **Function**: Identifies flat regions in the time series where the values remain relatively constant over a period.
   - **Use Case**: Helps in detecting periods of stability or inactivity in the time series.

2. **Mean Shift Detection**
   - **Function**: Detects abrupt changes in the mean level of the time series.
   - **Details**:
     - **Change Point**: The exact point where the shift occurs.
     - **Scale**: The magnitude of the change in mean.
   - **Use Case**: Useful for identifying sudden changes in the underlying process generating the time series.

3. **Variance Shift Detection**
   - **Function**: Identifies changes in the variability of the time series.
   - **Details**:
     - **Change Point**: The point in time where the variance changes.
     - **Scale**: The extent of the change in variance.
   - **Use Case**: Helps in recognizing periods of increased or decreased volatility in the data.

4. **Noise Strength and Variance**
   - **Function**: Measures the strength and variability of the noise present in the time series.
   - **Use Case**: Assists in distinguishing between the true signal and random fluctuations.

5. **Trend Order and Slope**
   - **Function**: Analyzes the underlying trend in the time series, determining both the order (linear, quadratic, etc.) and the slope.
   - **Use Case**: Provides insight into the long-term direction of the time series.

6. **Seasonality and Its Strength**
   - **Function**: Detects recurring patterns or cycles within the time series and measures the strength of these seasonal effects.
   - **Use Case**: Useful for forecasting and understanding periodic behaviors in the data.

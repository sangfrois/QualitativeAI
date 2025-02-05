import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file into a DataFrame
df = pd.read_csv('sentiment_analysis_results.csv')

# 1. Handle potential issues in `sentiment_result` early on
df['sentiment_result'] = df['sentiment_result'].astype(str).str.replace('[^0-9.]', '', regex=True)
df['sentiment_result'] = pd.to_numeric(df['sentiment_result'], errors='coerce')

# Remove rows with empty values in `sentiment_result` AFTER cleaning
df = df.dropna(subset=['sentiment_result'])

# Extract window number from `window_indices`
df['window_number'] = df['window_indices'].str.extract(r'(\d+)').astype(int)

# Drop the original `window_indices` column
df = df.drop(columns=['window_indices'])

# Rename the `sentiment_result` column to `alliance_score`
df = df.rename(columns={'sentiment_result': 'alliance_score'})

# Group by `transcript_file` and get the cumulative count for the x-axis
df['time_point'] = df.groupby('transcript_file').cumcount() + 1

# 2. Force conversion to integers, but provide a fallback
df['alliance_score'] = df['alliance_score'].astype(int, errors='ignore')
# If it can't convert cleanly, it will leave the value as is

# Create the line plot
sns.relplot(
    data=df,
    x="time_point",
    y="alliance_score",
    col="transcript_file",  # Separate subplots for each transcript file
    kind="line",
    marker="o"
)

# Set the plot title and axis labels
plt.suptitle("Therapeutic Alliance Over Time", y=1.02)
plt.xlabel("Time Point")
plt.ylabel("Therapeutic Alliance Score")

# Show the plot
plt.show()
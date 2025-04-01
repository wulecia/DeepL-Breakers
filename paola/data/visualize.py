import pandas as pd
import matplotlib.pyplot as plt


# Load the dataset (adjust the path if needed)
df = pd.read_parquet("data/measuring-hate-speech.parquet")

# Show basic info
#print(df.info())
#print(df.head())
#print(df.columns())
print(df.size)


'''

# Check label distribution (try 'hate_speech_score' or others)
if 'hate_speech_score' in df.columns:
    plt.hist(df['hate_speech_score'], bins=20)
    plt.title("Distribution of Hate Speech Scores")
    plt.xlabel("Score")
    plt.ylabel("Frequency")
    plt.show()
else:
    print("No 'hate_speech_score' column found — check column names:")
    print(df.columns)
'''
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_stats():
    filename = "dummy_stats.csv"
    
    if not os.path.exists(filename):
        print(f"File {filename} not found. Make sure you have fetched it from the orphan branch.")
        print("Run: git fetch origin test-stats-data && git checkout origin/test-stats-data -- dummy_stats.csv")
        return

    try:
        df = pd.read_csv(filename)
        print("Data loaded:")
        print(df)
        
        df.plot(x='date', y='value', kind='bar')
        plt.title("Test Strategy Results")
        plt.tight_layout()
        plt.show()
        print("Plot generated.")
    except Exception as e:
        print(f"Error plotting data: {e}")

if __name__ == "__main__":
    plot_stats()

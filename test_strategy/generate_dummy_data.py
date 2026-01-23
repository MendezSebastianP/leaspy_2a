import csv
import random
import datetime
import os
import sys

def main():
    filename = "dummy_stats.csv"
    
    # If a filename argument is provided, use that instead (useful for temporary files)
    if len(sys.argv) > 1:
        filename = sys.argv[1]

    header = ["date", "value"]
    file_exists = os.path.isfile(filename)

    # Generate data
    row = [datetime.datetime.now().isoformat(), random.randint(1, 100)]

    # Write/Append
    try:
        with open(filename, 'a', newline='') as f:
            writer = csv.writer(f)
            # Only write header if file didn't exist and we aren't appending to a temp file that is meant to be cat'ed later
            # For this test, let's keep it simple: if file doesn't exist, write header.
            if not file_exists and os.stat(filename).st_size == 0: 
                writer.writerow(header)
            writer.writerow(row)
        print(f"Successfully added row: {row}")
    except Exception as e:
        print(f"Error writing to file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

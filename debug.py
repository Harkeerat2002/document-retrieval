import csv
import collections

# Open the submission file
path = "data/saved_files/final_results.csv"

# Read the submission file
with open(path, "r") as file:
    reader = csv.reader(file)
    allId = [row[0] for row in reader]
    
    # print all the duplicated Ids
    print("Duplicated Ids:")
    print([item for item, count in collections.Counter(allId).items() if count > 1])
    
import json
import csv
import gzip
import pandas as pd
#Amazon
def amazon_data():
    input_file = './Tools_and_Home_Improvement_5.json.gz'  
    output_file = './Home.csv' 
    csv_data = []
    with gzip.open(input_file, 'r') as f:
        for line in f:
            
            entry = json.loads(line.strip())
            reviewerID = entry["reviewerID"]
            asin = entry["asin"]
            overall = int(entry["overall"])  
            unixReviewTime = entry["unixReviewTime"]
            csv_data.append([reviewerID, asin, overall, unixReviewTime])

   
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print(f"CSV ：{output_file}")

if __name__ == '__main__':
    amazon_data()
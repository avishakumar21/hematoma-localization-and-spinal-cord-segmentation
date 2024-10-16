import csv

# Replace 'your_file.csv' with the path to your CSV file
csv_file_path = 'runs/detect/train3/results.csv'

# Read the CSV file and store each column in a separate list
columns = []
with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    headers = next(reader)  # Skip the header row
    for header in headers:
        columns.append([])  # Create a list for each column

    for row in reader:
        for i, value in enumerate(row):
            columns[i].append(value)

# Write the column data to a text file
with open('output.txt', 'w', encoding='utf-8') as file:
    for i, column in enumerate(columns):
        file.write(f"Column {headers[i]}: {column}\n\n")

print("Data has been successfully written to 'output.txt'")

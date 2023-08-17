import csv

def save_results_to_csv(new_row, filename):
    with open(filename, 'a', newline='') as file:
     writer = csv.writer(file)
     writer.writerows(new_row)
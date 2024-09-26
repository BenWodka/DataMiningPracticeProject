import csv

def txt_to_csv(input_txt_file, output_csv_file):
    with open(input_txt_file, 'r', encoding='utf-8') as txtfile:
        reader = csv.reader(txtfile, delimiter=',')
        with open(output_csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            for row in reader:
                writer.writerow(row)

# Example usage:
input_txt_file = 'adult/adult.test'    # Replace with your input text file name
output_csv_file = 'adult/adult-test.csv'  # Replace with your desired output CSV file name

txt_to_csv(input_txt_file, output_csv_file)

def remove_question_mark_lines(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    with open(output_file, 'w') as file:
        for line in lines:
            if '?' not in line:
                file.write(line)

# Usage example
input_file = 'adult/adult.data'  # Replace with your input file path
output_file = 'adult/adult.data'  # Replace with your desired output file path
remove_question_mark_lines(input_file, output_file)

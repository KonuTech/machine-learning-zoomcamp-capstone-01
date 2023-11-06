import os

def should_exclude_directory(directory_name):
    # Add directory names you want to exclude here
    excluded_directories = ['.ipynb_checkpoints', 'venv', 'refs', 'HEAD']
    return directory_name in excluded_directories

def print_project_structure(directory, indent='', output_file=None):
    if output_file:
        with open(output_file, 'a') as file:
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isfile(item_path):
                    print(indent + '|-- ' + item, file=file)
                elif os.path.isdir(item_path) and not should_exclude_directory(item):
                    print(indent + '|-- ' + item + '/ (directory)', file=file)
                    print_project_structure(item_path, indent + '    ', output_file=output_file)

# Replace 'your_project_directory' with the actual path to your project directory
project_directory = os.path.join('C:\\', 'Users', 'KonuTech', 'zoomcamp-capstone-01')

output_file = 'project_structure.txt'  # Specify the output file name

if os.path.exists(project_directory) and os.path.isdir(project_directory):
    with open(output_file, 'w') as file:
        file.write('Project Structure:\n')
    print_project_structure(project_directory, output_file=output_file)
else:
    print('The specified project directory does not exist.')

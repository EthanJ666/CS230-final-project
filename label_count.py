import os

# Path to the directory containing your label .txt files
label_dir = './drone_dataset/train/labels'

# Initialize counters
count_0 = 0
count_1 = 0

# Loop through each .txt file in the directory
for filename in os.listdir(label_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(label_dir, filename)
        
        # Open and read each line in the file
        with open(file_path, 'r') as file:
            for line in file:
                # Split the line by spaces and get the first element
                drone_presence = line.strip().split()[0]
                
                # Check if the first number is '0' or '1' and update counters
                if drone_presence == '0':
                    count_0 += 1
                elif drone_presence == '1':
                    count_1 += 1

# Print the results
print(f"Number of labels starting with 0: {count_0}")
print(f"Number of labels starting with 1: {count_1}")

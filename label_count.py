import os

label_dir = './drone_dataset/train/labels'

count_0 = 0
count_1 = 0

for filename in os.listdir(label_dir):
    if filename.endswith('.txt'):
        file_path = os.path.join(label_dir, filename)
        
        with open(file_path, 'r') as file:
            for line in file:
                drone_presence = line.strip().split()[0]
                
                if drone_presence == '0':
                    count_0 += 1
                elif drone_presence == '1':
                    count_1 += 1

print(f"Number of labels starting with 0: {count_0}")
print(f"Number of labels starting with 1: {count_1}")

import pandas as pd

# Replace with your Excel file path and sheet name
excel_file_path = 'results.xlsx'
sheet_name = 'Sheet1'  # or the name of your sheet

# Read the Excel file
df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

# Replace 'column_name' with your actual column names
i1 = epoch = df['epoch'].tolist()
i2 = gpu_mem = df['gpu_mem'].tolist()
i3 = train_box = df['train_box'].tolist()
i4 = train_obj = df['train_obj'].tolist()
i5 = train_cls = df['train_cls'].tolist()
i6 = total = df['total'].tolist()
i7 = labels = df['labels'].tolist()
i8 = img_size = df['img_size'].tolist()
i9 = precision = df['precision'].tolist()
i10 = recall = df['recall'].tolist()
i11 = map50 = df['map50'].tolist()
i12 = map5095 = df['map50:95'].tolist()
i13 = val_box = df['val_box'].tolist()
i14 = val_obj = df['val_obj'].tolist()
i15 = val_cls = df['val_cls'].tolist()

columns = ['epoch', 'gpu_mem', 'train_box', 'train_obj', 'train_cls', 'total', 'labels', 'img_size', 'precision', 'recall', 'map50', 'map5095', 'val_box', 'val_obj', 'val_cls']

# Path for the output text file
output_text_file = 'results.txt'

# Writing data to a text file
with open(output_text_file, 'w') as file:
    # for c1, c2 in zip(epoch, gpu_mem, train_box, train_obj, train_cls, total, labels, img_size, precision, recall, map50, map5095, val_box, val_obj, val_cls):  # Add more columns if needed
    c = 1
    for i in columns:
        file.write(f'{i}: \n')  

print(f'Data written to {output_text_file}')

# code to extract the focus depth from each dicom file and save it in a csv file
import os
import csv
import pydicom

def extract_focus_depth(ippath, oppath):
    # Open the CSV file in write mode
    with open(oppath, 'w', newline='') as csvfile:
        fieldnames = ['File Name', 'Depth of Scan Field']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Iterate over the DICOM files in the folder
        for filename in os.listdir(ippath):
            if filename.endswith('.dcm'):
                filepath = os.path.join(ippath, filename)

                # Read the DICOM file
                dcm = pydicom.dcmread(filepath)


                # print(dcm.FocusDepth)
                # input()
                # Extract the focus depth value (replace 'Focus Depth' with the actual tag you want to extract)
                focus_depth = dcm.DepthOfScanField

                # Write the data to the CSV file
                writer.writerow({'File Name': filename, 'Depth of Scan Field': focus_depth})


if __name__ == '__main__':
    # Specify the folder containing DICOM files and the output CSV file path
    # Path to the folder containing the dicom files
    ippath = "C:/Users/kkotkar1/Desktop/DicomScaling/DicomSlices"

    # Path to the csv file
    oppath = "C:/Users/kkotkar1/Desktop/DicomScaling/focusdepthdata1.csv"

    # Call the function to extract the focus depth
    extract_focus_depth(ippath, oppath)

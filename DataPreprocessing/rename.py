import os
 
# Function to rename multiple files
def main():
   
    folder = "C:/Users/kkotkar1/Desktop/CropAllDicoms/YoloData/GroundTruthFixedCropDicom"
    for count, filename in enumerate(os.listdir(folder)):
        count1 = str(count+1).zfill(3)
        dst = f"dicom-{count1}.dcm"
        src = f"{folder}/{filename}"  # foldername/filename, if .py file is outside folder
        dst = f"{folder}/{dst}"
         
        # rename() function will rename all the files
        os.rename(src, dst)
 
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()
import pydicom as dicom
import os

def split_frames(ippath, oppath):
    # Load the dicom file
    for file in os.listdir(ippath):
        file_path = os.path.join(ippath, file)
        file = file.replace('.dcm', '')
        
        # Read the dicom file
        ds = dicom.dcmread(file_path)

        # Check if the dicom file has multiple frames
        if ((ds.NumberOfFrames) and (ds.NumberOfFrames > 1)):
            num_frames = int(ds.NumberOfFrames)

            os.makedirs(oppath, exist_ok=True)

            # Loop through the frames and save each frame as a dicom file
            for i in range(num_frames):
                frame = ds.copy()
                # frame.PixelData = ds.pixel_array[i].tobytes()
                frame.NumberOfFrames = 1
                frame.InstanceNumber = str(i + 1)
                saveoppath = os.path.join(oppath,f'{file}_frame{str(i + 1)}.dcm')
                frame.save_as(saveoppath)
            
            print(f'{num_frames} Frames extracted for {file}')
        else:
            print(f'{file} has only 1 frame or no frames so skipped')

if __name__ == '__main__': 
    ippath = 'C:/Users/kunal/Desktop/This PC/HEPIUS/Pre Injury Sagittal B Mode'
    oppath = 'C:/Users/kunal/Desktop/This PC/HEPIUS/Pre Injury Sagittal B Mode/frames'

    split_frames(ippath, oppath)

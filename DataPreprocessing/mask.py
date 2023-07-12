import os
import pydicom as pdcm
import numpy as np
import cv2

def keep_largest_area(img, connectivity=4, min_area=1, components=1):
    img_cp = img.copy()
    img_adj = np.asarray(img_cp, dtype=np.uint8)
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(img_adj, connectivity=connectivity)
    sort_indices = np.argsort(stats[:, -1])[::-1]

    for i in range(components + 1, nb_components):
        img_cp[output == sort_indices[i]] = 0

    # Discard largest component(s) if size less than min_area
    components_rv = []
    for i in range(components):
        if i < nb_components - 1:
            if stats[:, -1][sort_indices[1 + i]] <= min_area:
                img_cp[output == sort_indices[1 + i]] = 0
            else:
                components_rv.append(output == sort_indices[1 + i])
    return img_cp, components_rv

def create_mask(dir, fname):
    # all_dcms='RandomGroundTruthMasks'
    # for fname in os.listdir(all_dcms):
            # print(fname)
        # print(os.path.join(dir, fname))
        # print(dir)
        # print(fname)
        # SystemExit
        src_file = os.path.join(dir, fname)
        # print(src_file)
        dcm = pdcm.dcmread(src_file)
        img = dcm.pixel_array
        if dcm.PhotometricInterpretation != 'RGB':
            img = pdcm.pixel_data_handlers.util.convert_color_space(img, dcm.PhotometricInterpretation,
                                                                        'RGB',
                                                                        per_frame=True)

        binary = np.asarray(np.where((img[255:815, 211:1059, 0] > 127) & (img[255:815, 211:1059, 1] > 127) & (img[255:815, 211:1059, 2] < 127), 1, 0), dtype=np.uint8)

        kernel = np.ones((1,10), 'uint8')
        dilate_img = cv2.dilate(binary, kernel, iterations=5)

        # Floodfill from point (0, 0)
        mask = np.zeros((dilate_img.shape[0]+2, dilate_img.shape[1]+2), np.uint8)
        cv2.floodFill(np.asarray(dilate_img.copy(), dtype=np.uint8), mask, (0,0), 255)

        thresholded = np.where(mask == 1, 0, 1)
        final_img, _ = keep_largest_area(thresholded.copy())
        final_img = np.where(final_img == 1, 255, 0)
        return final_img

if __name__=='__main__':
    final_img = create_mask('..\RandomDicomGroundTruthMasks','dicom-045.dcm')
    print(final_img.shape)
    print(final_img)
    for i in final_img:
        print(i)

        
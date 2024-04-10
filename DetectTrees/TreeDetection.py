from ultralytics import YOLO
import cv2
import numpy as np
import os
import math
from PIL import Image

class FindTree:
    def __init__(self, model, showAllTrees=False) -> None:
        self.model = model
        self.showAllTrees = showAllTrees

    def process(self, image_location):

        result = self.model(image_location, verbose=False)

        #only take first (rest are not trees)
        result = result[0]
    
        return result


    def process_show(self, image_location):

        result = self.process(image_location)
        im_bgr = result.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
        result.show()


    def RemoveBackground(self, image, results):

        black_image = np.zeros_like(image)

        # only get the first tree
        if not self.showAllTrees:
            xyxy = [results.boxes.xyxy[0]]
        else:
            xyxy = results.boxes.xyxy

        for detection in xyxy:
            y_min = int(min(detection[1], detection[3]))
            y_max = int(max(detection[1], detection[3]))
            x_min = int(min(detection[0], detection[2]))
            x_max = int(max(detection[0], detection[2]))
            black_image[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]
        
        return black_image
    
    def RemoveBackground_Show(self, image_location):

        results = self.process(image_location)

        # check if there is at least 1 tree
        if results.boxes.xyxy == []:
            print("No tree found")
            return
        image = cv2.imread(image_location)

        masked_image = self.RemoveBackground(image, results)

        cv2.imshow(masked_image)


    def RemoveBackground_Save(self, image_location, save_location):

        results = self.process(image_location)

        # if results.boxes.xyxy == []:
        #     print("No tree found")
        #     return
        image = cv2.imread(image_location)

        try:
            masked_image = self.RemoveBackground(image, results)
            cv2.imwrite(save_location, masked_image)
        except:
            print("Error processing", image_location)
            return


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

# Example usage
if __name__ == "__main__":

   
    dataset_input_folder = "Data/drive-download-20240403T140634Z-001"
    dataset_output_folder = "Data/drive_processed_singleTree"

    model = YOLO("DetectTrees/best.pt")
    ft = FindTree(model,showAllTrees=False)

    # Count total files to count progress only .jpg files
    files = [f for _, _, files in os.walk(dataset_input_folder) for f in files if f.endswith('.jpg')]
    total_files = sum([len(files)])
    processed_files = 0
    old_progress = 0
    printProgressBar(0, total_files, prefix = 'Progress:', suffix = 'Complete', length = 50)

    for photo_folder_for in os.listdir(dataset_input_folder):

        # make sure the folder structure is the same
        photo_folder = os.path.join(dataset_input_folder, photo_folder_for)
        output_folder = os.path.join(dataset_output_folder, photo_folder_for)
    
        
        # check if the folder is a directory
        if not os.path.isdir(photo_folder):
            continue

        # create output folder if it does not exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(photo_folder):
            if filename.endswith(".jpg"):
                image_path = os.path.join(photo_folder, filename)
                # try:
                ft.RemoveBackground_Save(image_path, os.path.join(output_folder, filename))
                # except:
                    # print("Error processing", image_path)
                processed_files += 1

                printProgressBar(processed_files, total_files, prefix = 'Progress:', suffix = 'Complete', length = 50)
                # progress = math.ceil((processed_files / total_files) * 100)
                # if not (progress == old_progress):
                #     print(f"Progress: {progress}%")
                # old_progress = progress

    
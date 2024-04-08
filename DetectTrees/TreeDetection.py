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

        result = self.model(image_location)

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

        if len(results.boxes.xyxy) == 0:
            print("No tree found")
            return
        image = cv2.imread(image_location)

        masked_image = self.RemoveBackground(image, results)

        cv2.imshow(masked_image)


    def RemoveBackground_Save(self, image_location, save_location):

        results = self.process(image_location)

        if results.boxes.xyxy is None:
            print("No tree found")
            return
        image = cv2.imread(image_location)

        masked_image = self.RemoveBackground(image, results)

        cv2.imwrite(save_location, masked_image)



# Example usage
if __name__ == "__main__":

   
    dataset_input_folder = "Data/DummyData"
    dataset_output_folder = "Data/DummyData_TreeDetected2"

    model = YOLO("DetectTrees/best.pt")
    ft = FindTree(model,showAllTrees=True)

    # Count total files to count progress only .jpg files
    files = [f for _, _, files in os.walk(dataset_input_folder) for f in files if f.endswith('.jpg')]
    total_files = sum([len(files)])
    processed_files = 0

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
                progress = math.ceil((processed_files / total_files) * 100)
                print(f"Progress: {progress}%")

    
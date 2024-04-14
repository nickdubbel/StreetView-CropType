import os
import cv2
import numpy as np
import sys

def process_image(img):
    # Convert to grayscale (recommended)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Bomen segmenteren
    thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1]
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((5,5)))
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Als er geen contouren zijn, retourneer een lege array
    if not contours:
        return np.array([])

    # Bounding boxes en schalen
    boxes = [cv2.boundingRect(contour) for contour in contours]
    scales = [max(box[2], box[3]) for box in boxes]
    scaled_trees = [cv2.resize(img[y:y+h, x:x+w], (scale, scale)) for (x,y,w,h), scale in zip(boxes, scales)]

    # Zwart verwijderen
    canvas = np.zeros((max(scales), max(scales), 3), dtype=np.uint8)
    for tree, scale in zip(scaled_trees, scales):
        x = (max(scales) - scale) // 2
        y = (max(scales) - scale) // 2
        canvas[y:y+scale, x:x+scale] = tree

    # Apply average pooling to resize the image to the same size
    pool_size = 320  # Set the desired output size
    canvas_resized = cv2.resize(canvas, (pool_size, pool_size), interpolation=cv2.INTER_AREA)

    return canvas_resized

def process_images_in_folder(input_folder, output_folder):
    # Get a list of all files and directories in the input folder
    entries = os.listdir(input_folder)
    # Iterate over each entry
    for entry in entries:
        # Create full path
        full_path = os.path.join(input_folder, entry)
        # Check if the entry is a file
        if os.path.isfile(full_path):
            # Check if the file is a jpg
            if entry.endswith('.jpg'):
                # Read the image
                img = cv2.imread(full_path)
                # Process the image
                output = process_image(img)
                # If output is not empty, write the output image to the output folder
                if output.size != 0:
                    output_file = os.path.join(output_folder, os.path.splitext(entry)[0] + "_output.jpg")
                    cv2.imwrite(output_file, output)
                    # Get and print image size
                    output_size = output.shape[:2]  # Get height and width as a tuple
                    #print("Output image", entry, "size:", output_size[0], "x", output_size[1])
        # If the entry is a directory, recursively call process_images_in_folder
        elif os.path.isdir(full_path):
            # Create corresponding output folder
            output_subfolder = os.path.join(output_folder, entry)
            os.makedirs(output_subfolder, exist_ok=True)
            # Recursively process images in subfolder
            process_images_in_folder(full_path, output_subfolder)

# Input folder containing subfolders with images
# input_folder = "./DataBlack"
# Output folder where processed images will be saved
# output_folder = "./ResizedTreeData"

if len(sys.argv) < 3:
    print("Usage: python ScaleImage.py <input_folder> <output_folder>")
    exit()
input_folder = sys.argv[1]
output_folder = sys.argv[2]

# Check if input folder exists
if not os.path.exists(input_folder):
    print("Input folder does not exist")
    exit()

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Process images in the input folder
process_images_in_folder(input_folder, output_folder)

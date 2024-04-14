# StreetView-CropType

This project aims to classify tree types using street view images. It utilizes deep learning techniques to analyze the images and predict the type of crops present.

## Installation

1. Clone the repository:

    ```bash
    git clone git@github.com:nickdubbel/StreetView-CropType.git
    ```

2. Install the required [dependencies](requirements.txt):

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Creating data:

    place the tree dataset (shape files) in creatingData/TreehealthDataset.
    In road_points.py edit the name and the save location of the resulting csv file containing the location ,road and qualities of the trees.

    modify the getGSVImages.py to grep the correct .csv file (the one just made) and run it for the 6 categories changing the save location between each category run.

    

1. Detect trees:

    Use [DetectTrees/TreeDetection.py](DetectTrees/TreeDetection.py) to detect trees
    ```bash
    python DetectTrees/TreeDetection.py <input_folder> <output_folder>
    ```

    Use DetectTrees/YoloV8.ipynb to retrain the model if necessary

2. Scale the tree images:

    Use [DetectTrees/ScaleImage.py](DetectTrees/ScaleImage.py) to scale the images.
    ```bash
    python DetectTrees/ScaleImage.py <input_folder> <output_folder>
    ```

3. Train and evaluate the model:

    Use [Model/streetviewcroptypemapping.ipynb](Model/streetviewcroptypemapping.ipynb) to train the model.
    Don't forget to change the imagesRoot to your image location.


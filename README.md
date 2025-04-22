# cv-light-correction

## Zero-DCE Super Resolution Enhancement Tool

This application combines Zero-reference Deep Curve Estimation (Zero-DCE) with a Super Resolution CNN to enhance low-light and low-resolution images and videos.

![Demo GIF](assets/demo.gif)

*Demo showing the enhancement process in action*

## Features

- Enhance low-light images and videos using Zero-DCE network
- Apply super-resolution enhancement to improve detail
- Process both images and videos through a user-friendly GUI
- Real-time preview of processing results
- Save processed images and videos to your chosen location

## Requirements

- Python 3.6+
- PyTorch
- OpenCV
- Tkinter (included in standard Python installation)
- PIL (Python Imaging Library)

See `requirements.txt` for specific version requirements.

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the pre-trained models:
   - `z_dce_network_ZESR.pth`
   - `sr_cnn_ZESR.pth`

   Place these files in the same directory as the main script.

## Usage

Run the testing application:

```
python test.py
```

For training the models:

```
python train.py
```

### Processing Images
1. Click "Select Image"
2. Choose an image file (.png, .jpg, .jpeg)
3. View the original and processed images side by side
4. Save the processed image to your desired location

### Processing Videos
1. Click "Select Video"
2. Choose a video file (.mp4, .avi, .mov)
3. Select a save location for the processed video
4. Watch the processing progress with frame-by-frame preview
5. Receive notification when processing is complete

## Technical Overview

This application combines two neural networks:

1. **Zero-DCE Network**: Enhances low-light regions without reference images
2. **Super Resolution CNN**: Improves detail and resolution

The processing pipeline:
- Convert input to grayscale
- Process with Zero-DCE to enhance lighting
- Apply SR-CNN for detail enhancement
- Blend results with original image for balanced output

## File Structure

- `test.py`: Script for testing the model and running the GUI application
- `train.py`: Script for training the Zero-DCE and SR-CNN models
- `models.py`: Contains neural network model definitions
- `z_dce_network_ZESR.pth`: Pre-trained Zero-DCE model weights
- `sr_cnn_ZESR.pth`: Pre-trained Super Resolution CNN model weights
- `assets/`: Directory containing demonstration files
  - `demo.gif`: Animated demonstration of the application in use

## Controls

- Press "Escape" to exit fullscreen mode
- "Exit" button to close the application

## License

GNU General Public License v3.0

## Datasets

- [Vimeo-90K Dataset](http://toflow.csail.mit.edu/) - Used for training and evaluation of video enhancement models
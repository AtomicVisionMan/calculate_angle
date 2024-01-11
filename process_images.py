import sys, os
import cv2
import numpy as np
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt


def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between three points.
    """
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    angle_rad = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

def draw_angle(image, p1, p2, p3, angle):
    """
    Draw the angle and lines on the image.
    """
    # Draw lines connecting the points
    # cv2.line(image, p1, p2, (0, 255, 0), 2)
    # cv2.line(image, p2, p3, (0, 255, 0), 2)

    # Draw angle arc
    radius = 40  # radius of the angle arc
    angle_p1_p2 = np.arctan2(p1[1] - p2[1], p1[0] - p2[0])
    angle_p3_p2 = np.arctan2(p3[1] - p2[1], p3[0] - p2[0])

    # cv2.ellipse(image, p2, (radius, radius), angle=np.degrees(angle_p1_p2), startAngle=0, endAngle=np.degrees(angle_p3_p2 - angle_p1_p2), color=(255, 0, 0), thickness=2)

    # Calculate the midpoint for angle annotation
    mid_x = int((p1[0] + p3[0]) / 2)
    mid_y = int((p1[1] + p3[1]) / 2)

    # Annotate the angle
    cv2.putText(image, f"{angle:.2f} degrees", (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def process_image(image_path):
    """
    Process an individual image.
    """
    try:
        # Load the image
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform edge detection (You may need to adjust parameters)
        edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)

        # Detect corners (This is a placeholder and may need refinement)
        corners = cv2.goodFeaturesToTrack(edges, maxCorners=4, qualityLevel=0.01, minDistance=10)
        print(corners)

        # Draw red circles around each detected corner
        if corners is not None:
            for corner in corners:
                x, y = corner[0]
                cv2.circle(image, (int(x), int(y)), radius=5, color=(0, 0, 255), thickness=2)  # Red circle

        # Calculate the angle at the electrode tip
        if corners is not None and len(corners) == 4:
            angle = calculate_angle(corners[0][0], corners[1][0], corners[3][0])

            # Draw the calculated angle on the image
            draw_angle(image, tuple(map(int, corners[0][0])), tuple(map(int, corners[1][0])), tuple(map(int, corners[3][0])), angle)

            # Overlay the calculated angle on the image
            # cv2.putText(image, f"Angle: {angle:.2f} degrees", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Construct the correct result path
            result_dir = "processed_images"
            os.makedirs(result_dir, exist_ok=True)
            result_path = os.path.join(result_dir, os.path.basename(image_path))

            # Save the processed image
            cv2.imwrite(result_path, image)
            return result_path
        else:
            return None  # Failed to detect corners or not enough corners
    except Exception as e:
        print(f"Error processing image: {e}")
        return None  # Error occurred during processing

class ImageProcessingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Processing Tool")
        self.setGeometry(100, 100, 800, 600)  # Adjust the size as needed

        # Create a vertical layout
        layout = QVBoxLayout()

        # Create image display areas
        self.original_image_label = QLabel("Original Image")
        self.processed_image_label = QLabel("Processed Image")

        # Add image labels to the layout
        layout.addWidget(self.original_image_label)
        layout.addWidget(self.processed_image_label)

        # Create buttons for image selection
        self.single_image_button = QPushButton("Select Single Image")
        self.single_image_button.clicked.connect(self.select_single_image)

        self.multiple_images_button = QPushButton("Select Multiple Images")
        self.multiple_images_button.clicked.connect(self.select_multiple_images)

        self.folder_button = QPushButton("Select Folder")
        self.folder_button.clicked.connect(self.select_folder)

        # Add buttons to the layout
        layout.addWidget(self.single_image_button)
        layout.addWidget(self.multiple_images_button)
        layout.addWidget(self.folder_button)

        # Set the layout for the main window
        self.setLayout(layout)


    def select_single_image(self):
        print('\t=== [select_single_image] ===')
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        print('process image at ', file_name)
        
        if file_name:
            self.display_image(file_name, self.original_image_label)
            processed_image_path = process_image(file_name)
            if processed_image_path:
                print(f"Processed image saved at: {processed_image_path}")
                self.display_image(processed_image_path, self.processed_image_label)
            else:
                print("Image processing failed.")
        
    def display_image(self, image_path, label):
        # Load the image
        image = QImage(image_path)
        # Scale the image to fit the label size, maintaining aspect ratio
        image = image.scaled(label.size(), Qt.KeepAspectRatio)
        # Set the pixmap of the label
        label.setPixmap(QPixmap.fromImage(image))

    def select_multiple_images(self):
        options = QFileDialog.Options()
        file_names, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        for file_name in file_names:
            processed_image_path = process_image(file_name)
            if processed_image_path:
                print(f"Processed image saved at: {processed_image_path}")
            else:
                print(f"Image processing failed for: {file_name}")

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            # Process all images in the selected folder
            for file_name in os.listdir(folder_path):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    file_path = os.path.join(folder_path, file_name)
                    processed_image_path = process_image(file_path)
                    if processed_image_path:
                        print(f"Processed image saved at: {processed_image_path}")
                    else:
                        print(f"Image processing failed for: {file_path}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())

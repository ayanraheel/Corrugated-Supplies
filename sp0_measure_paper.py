import cv2
import numpy as np

def measure_paper(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around all detected contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with the bounding boxes
    cv2.imshow('Detected Damage', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = '/Users/colleenjung/Downloads/istockphoto-1222212636-612x612.jpg'
width, height = measure_paper(image_path)
print(f'Width: {width}px, Height: {height}px')
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "/Users/colleenjung/Desktop/UChicago/24SummerCorrugated/filtered_data/Foreign Substance/AMM14E25033B0 - Damage.jpg"
image = cv2.imread(image_path)

# Convert the image to grayscale
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of the original image for visualization
image_contours = image.copy()
cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)

# Calculate the area of the largest contour, assuming it represents the damaged area
largest_contour = max(contours, key=cv2.contourArea)
damage_area = cv2.contourArea(largest_contour)

# Calculate the area of the paper roll (excluding the core)
# Assuming a radius of 27 inches for the paper roll (diameter 54 inches)
total_paper_area = np.pi * (27) ** 2

# Calculate damage percentage
damage_percentage = (damage_area / total_paper_area) * 100
damagearea = (damage_percentage / 100) * 58 - 2  # core = 4 inches

# Resize the image to be larger
scale_factor = 2  # Scaling by 200%
dimensions = (image.shape[1] * scale_factor, image.shape[0] * scale_factor)
resized_image = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

# Convert to grayscale
image_gray_resized = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred_resized = cv2.GaussianBlur(image_gray_resized, (5, 5), 0)

# Apply Canny edge detection
edges_resized = cv2.Canny(blurred_resized, 50, 150)

# Find contours
contours_resized, _ = cv2.findContours(edges_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of the resized image for visualization
image_contours_resized = resized_image.copy()
cv2.drawContours(image_contours_resized, contours_resized, -1, (0, 255, 0), 2)

# Add grid lines to the marked image
step_size = 40  # Adjusting the grid size to match the resized image
for i in range(0, image_contours_resized.shape[1], step_size):
    cv2.line(image_contours_resized, (i, 0), (i, image_contours_resized.shape[0]), color=(255, 255, 255), thickness=1)
for j in range(0, image_contours_resized.shape[0], step_size):
    cv2.line(image_contours_resized, (0, j), (image_contours_resized.shape[1], j), color=(255, 255, 255), thickness=1)

# Define the text information
ID = "ID: 53764"
damage_text = f"Damage Area: {damage_percentage:.2f}%"
time_of_day = "Time: 14:30"

# Calculate the new height to add for the text box
extra_height = 50
new_height = image_contours_resized.shape[0] + extra_height

# Create a new image with extra space at the bottom
final_image = np.full((new_height, image_contours_resized.shape[1], 3), 255, dtype=np.uint8)  # White background
final_image[:image_contours_resized.shape[0], :] = image_contours_resized

# Define the position for the text in the new bottom area
org_id = (10, image_contours_resized.shape[0] + 20)
org_damage = (10, image_contours_resized.shape[0] + 35)
org_time = (10, image_contours_resized.shape[0] + 50)

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.4
color = (0, 0, 0)  # Black color text
thickness = 1

# Add the text to the new bottom area
cv2.putText(final_image, ID, org_id, font, font_scale, color, thickness, cv2.LINE_AA)
cv2.putText(final_image, damage_text, org_damage, font, font_scale, color, thickness, cv2.LINE_AA)
cv2.putText(final_image, time_of_day, org_time, font, font_scale, color, thickness, cv2.LINE_AA)

# Display the final image inline
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
plt.title('Final Annotated Image with Smaller Text Box')
plt.axis('off')
plt.show()

# Print damage percentage and adjusted damage area
print(f"Damage Percentage: {damage_percentage:.2f}%")
print(f"Adjusted Damage Area: {damagearea:.2f} square inches")

# Reload the image, resize it, and reapply the annotations with adjusted text box size and position

# Load the original image
image = cv2.imread('/mnt/data/image.png')

# Resize the image to be larger
scale_factor = 2  # Scaling by 200%
dimensions = (image.shape[1] * scale_factor, image.shape[0] * scale_factor)
resized_image = cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)

# Convert to grayscale
image_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(image_gray, (5, 5), 0)

# Apply Canny edge detection
edges = cv2.Canny(blurred, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on a copy of the resized image for visualization
image_contours = resized_image.copy()
cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)

# Add grid lines to the marked image
step_size = 40  # Adjusting the grid size to match the resized image
for i in range(0, image_contours.shape[1], step_size):
    cv2.line(image_contours, (i, 0), (i, image_contours.shape[0]), color=(255, 255, 255), thickness=1)
for j in range(0, image_contours.shape[0], step_size):
    cv2.line(image_contours, (0, j), (image_contours.shape[1], j), color=(255, 255, 255), thickness=1)

# Define the text information
ID = "ID: 53764"
damage_text = "Damage Area: 40.52%"
time_of_day = "Time: 14:30"

# Calculate the new height to add for the text box
extra_height = 50
new_height = image_contours.shape[0] + extra_height

# Create a new image with extra space at the bottom
final_image = np.full((new_height, image_contours.shape[1], 3), 255, dtype=np.uint8)  # White background
final_image[:image_contours.shape[0], :] = image_contours

# Define the position for the text in the new bottom area
org_id = (10, image_contours.shape[0] + 20)
org_damage = (10, image_contours.shape[0] + 35)
org_time = (10, image_contours.shape[0] + 50)

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

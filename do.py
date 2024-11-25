import cv2
image = cv2.imread('star/star/1.png')
print(f"Image shape: {image.shape}, Value range: {image.min()}-{image.max()}")

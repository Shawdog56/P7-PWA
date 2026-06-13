import cv2
import numpy as np

class ImageProcessor:
    @staticmethod
    def process(img, task):
        if task == "GRAYSCALE":
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif task == "EDGES":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
            return cv2.Canny(gray, 100, 200)
        elif task == "BLUR":
            return cv2.GaussianBlur(img, (7, 7), 0)
        elif task == "SHARPEN":
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            return cv2.filter2D(img, -1, kernel)
        elif task == "THRESHOLD":
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            return thresh
        return cv2.resize(img, (64, 64))
    

    @staticmethod
    def to_grayscale(image):
        """Converts image to Black and White."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    @staticmethod
    def apply_blur(image, sigma=5):
        """Applies Gaussian Blur to smooth the image."""
        return cv2.GaussianBlur(image, (sigma, sigma), 0)


    @staticmethod
    def detect_edges(image):
        """Uses Canny algorithm to find outlines."""
        # Grayscale is usually required before edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)


    @staticmethod
    def resize(image, width=100, height=100):
        """Rescales the image to specific dimensions."""
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    

    @staticmethod
    def rotate(image, angle=90):
        """Rotates the image by 90, 180, or 270 degrees."""
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image


    @staticmethod
    def flip(image, mode=1):
        """Flips image: 1 for horizontal, 0 for vertical, -1 for both."""
        return cv2.flip(image, mode)


    @staticmethod
    def adjust_brightness(image, value=30):
        """Increases or decreases brightness."""
        # We use a helper to prevent 'wraparound' (e.g., 250 + 10 becoming 4)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if len(image.shape) == 3 else None
        if hsv is not None:
            h, s, v = cv2.split(hsv)
            v = cv2.add(v, value)
            final_hsv = cv2.merge((h, s, v))
            return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return cv2.add(image, value)


    @staticmethod
    def threshold(image, limit=127):
        """Binarization: Turns image into pure Black and White (no gray)."""
        # Ensure it is grayscale first
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(image, limit, 255, cv2.THRESH_BINARY)
        return thresh


    @staticmethod
    def sharpen(image):
        """Applies a sharpening kernel to make edges pop."""
        kernel = np.array([[-1, -1, -1], 
                           [-1,  9, -1], 
                           [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)


    @staticmethod
    def add_watermark(image, text="PROCESSED"):
        """Draws text on the bottom-right of the image."""
        output = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Scale font based on image size
        scale = image.shape[1] / 500
        cv2.putText(output, text, (10, image.shape[0] - 20), 
                    font, scale, (255, 255, 255), 2, cv2.LINE_AA)
        return output
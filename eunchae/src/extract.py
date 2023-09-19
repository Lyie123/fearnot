from dataclasses import dataclass
from ultralytics import YOLO
import numpy as np
import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract'
model = YOLO('.././model/detector.pt')

@dataclass
class Area:
    x1: int
    y1: int
    width: int
    height: int
    config: str

    @property
    def x2(self) -> int:
        return self.x1+self.width
    
    @property
    def y2(self) -> int:
        return self.y1+self.height

def crop(img: np.array, area: Area):
    return img[area.y1:area.y2, area.x1:area.x2, :]


static_content = {
    '2560x1440': {
        'gold': Area(x1=1157, y1=1179, width=73, height=29, config='--psm 6 digits'),
        'level': Area(x1=420, y1=1179, width=23, height=28, config='--psm 6 digits'),
        'stage': Area(x1=1026, y1=13, width=50, height=31, config='--psm 6'),
        'streak': Area(x1=1324, y1=1173, width=35, height=33, config='--psm 6 digits'),
        'clock': Area(x1=1504, y1=11, width=42, height=33, config='--psm 6 digits'),
        'store': [
            Area(x1=643, y1=1387, width=181, height=38, config=''),
            Area(x1=912, y1=1387, width=181, height=38, config=''),
            Area(x1=1180, y1=1387, width=181, height=38, config=''),
            Area(x1=1450, y1=1387, width=181, height=38, config=''),
            Area(x1=1718, y1=1387, width=181, height=38, config=''),
        ]
        #'traits': [],
    }
}


def extract_dynamic_content(img: np.array) -> dict[str, str]:
    """
    Detection task via detection/classification network
    All image dimensions are supported

    Args:
        img (np.array): image where objects are searched in

    Returns:
        dict[str, str]: json contain detected and classified objects in images
    """ 
    results = model.predict(img, save=True, conf=0.80, line_width=1, iou=0.1)

    detections = {
        'board-champion': [],
        'bench-champion': [],
        'events': []
    }

    for result in results:
        for n in result:
            [t] = n.boxes.xyxy.tolist()
            names = model.names[int(n.boxes.cls)]
            conf = n.boxes.conf
            detections[names].append({
                'label': '',
                'stars': '',
                'x': int(t[0]),
                'y': int(t[1]),
                'width': int(t[2]-t[0]),
                'height': int(t[3]-t[1]),
                'probability': round(float(conf), 2)
            })
    return detections


def extract_static_content(img: np.array) -> dict[str, str]:
    """
    Read text from image via tesseract
    Crop image area which is defined in static_content
    Supported image dimensions:
    - 2560x1440

    Args:
        img (np.array): Fullscreen screenshot of tft. Dimension have to be supported

    Returns:
        dict[str, str]: json cotains attributes from image
    """
    def read_text(img: np.array, config: str='', thresholding: bool=True) -> str:
        """
        Read text from image via tesseract

        Args:
            img (np.array): fullscreen image of team fight tactics
            thresholding (bool, optional): Preprocessing image to enhance precision via tesseract
                                           Set text to black and background to white
                                           Defaults to True.

        Returns:
            str: text from image
        """

        if thresholding:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)

        return pytesseract.image_to_string(img, config=config).rstrip()
    
    data = {}
    resolution = f'{img.shape[1]}x{img.shape[0]}'

    assert resolution in static_content, f'"{resolution}" is no valid resolution'

    for k, v in static_content[f'{img.shape[1]}x{img.shape[0]}'].items():
        if isinstance(v, list):
            data[k] = []
            for n in v:
                data[k].append(read_text(crop(img, n), n.config))
            
        else:
            data[k] = read_text(crop(img, v), v.config)
    
    return data
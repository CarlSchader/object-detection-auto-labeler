import cv2, requests
import numpy as np
from io import BytesIO
from PIL import Image

def load_im(location):
  img = None
  # If location is a url
  if location.startswith('http'):
      response = requests.get(location)
      img = Image.open(BytesIO(response.content))
      img = np.array(img)
  else:
      img = cv2.imread(location)
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img
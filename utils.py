import requests
from PIL import Image

def load_im(location):
    # If location is a url
    if location.startswith('http') or location.startswith('https'):
        return Image.open(requests.get(location, stream=True).raw)
    else:
        return Image.open(location)
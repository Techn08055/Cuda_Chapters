import requests
from PIL import Image
import io
import os

def test_inference(image_path, url="http://localhost:8000/infer"):
    if not os.path.exists(image_path):
        # Create a dummy image if it doesn't exist
        print(f"Creating dummy image at {image_path}")
        img = Image.new('RGB', (640, 480), color = (73, 109, 137))
        img.save(image_path)

    with open(image_path, 'rb') as f:
        files = {'file': f}
        try:
            response = requests.post(url, files=files)
            if response.status_code == 200:
                print("Success!")
                print("Response:", response.json())
            else:
                print(f"Error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            print("Connection Error: Is the server running?")

if __name__ == "__main__":
    test_inference("bus.jpg")

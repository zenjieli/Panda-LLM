import re
from PIL import Image, ImageDraw


def extract_bounding_box(text):
    """
    Extract the bounding box from the text as (left, top, right, bottom)
    Example text: "Bounding box: [0, 0.5, 100, 100]"
    Example output: (0, 0.5, 100, 100)
    """

    # The regex pattern
    pattern = r"(\d+(\.\d+)?),\s*(\d+(\.\d+)?),\s*(\d+(\.\d+)?),\s*(\d+(\.\d+)?)"

    match = re.search(pattern, text)
    if match:
        # return left, top, right, bottom
        return float(match.group(1)), float(match.group(3)), float(match.group(5)), float(match.group(7))
    else:
        return None


def draw_bounding_box(img: Image.Image, bbox: tuple):
    """
    Draw a bounding box on the image
    Params:
        img: PIL Image
        bbox: (left, top, right, bottom) in range [0, 1]
    """
    draw = ImageDraw.Draw(img)
    w = img.width
    h = img.height
    draw.rectangle([bbox[0]*w, bbox[1]*h, bbox[2]*w, bbox[3]*h], outline="orange")
    return img


if __name__ == '__main__':
    print(extract_bounding_box("[0.123, 0.123, 0.234, 0.434]"))

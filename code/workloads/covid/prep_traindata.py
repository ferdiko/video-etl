import os,sys
# import ImageResizeF
# import PIL
from PIL import Image
# from PIL import Image,ImageFile



def process_image(image):
    "Processes the image"
    image = image.resize((50, 50), Image.ANTIALIAS) # or whatever you are doing to the image
    return image


if __name__ == "__main__":
    in_dir = 'train/incorrect'
    out_dir = 'train/incorrect_small'

    for f in os.listdir(in_dir):
        image = Image.open(os.path.join(in_dir, f))
        proc_image = process_image(image)
        proc_image.save(os.path.join(out_dir, f))

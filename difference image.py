from PIL import Image, ImageChops

img1 = Image.open('gambar.jpg')
img2 = Image.open('gambar1.jpg')

diff = ImageChops.difference(img1, img2)
if diff.getbbox():
    diff.show()
# else:
#     print("no difference at all")

# ImageChops.invert(img1) 
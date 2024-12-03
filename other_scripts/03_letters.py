from PIL import Image, ImageDraw, ImageFont
import numpy as np

list_of_fonts = ['arial.ttf', 
                 'times.ttf',
                 'cour.ttf', 
                 'verdana.ttf',
                 'georgia.ttf',
                 'comic.ttf',
                 'tahoma.ttf', 
                 'calibri.ttf', 
                 'arialbi.ttf',
                 'verdanab.ttf',
                 'timesbi.ttf', 
                 'trebuc.ttf', 
                 'NotoSans-Regular.ttf',
                 'DejaVuSerif.ttf',
                 'SourceSansPro-Regular.ttf',
                 'simfang.ttf', 
                 'pala.ttf', 
                 'Nirmala.ttf',
                 'LucidaSansRegular.ttf', 
                 'javatext.ttf']

def create_letter_image(letter, font, font_size=50):
    # Create a blank white image
    image = Image.new('L', (50, 50), color=255)  # 'L' mode is for grayscale
    draw = ImageDraw.Draw(image)

    # Load a font and render the letter
    font = ImageFont.truetype(font, font_size)
    draw.text((25, 25), letter, fill=0, font=font)  # Fill=0 for black text
    image = np.array(image)
    return image

import matplotlib.pyplot as plt

plt.figure()

for i, f in enumerate(list_of_fonts):
    # Example: Create an image for the letter 'A'
    plt.subplot(4, 5, i+1)
    image_A = create_letter_image('C', f, 20)
    plt.title(f)
    plt.imshow(image_A[25:, 25:40], cmap='Greys')
plt.show()

'''
LIST ALL FONTS:
'''

# import os
# import glob

# # Path to the Fonts directory
# fonts_directory = r"C:\Windows\Fonts"

# # List all .ttf and .otf files (TrueType and OpenType fonts)
# fonts = glob.glob(os.path.join(fonts_directory, "*.ttf")) + glob.glob(os.path.join(fonts_directory, "*.otf"))

# # Print the font file names
# for font in fonts:
#     print(font)
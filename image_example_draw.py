
from PIL import ImageDraw
from PIL import Image




CHEERFUL_COLOR = (255, 255, 0)  # yellow
SAD_COLOR = (0, 0, 255)  # blue
# Define the size of the smileys
SIZE = 28

cheerful_smiley = Image.new('RGB', (SIZE, SIZE), color=(255, 255, 255))
draw = ImageDraw.Draw(cheerful_smiley)
draw.arc((3, 12, SIZE-4, SIZE+10), 180, 360, fill=(0, 0, 0), width=2)
draw.rectangle((6, 3, 8, 7), fill=(0, 0, 0))  # Исправленная версия
draw.rectangle((20, 3, 22, 7), fill=(0, 0, 0))  # Исправленная версия
cheerful_smiley.show()




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

def create_letter_image(letter, font, noise=0.2, font_size=20):
    # Create a blank white image
    image = Image.new('L', (500, 20), color=255)  # 'L' mode is for grayscale
    draw = ImageDraw.Draw(image)

    # Load a font and render the letter
    font = ImageFont.truetype(font, font_size)
    draw.text((1, 1), letter, fill=0, font=font)  # Fill=0 for black text
    image = 1-np.array(image)/255.0
    noise = np.random.rand(image.shape[0], image.shape[1])<noise
    image = np.clip(image+noise, 0, 1)

    #return image[25:, 25:40]
    return image[:15, :8+8*len(letter)] # height, width
    #return image

import matplotlib.pyplot as plt

plt.figure()

for i, f in enumerate(list_of_fonts):
    # Example: Create an image for the letter 'A'
    plt.subplot(4, 5, i+1)
    image_A = create_letter_image('A', f, 0.05, 15)
    plt.title(f)
    plt.imshow(image_A, cmap='Greys')
plt.show()

import matplotlib.pyplot as plt

plt.figure()

image_word = create_letter_image('TAMALAMEQUE', 'calibri.ttf', 0.05, 15)
plt.imshow(image_word, cmap='Greys')
plt.show()


n = 10
symbols = ['A', 'B', 'C']

marker = 'X'

# words = ["Bat", "Car", "Dog", "Ego", "Fox", "Hat", "Jet", "Lip", "Man", "Net",
#     "Oak", "Pen", "Run", "Sun", "Top", "Vet", "Wet", "Zoo", "Cup", "Box"]



# # Prepare the dataset for 3-word prediction
# num_samples_per_word = 500  # Number of samples per word
# train_data_words = np.zeros((int(num_samples_per_word * len(words) * 0.8), n*16, 25))
# test_data_words = np.zeros((int(num_samples_per_word * len(words) * 0.2), n*16, 25))

# train_labels_words = np.zeros((int(num_samples_per_word * len(words) * 0.8), len(words)))
# test_labels_words = np.zeros((int(num_samples_per_word * len(words) * 0.2), len(words)))

# # Create training dataset
# for i in range(train_data_words.shape[0]):
#     random_word = np.random.choice(words)
#     random_font = np.random.choice(list_of_fonts[:15])  # Using the first 15 fonts for training
#     train_data_words[i, :, :] = create_letter_image(random_word, random_font).T
#     train_labels_words[i, words.index(random_word)] = 1  # One-hot encoding the word label

# # Create testing dataset
# for i in range(test_data_words.shape[0]):
#     random_word = np.random.choice(words)
#     random_font = np.random.choice(list_of_fonts[15:])  # Using the last 5 fonts for testing
#     test_data_words[i, :, :] = create_letter_image(random_word, random_font).T
#     test_labels_words[i, words.index(random_word)] = 1  # One-hot encoding the word label

# # Save the 3-word prediction dataset
# np.savez('three_letter_classification_dataset.npz', 
#          train_data=train_data_words, 
#          test_data=test_data_words, 
#          train_labels=train_labels_words, 
#          test_labels=test_labels_words)
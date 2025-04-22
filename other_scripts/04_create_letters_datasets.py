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
    image = Image.new('L', (500, 50), color=255)  # 'L' mode is for grayscale
    draw = ImageDraw.Draw(image)

    # Load a font and render the letter
    font = ImageFont.truetype(font, font_size)
    draw.text((25, 25), letter, fill=0, font=font)  # Fill=0 for black text
    image = 1-np.array(image)/255.0
    noise = np.random.rand(image.shape[0], image.shape[1])<0.1
    image = np.clip(image+noise, 0, 1)

    #return image[25:, 25:40]
    return image[25:, 25:25+16*len(letter)] # height, width
    #return image

# def create_word_image(word, font, font_size=20, noise=0.2):
#     """
#     Create an image for a word by rendering each letter and combining them.
#     """
#     image_width = 50 * len(word)  # 50 pixels per letter
#     image = Image.new('L', (image_width, 50), color=255)  # Blank white image
#     draw = ImageDraw.Draw(image)

#     x_offset = 0
#     for letter in word:
#         # Load font and render the letter
#         font = ImageFont.truetype(font, font_size)
#         draw.text((x_offset, 15), letter, fill=0, font=font)  # Fill=0 for black text
#         x_offset += 50  # Move the x position for the next letter

#     image = 1 - np.array(image) / 255.0  # Invert colors to have black text on white background
#     noise_layer = np.random.rand(image.shape[0], image.shape[1]) < noise
#     image = np.clip(image + noise_layer, 0, 1)

#     return image


# import matplotlib.pyplot as plt

# plt.figure()

# for i, f in enumerate(list_of_fonts):
#     # Example: Create an image for the letter 'A'
#     plt.subplot(4, 5, i+1)
#     image_A = create_letter_image('A', f, 20)
#     plt.title(f)
#     plt.imshow(image_A, cmap='Greys')
# plt.show()

# import matplotlib.pyplot as plt

# plt.figure()

# image_word = create_letter_image('TAMALAMEQUE', 'calibri.ttf', 20)
# plt.imshow(image_word, cmap='Greys')
# plt.show()

# '''
# single letter classification dataset
# train uses 15 fonts, test uses the 5 remaining fonts
# '''

# letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# num_samples_per_letter = 500

# random_flip_p = 0.2 # 20% of the signal randomly flipped for variability
# train_data = np.zeros((int(num_samples_per_letter*0.8*len(letters)), 16, 25))
# test_data = np.zeros((int(num_samples_per_letter*0.2*len(letters)), 16, 25))

# train_labels = np.zeros((int(num_samples_per_letter*0.8*len(letters)), len(letters)))
# test_labels = np.zeros((int(num_samples_per_letter*0.2*len(letters)), len(letters)))

# for x in range(train_data.shape[0]):
#     random_letter = np.random.choice([letter for letter in letters])
#     random_font = np.random.choice(list_of_fonts[:15])
#     train_data[x, :, :] = create_letter_image(random_letter, random_font).T
#     train_labels[x, letters.find(random_letter)]=1

# for x in range(test_data.shape[0]):
#     random_letter = np.random.choice([letter for letter in letters])
#     random_font = np.random.choice(list_of_fonts[15:])
#     test_data[x, :, :] = create_letter_image(random_letter, random_font).T
#     test_labels[x, letters.find(random_letter)]=1

# np.savez('letter_classification_dataset.npz', train_data=train_data, test_data=test_data, train_labels=train_labels, test_labels=test_labels)

# '''
# 3-word classification dataset
# '''

# words = ["Bat", "Car", "Dog", "Ego", "Fox", "Hat", "Jet", "Lip", "Man", "Net",
#     "Oak", "Pen", "Run", "Sun", "Top", "Vet", "Wet", "Zoo", "Cup", "Box"]

# n = 3

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

# '''
# 3-word prediction dataset
# '''

# words = ["Bat", "Car", "Dog", "Ego", "Fox", "Hat", "Jet", "Lip", "Man", "Net",
#     "Oak", "Pen", "Run", "Sun", "Top", "Vet", "Wet", "Zoo", "Cup", "Box"]

# n = 3

# # Prepare the dataset for 3-word prediction
# num_samples_per_word = 500  # Number of samples per word
# train_data_words = np.zeros((int(num_samples_per_word * len(words) * 0.8), n*8, 25))
# test_data_words = np.zeros((int(num_samples_per_word * len(words) * 0.2), n*8, 25))

# train_labels_words = np.zeros((int(num_samples_per_word * len(words) * 0.8), n*8, 25))
# test_labels_words = np.zeros((int(num_samples_per_word * len(words) * 0.2), n*8, 25))

# # Create training dataset
# for i in range(train_data_words.shape[0]):
#     random_word = np.random.choice(words)
#     random_font = np.random.choice(list_of_fonts[:15])  # Using the first 15 fonts for training
#     whole_letter = create_letter_image(random_word, random_font).T
#     train_data_words[i, :, :] = whole_letter[:n*8, :]
#     train_labels_words[i, :, :] = whole_letter[n*8:, :]  # One-hot encoding the word label

# # Create testing dataset
# for i in range(test_data_words.shape[0]):
#     random_word = np.random.choice(words)
#     random_font = np.random.choice(list_of_fonts[15:])  # Using the last 5 fonts for testing
#     whole_letter = create_letter_image(random_word, random_font).T
#     test_data_words[i, :, :] = whole_letter[:n*8, :]
#     test_labels_words[i, :, :] = whole_letter[n*8:, :]  # One-hot encoding the word label

# # Save the 3-word prediction dataset
# np.savez('three_letter_prediction_dataset.npz', 
#          train_data=train_data_words, 
#          test_data=test_data_words, 
#          train_labels=train_labels_words, 
#          test_labels=test_labels_words)

# '''
# 10-word classification dataset
# '''

# words = [
#     "Adventures", "Basketball", "Confidence", "Cryptogram", "Generation",
#     "Importance", "Javascript", "Literature", "Management", "Fellowship",
#     "Noteworthy", "Opposition", "Watermelon", "Questioner", "Revolution",
#     "Storyboard", "Technician", "Underrated", "Sculptures", "Journalist"
# ]


# n = 10

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
# np.savez('ten_letter_classification_dataset.npz', 
#          train_data=train_data_words, 
#          test_data=test_data_words, 
#          train_labels=train_labels_words, 
#          test_labels=test_labels_words)


# '''
# 10-word prediction dataset
# '''

# n = 10

# # Prepare the dataset for 3-word prediction
# num_samples_per_word = 500  # Number of samples per word
# train_data_words = np.zeros((int(num_samples_per_word * len(words) * 0.8), n*8, 25))
# test_data_words = np.zeros((int(num_samples_per_word * len(words) * 0.2), n*8, 25))

# train_labels_words = np.zeros((int(num_samples_per_word * len(words) * 0.8), n*8, 25))
# test_labels_words = np.zeros((int(num_samples_per_word * len(words) * 0.2), n*8, 25))

# # Create training dataset
# for i in range(train_data_words.shape[0]):
#     random_word = np.random.choice(words)
#     random_font = np.random.choice(list_of_fonts[:15])  # Using the first 15 fonts for training
#     whole_letter = create_letter_image(random_word, random_font).T
#     train_data_words[i, :, :] = whole_letter[:n*8, :]
#     train_labels_words[i, :, :] = whole_letter[n*8:, :]  # One-hot encoding the word label

# # Create testing dataset
# for i in range(test_data_words.shape[0]):
#     random_word = np.random.choice(words)
#     random_font = np.random.choice(list_of_fonts[15:])  # Using the last 5 fonts for testing
#     whole_letter = create_letter_image(random_word, random_font).T
#     test_data_words[i, :, :] = whole_letter[:n*8, :]
#     test_labels_words[i, :, :] = whole_letter[n*8:, :]  # One-hot encoding the word label

# # Save the 3-word prediction dataset
# np.savez('ten_letter_prediction_dataset.npz', 
#          train_data=train_data_words, 
#          test_data=test_data_words, 
#          train_labels=train_labels_words, 
#          test_labels=test_labels_words)


# '''
# 4-word prediction dataset (permuted without repetition)

# in total, 24 classes

# words = [
#     "ABCD", "ABDC", "ACBD", "ACDB", "ADBC", "ADCB",
#     "BACD", "BADC", "BCAD", "BCDA", "BDAC", "BDCA",
#     "CABD", "CADB", "CBAD", "CBDA", "CDAB", "CDBA",
#     "DABC", "DACB", "DBAC", "DBCA", "DCAB", "DCBA"
# ]

# '''

# from itertools import permutations
# words = [''.join(p) for p in permutations("ABCD")]

# n = 4

# # Prepare the dataset for 3-word prediction
# num_samples_per_word = 500  # Number of samples per word
# train_data_words = np.zeros((int(num_samples_per_word * len(words) * 0.8), n*8, 25))
# test_data_words = np.zeros((int(num_samples_per_word * len(words) * 0.2), n*8, 25))

# train_labels_words = np.zeros((int(num_samples_per_word * len(words) * 0.8), n*8, 25))
# test_labels_words = np.zeros((int(num_samples_per_word * len(words) * 0.2), n*8, 25))

# # Create training dataset
# for i in range(train_data_words.shape[0]):
#     random_word = np.random.choice(words)
#     random_font = np.random.choice(list_of_fonts[:15])  # Using the first 15 fonts for training
#     whole_letter = create_letter_image(random_word, random_font).T
#     train_data_words[i, :, :] = whole_letter[:n*8, :]
#     train_labels_words[i, :, :] = whole_letter[n*8:, :]  # One-hot encoding the word label

# # Create testing dataset
# for i in range(test_data_words.shape[0]):
#     random_word = np.random.choice(words)
#     random_font = np.random.choice(list_of_fonts[15:])  # Using the last 5 fonts for testing
#     whole_letter = create_letter_image(random_word, random_font).T
#     test_data_words[i, :, :] = whole_letter[:n*8, :]
#     test_labels_words[i, :, :] = whole_letter[n*8:, :]  # One-hot encoding the word label

# # Save the 3-word prediction dataset
# np.savez('four_letter_prediction_dataset.npz', 
#          train_data=train_data_words, 
#          test_data=test_data_words, 
#          train_labels=train_labels_words, 
#          test_labels=test_labels_words)


# '''
# 4-word clasddification dataset (permuted without repetition)

# in total, 24 classes

# words = [
#     "ABCD", "ABDC", "ACBD", "ACDB", "ADBC", "ADCB",
#     "BACD", "BADC", "BCAD", "BCDA", "BDAC", "BDCA",
#     "CABD", "CADB", "CBAD", "CBDA", "CDAB", "CDBA",
#     "DABC", "DACB", "DBAC", "DBCA", "DCAB", "DCBA"
# ]

# '''

# from itertools import permutations
# words = [''.join(p) for p in permutations("ABCD")]

# n = 4

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
# np.savez('four_letter_classification_dataset.npz', 
#          train_data=train_data_words, 
#          test_data=test_data_words, 
#          train_labels=train_labels_words, 
#          test_labels=test_labels_words)


'''
3-word clasddification dataset (permuted without repetition)

in total, 6 classes

words = [
    "ABC", "ACB", "BAC", "BCA", "CAB", "CBA"
]

'''

from itertools import permutations
words = [''.join(p) for p in permutations("ABC")]

n = 3

# Prepare the dataset for 3-word prediction
num_samples_per_word = 500  # Number of samples per word
train_data_words = np.zeros((int(num_samples_per_word * len(words) * 0.8), n*16, 25))
test_data_words = np.zeros((int(num_samples_per_word * len(words) * 0.2), n*16, 25))

train_labels_words = np.zeros((int(num_samples_per_word * len(words) * 0.8), len(words)))
test_labels_words = np.zeros((int(num_samples_per_word * len(words) * 0.2), len(words)))

# Create training dataset
for i in range(train_data_words.shape[0]):
    random_word = np.random.choice(words)
    random_font = np.random.choice(list_of_fonts[:15])  # Using the first 15 fonts for training
    train_data_words[i, :, :] = create_letter_image(random_word, random_font).T
    train_labels_words[i, words.index(random_word)] = 1  # One-hot encoding the word label

# Create testing dataset
for i in range(test_data_words.shape[0]):
    random_word = np.random.choice(words)
    random_font = np.random.choice(list_of_fonts[15:])  # Using the last 5 fonts for testing
    test_data_words[i, :, :] = create_letter_image(random_word, random_font).T
    test_labels_words[i, words.index(random_word)] = 1  # One-hot encoding the word label

# Save the 3-word prediction dataset
np.savez('three_permuted_letter_classification_dataset.npz', 
         train_data=train_data_words, 
         test_data=test_data_words, 
         train_labels=train_labels_words, 
         test_labels=test_labels_words)

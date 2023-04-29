from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import numpy as np
import os
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import ckwrap
 
app = Flask(__name__)
 
UPLOAD_FOLDER = 'static/uploads/'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
 
@app.route('/')
def home():
    return render_template('index.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        path = 'static/uploads/'
        text = ocr(path+filename)
        print(text)
        # return render_template('index.html', filename=filename)
        return render_template('text.html', content=text)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

def image_show(arr):
    plt.imshow(arr, cmap='gray')
    plt.show


def ocr(path):
    img = Image.open(path)
    img.show

    def add_margin(img, top, right, bottom, left, color):
        width, height = img.size
        new_width = width + right + left
        new_height = height + top + bottom

        result = Image.new('L', (new_width, new_height), color)
        result.paste(img, (left, top))

        return result

    img = add_margin(img, 10, 10, 10, 10, 255)

    img = ImageOps.grayscale(img)
    image_arr = np.asarray(img)

    def black_and_white(a):
        for i in range(len(a)):
            for j in range(len(a[0])):
                if a[i][j] > 200:
                    a[i][j] = 255
                else:
                    a[i][j] = 0
        return a

    image_arr = black_and_white(image_arr)

    # Line segementation
    line_path = './lines'
    if not os.path.isdir(line_path):
        os.makedirs(line_path)

    def calc_line_coords(coords):
        x_min = coords[0][0][0]
        x_max = coords[-1][0][0]
        y_min = 10000
        y_max = 0
        for i in coords:
            for j in i:
                y_min = min(y_min, j[1])
                y_max = max(y_max, j[1])

        return [x_min, x_max, y_min, y_max]

    coords = []
    line_coords = []
    for i in range(len(image_arr)):
        coo = []
        flag = 0
        for j in image_arr[i]:
            if j == 0:
                flag = 1
                break

        if flag == 1:
            for j in range(len(image_arr[i]) - 1):
                if image_arr[i][j] == 255:
                    if image_arr[i][j+1] == 0:
                        coo.append([i, j+1])

                if image_arr[i][j] == 0:
                    if image_arr[i][j+1] == 255:
                        coo.append([i, j])

        else:
            if (len(coords) > 0):
                cords = calc_line_coords(coords)
                line_coords.append(cords)
                Image.fromarray(image_arr[cords[0]: cords[1], cords[2]: cords[3]]).save(
                    line_path + '/line' + str(len(line_coords)) + '.png')
                coords = []

        if len(coo) > 0:
            coords.append(coo)

    avg_height = 0
    for i in line_coords:
        avg_height += (i[1] - i[0])
    avg_height = avg_height/len(line_coords)
    min_height = avg_height/3

    # Word segementation
    word_path = './words'
    if not os.path.isdir(word_path):
        os.makedirs(word_path)

    spaces_px = []
    letters_px = []
    word_sp = []
    words_coords = []

    for line in line_coords:
        sp = []
        lt = []
        sp_ctr = 0
        lt_ctr = 0
        a = image_arr[line[0]:line[1], line[2]:line[3]]

        for i in range(len(a[0, :])):
            f = 0
            for j in range(len(a[:, i])):
                if a[j, i] == 0:
                    lt_ctr += 1
                    f = 1
                    if sp_ctr != 0:
                        sp.append(sp_ctr)
                        sp_ctr = 0
                    break

            if f != 1:
                sp_ctr += 1
                if lt_ctr != 0:
                    lt.append(lt_ctr)
                    lt_ctr = 0
        lt.append(lt_ctr)

        # min_sp = (np.sum(sp)/len(sp)) + np.min(sp)
        sp_labels = ckwrap.ckmeans(sp, 2).labels
        min_sp = np.max(
            np.array(sp)[[i for i, x in enumerate(sp_labels) if x == 0]])
        word_sp.append(min_sp)
        spaces_px.append(sp)
        letters_px.append(lt)

        line_wrds = []
        x_min = line[0]
        x_max = line[1]
        y_min = 0
        y_max = 0

        for i in range(len(sp)):
            if sp[i] <= min_sp:
                y_max += lt[i] + sp[i]
            else:
                y_max += lt[i]
                line_wrds.append(
                    [x_min, x_max, y_min + line[2], y_max + line[2]])
                Image.fromarray(image_arr[x_min: x_max, y_min + line[2]: y_max + line[2]]).save(
                    word_path + '/line' + str(len(words_coords) + 1) + '_word' + str(len(line_wrds)) + '.png')

                y_min = y_max + sp[i]
                y_max = y_min

        y_max += lt[-1]
        line_wrds.append([x_min, x_max, y_min + line[2], y_max + line[2]])

        Image.fromarray(image_arr[x_min: x_max, y_min + line[2]: y_max + line[2]]).save(
            word_path + '/line' + str(len(words_coords) + 1) + '_word' + str(len(line_wrds)) + '.png')
        words_coords.append(line_wrds)

    # Letter segementation
    letters_coords = []
    for line in words_coords:
        li = []
        for word in line:
            wd = []
            a = image_arr[word[0]: word[1], word[2]:word[3]]
            lt_ctr = 0
            sp_ctr = 0
            x_min = word[0]
            x_max = word[1]
            y_min = word[2]
            y_max = 0
            for i in range(len(a[0, :])):
                f = 0
                for j in range(len(a[:, i])):
                    if a[j][i] == 0:
                        f = 1
                        lt_ctr += 1
                        if sp_ctr != 0:
                            y_min = y_max + sp_ctr
                            sp_ctr = 0
                        break

                if f != 1:
                    sp_ctr += 1
                    if lt_ctr != 0:
                        y_max = y_min + lt_ctr
                        lt_ctr = 0
                        wd.append([x_min, x_max, y_min, y_max])
            y_max = y_min + lt_ctr
            wd.append([x_min, x_max, y_min, y_max])
            li.append(wd)
        letters_coords.append(li)

    def border_rm(a):
        x_max = 0
        x_min = 10000


        for j in range(len(a[0])):
            for i in range(len(a)):
                if a[i][j] == 0:
                    x_min = min(x_min, i)
                    break

        for j in range(len(a[0])):
            for i in range(len(a) - 1, 0, -1):
                if a[i][j] == 0:
                    x_max = max(x_max, i)
                    break

        a = a[x_min: x_max, 0: len(a[0]) - 1]
        return a, x_max, x_min

    # model = load_model('./d_model_4_50_A.h5', compile=False)
    # model = load_model('./d_model_4_100_S.h5', compile=False)
    model = load_model('./d_model_4_50_A.h5', compile=False)

    labels = {
        0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',
        36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'
    }

    letter_path = './letters'
    if not os.path.isdir(letter_path):
        os.makedirs(letter_path)

    # Prediction
    final_text = ''
    line_ctr = 0
    word_ctr = 0
    lt_ctr = 0
    for line in letters_coords:
        line_ctr += 1
        for word in line:
            word_ctr += 1
            for letter in word:
                lt_ctr += 1
                a = image_arr[letter[0]: letter[1], letter[2]: letter[3]]
                a, t, b = border_rm(a)
                if t - b < min_height:
                    final_text += '.'
                    continue

                letter_image = Image.fromarray(a, mode='L')
                letter_image = add_margin(letter_image, 10, 10, 10, 10, 255)

                letter_image.save(letter_path + '/line' + str(line_ctr) +
                                  '_word' + str(word_ctr) + '_letter' + str(lt_ctr) + '.png')

                letter_image = letter_image.resize((32, 32))
                a = np.asarray(letter_image)/255

                ans = labels[np.argmax(model.predict(
                    [a.reshape(32, 32, 1).tolist()], verbose=False))]

                final_text += ans
            final_text += ' '
        final_text += '\n'

    return final_text


 
if __name__ == "__main__":
    app.run()

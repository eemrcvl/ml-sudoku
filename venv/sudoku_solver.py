import cv2
import numpy as np
from grid import extract_digit, find_grid
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import imutils
import json

def solver(path, debug=False):
    model = load_model("MNIST_classifier_v2")
    image = cv2.imread(path)
    image = imutils.resize(image, width=600)

    try:
        (gridImage, warped) = find_grid(image, debug)
    except:
        return json.dumps("An error occurred, please try uploading a different image")

    board = np.zeros((9, 9), dtype="int")

    step_x = warped.shape[1] // 9
    step_y = warped.shape[0] // 9
    cells = []

    for i in range(0, 9):
        row = []
        for j in range(0, 9):
            start_x = j * step_x
            start_y = i * step_y
            end_x = (j + 1) * step_x
            end_y = (i + 1) * step_y
            #crop the cell from the warped transform image
            #extract the digit
            row.append((start_x, start_y, end_x, end_y))

            cell = warped[start_y:end_y, start_x:end_x]
            digit = extract_digit(cell, debug)

            if digit is not None:
                #resize the cell to 28x28 and do preprocessing for classification
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = model.predict(roi).argmax(axis=1)[0]
                board[i, j] = prediction

        cells.append(row)

    print("OCR Sudoku Grid:")
    grid = Sudoku(3, 3, board = board.tolist())
    grid.show()

    print("Solving...")
    solution = grid.solve()
    solution.show_full()
    final = json.dumps(solution.board)
    return final


'''SHOW SOLUTION ON IMAGE

for (cellR, boardR) in zip(cells, solution.board):
    for (box, digit) in zip(cellR, boardR):
        start_x, start_y, end_x, end_y = box

        text_x = int((end_x - start_x) * 0.33)
        text_y = int((end_y - start_y) * -0.2)
        text_x += start_x
        text_y += end_y

        cv2.putText(gridImage, str(digit), (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 0), 2)

    cv2.imshow("Sudoku Result", gridImage)
    cv2.waitKey(0)
'''
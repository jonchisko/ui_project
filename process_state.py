import numpy as np
import cv2
import copy

def get_mario(grayscale_image):
    #mario value
    mario = 107

    #get mario state
    mario_state = grayscale_image == mario
    mario_state.dtype = np.uint8
    mario_state[mario_state == 1] = 250

    #return mario state
    return mario_state

def get_floor(grayscale_image):
    #floor value
    floor = 124

    #get floor
    floor_state = grayscale_image == floor
    floor_state.dtype = np.uint8
    floor_state[floor_state == 1] = 130

    #return floor state
    return floor_state

def get_coinblocks(grayscale_image):
    #coinblocks 1 and 2 values
    coinblocks1 = 177
    coinblocks2 = 52

    #get coinblocks with value1
    coinblocks1_state = grayscale_image == coinblocks1
    coinblocks1_state.dtype = np.uint8
    coinblocks1_state[coinblocks1_state == 1] = 130

    #get coinblocks with value2
    coinblocks2_state = grayscale_image == coinblocks2
    coinblocks2_state.dtype = np.uint8
    coinblocks2_state[coinblocks2_state == 1] = 130

    #return both combined
    return coinblocks1_state + coinblocks2_state

def get_pipes(grayscale_image):
    #pipe ligh green value
    pipe_light_green = 203

    #extract just the light green parts
    pipe_light_green_state = grayscale_image == pipe_light_green

    #convert to uint
    pipe_light_green_state.dtype = np.uint8

    #change to binary picture
    pipe_light_green_state[pipe_light_green_state == 1] = 255

    #find countours
    contours, hierarchy = cv2.findContours(pipe_light_green_state, 1, 2)

    #according to y coordinate of the bounding rectangles we sort the light green regions into pipes and not pipes
    pipes = []
    not_pipes = []
    for i in range(len(contours)):

        x, y, w, h = cv2.boundingRect(contours[i])
        if y < 185:
            pipes.append((x,y,w,h))
        else:
            not_pipes.append((x,y,w,h))

    #we then set the pixels of the regions according to pipe or not pipe status
    for pipe in pipes:
        x,y,w,h = pipe
        pipe_light_green_state[y:y+h,x:x+w] = 130
    for not_pipe in not_pipes:
        x, y, w, h = not_pipe
        pipe_light_green_state[y:y + h, x:x + w] = 0

    #return only pipes
    return pipe_light_green_state

def get_koopas(grayscale):
    #dark green value
    koopa_dark_green = 99

    #get dark green areas
    koopa_dark_green_state = grayscale == koopa_dark_green
    koopa_dark_green_state.dtype = np.uint8
    koopa_dark_green_state[koopa_dark_green_state == 1] = 130

    # find countours
    contours, hierarchy = cv2.findContours(koopa_dark_green_state, 1, 2)

    # according to y coordinate of the bounding rectangles we sort the light green regions into pipes and not pipes
    koopas = []
    not_koopas = []
    for i in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[i])
        if y < 190 and y > 180:
            koopas.append((x, y, w, h))
        else:
            not_koopas.append((x, y, w, h))

    # we then set the pixels of the regions according to pipe or not pipe status
    for koopa in koopas:
        x, y, w, h = koopa
        koopa_dark_green_state[y:y + h, x:x + w] = 130
    for not_koopa in not_koopas:
        x, y, w, h = not_koopa
        koopa_dark_green_state[y:y + h, x:x + w] = 0

    return koopa_dark_green_state

def get_flag(grayscale_image):
    # pipe ligh green value
    pipe_light_green = 203

    # extract just the light green parts
    pipe_light_green_state = grayscale_image == pipe_light_green

    # convert to uint
    pipe_light_green_state.dtype = np.uint8

    # change to binary picture
    pipe_light_green_state[pipe_light_green_state == 1] = 255

    # find countours
    contours, hierarchy = cv2.findContours(pipe_light_green_state, 1, 2)

    # according to y coordinate of the bounding rectangles we sort the light green regions into pipes and not pipes
    pipes = []
    not_pipes = []
    for i in range(len(contours)):

        x, y, w, h = cv2.boundingRect(contours[i])
        if y < 60:
            pipes.append((x, y, w, h))
        else:
            not_pipes.append((x, y, w, h))

    # we then set the pixels of the regions according to pipe or not pipe status
    for pipe in pipes:
        x, y, w, h = pipe
        pipe_light_green_state[y:y + h, x:x + w] = 70
    for not_pipe in not_pipes:
        x, y, w, h = not_pipe
        pipe_light_green_state[y:y + h, x:x + w] = 0

    # return only pipes
    return pipe_light_green_state

"""self.tile_height = 13
self.tile_width = 16
state.shape = (240, 256, 3)
240 = vrstic sepravi y
256 = stolpcev sepravi x
stolpci = 256
vrstice = 240"""

def process(state, mode=0):
    if mode == 0:
        state = state[...,::-1]
        img5 = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        result = get_mario(img5) + get_floor(img5) + get_coinblocks(img5)  + get_pipes(img5) + get_koopas(img5) + get_flag(img5)
        return get_discrete_state(result)
    if mode == 1:
        img5 = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
        result = get_mario(img5) + get_floor(img5) + get_coinblocks(img5) + get_pipes(img5) + get_koopas(img5) + get_flag(img5)
        show_grid(result)
        return result

def mario_in(presteto):
    for a in presteto:
        if a[0] == 250:
            return True
    else:
        return False

def flag_in(presteto):
    for a in presteto:
        if a[0] == 200:
            return True
    else:
        return False

#counts unique values in the square and returns the value for this square
def handle_square(square):
    flat_square = square.flatten()
    y = np.bincount(flat_square)
    ii = np.nonzero(y)[0]

    #list terk, kjer terka[0] enako element, in terka[1] enako koliko je tega elementa
    presteto = list(zip(ii,y[ii]))
    if mario_in(presteto):
        return 3
    elif flag_in(presteto):
        return 2
    else:
        najvecji = max(presteto, key=lambda x: x[1])
        if najvecji[0] == 0:
            return 0
        elif najvecji[0] == 130:
            return 1


def get_discrete_state(processed_state):
    image_height = processed_state.shape[0]
    image_width = processed_state.shape[1]
    tile_height = 13
    tile_width = 16
    matrikca = []
    for j in range(0, image_height, tile_height):
        vrstica = []
        for i in range(0, image_width, tile_width):
            """upper_left = (i, j)
            bottom_right = (i + tile_width, j + tile_height)"""

            square = processed_state[j:j+tile_height, i:i+tile_width]
            vrednost = handle_square(square)
            vrstica.append(vrednost)
        matrikca.append(vrstica)
    return matrikca

def show_grid(processed_state):
    processed_state = copy.copy(processed_state)
    image_height = processed_state.shape[0]
    image_width = processed_state.shape[1]
    tile_height = 13
    tile_width = 16
    ploscice_x = image_width/tile_width
    ploscice_y = image_height/tile_height
    print(ploscice_x, ploscice_y)
    print(processed_state.shape)
    vertical_lines_start = []
    vertical_lines_end = []
    for i in range(0,image_width,tile_width):
        vertical_lines_start.append((i,0))
    for i in range(0, image_width, tile_width):
        vertical_lines_end.append((i, image_height))

    horizontal_lines_start = []
    horizontal_lines_end = []
    for i in range(0,image_height,tile_height):
        horizontal_lines_start.append((0, i))
    for i in range(0,image_height,tile_height):
        horizontal_lines_end.append((image_width, i))

    for i in range(len(vertical_lines_start)):
        cv2.line(processed_state, vertical_lines_start[i], vertical_lines_end[i], 255, 1, 1)

    for i in range(len(horizontal_lines_start)):
        cv2.line(processed_state, horizontal_lines_start[i], horizontal_lines_end[i], 255, 1, 1)

    cv2.imshow("grid", processed_state)
    cv2.waitKey()

    return processed_state
if __name__ == "__main__":
    img1 = cv2.imread("testflag.png")
    cv2.imshow("original", img1)
    cv2.waitKey()
    processed = process(img1, mode=1)
    get_discrete_state(processed)


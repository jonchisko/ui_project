import numpy as np
import cv2
import copy
import time


class Img2State:

    __MARIO = 3
    __EMPTY = 0
    __OBSTACLE = 1
    __FLAG = 2

    def __init__(self, rows = 19, columns = 16):
        self.__n_row = rows
        self.__n_col = columns

        self.__right_close = 2
        self.__left_close = 2

    def transfrom(self, state, raw = False, variant = 0):
        if raw:
            return Img2State.__process(state, variant=variant, mode=0)
        else:
            return Img2State.__createFeatures(Img2State.__process(state, variant=variant, mode=0))

# Domen
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    def __process(state, variant=0, mode=0):
        if mode == 0:
            state = state[..., ::-1]
            img5 = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            result = Img2State.__get_mario(img5) + Img2State.__get_floor(img5) + Img2State.__get_coinblocks(img5) + \
                     Img2State.__get_pipes(img5) + Img2State.__get_koopas(img5) + Img2State.__get_flag(img5)
            if variant == 0:
                return Img2State.__get_discrete_state(result)
            if variant == 1:
                return Img2State.__get_discrete_state2(result)
            if variant == 2:
                return Img2State.__get_discrete_state3(result)
        if mode == 1:
            img5 = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
            result = Img2State.__get_mario(img5) + Img2State.__get_floor(img5) + Img2State.__get_coinblocks(img5) + \
                     Img2State.__get_pipes(img5) + Img2State.__get_koopas(img5) + Img2State.__get_flag(img5)
            # show_grid(result)
            if variant == 0:
                return Img2State.__get_discrete_state(result)
            if variant == 1:
                return Img2State.__get_discrete_state2(result)
            if variant == 2:
                return Img2State.__get_discrete_state3(result)

    def __get_mario(grayscale_image):
        #mario value
        mario = 107

        #get mario state
        mario_state = grayscale_image == mario
        mario_state.dtype = np.uint8
        mario_state[mario_state == 1] = 250

        #return mario state
        return mario_state

    def __get_floor(grayscale_image):
        #floor value
        floor = 124

        #get floor
        floor_state = grayscale_image == floor
        floor_state.dtype = np.uint8
        floor_state[floor_state == 1] = 130

        #return floor state
        return floor_state

    def __get_coinblocks(grayscale_image):
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

    def __get_pipes(grayscale_image):
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

    def __get_koopas(grayscale):
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

    def __get_flag(grayscale_image):
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

    def __mario_in(presteto):
        for a in presteto:
            if a[0] == 250:
                return True
        else:
            return False

    def __flag_in(presteto):
        for a in presteto:
            if a[0] == 200:
                return True
        else:
            return False

    #counts unique values in the square and returns the value for this square
    def __handle_square(square):
        flat_square = square.flatten()
        y = np.bincount(flat_square)
        ii = np.nonzero(y)[0]

        #list terk, kjer terka[0] enako element, in terka[1] enako koliko je tega elementa
        presteto = list(zip(ii,y[ii]))
        if Img2State.__mario_in(presteto):
            return 3
        elif Img2State.__flag_in(presteto):
            return 2
        else:
            najvecji = max(presteto, key=lambda x: x[1])
            if najvecji[0] == 0:
                return 0
            elif najvecji[0] == 130:
                return 1

    def __get_discrete_state(processed_state):
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
                vrednost = Img2State.__handle_square(square)
                vrstica.append(vrednost)
            matrikca.append(vrstica)
        return matrikca

    # state revolves around mario
    def __get_discrete_state2(state):
        image_height = state.shape[0]
        image_width = state.shape[1]
        y, x = np.where(state == 250)
        if y.size == 0:
            return Img2State.__get_discrete_state(state)
        up_x = x[0]
        up_y = y[0]

        down_x = x[-1]
        down_y = y[-1]
        cv2.rectangle(state, (up_x, up_y), (down_x, down_y), 165, 1)
        m_width = 15
        m_height = 15

        coef_left = up_x // m_width

        starting_x = up_x - (coef_left * m_width)

        coef_up = down_y // m_height
        starting_y = down_y - (coef_up * m_height)
        matrikca = []
        for i in range(starting_y, image_width, 15):
            vrstica = []
            for j in range(starting_x, image_height, 15):
                square = state[i:i + 15, j:j + 15]

                h, w = square.shape
                if w and h:
                    v = Img2State.__handle_square(square)
                    vrstica.append(v)
            if vrstica:
                matrikca.append(vrstica)
        return matrikca

    def __get_discrete_state3(processed_state):
        image_height = processed_state.shape[0]
        image_width = processed_state.shape[1]
        tile_height = 15
        tile_width = 15
        matrikca = []
        for j in range(0, image_height, tile_height):
            vrstica = []
            for i in range(0, image_width, tile_width):
                """upper_left = (i, j)
                bottom_right = (i + tile_width, j + tile_height)"""

                square = processed_state[j:j + tile_height, i:i + tile_width]
                vrednost = Img2State.__handle_square(square)
                vrstica.append(vrednost)
            matrikca.append(vrstica)
        return matrikca



    def __show_grid(processed_state):
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

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

# Jon
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    def __createFeatures(self, current_state):
        """
        :param state: image nXn
        :return: returns an array of size of number of features
        """
        features = [False for _ in range(10)]

        # find Mario
        self.__mario_row, self.__mario_col = self.__findMario(current_state)

        # stuck
        features[0] = self.__stuck(current_state)
        # gap left far
        features[1] = self.__gap_left_far(current_state)
        # gap right far
        features[2] = self.__gap_right_far(current_state)
        # gap left close
        features[3] = self.__gap_left_close(current_state)
        # gap right close
        features[4] = self.__gap_right_close(current_state)
        # gap below
        features[5] = self.__gap_below(current_state)
        # obstacle right close
        features[6] = self.__obstacle_right_close(current_state)
        # obstacle right far
        features[7] = self.__obstacle_right_far(current_state)
        # goal
        goals = self.__goal_sight(current_state)
        # goal right close
        features[8] = goals[0]
        # goal in sight
        features[9] = goals[1]

        return features

    def __stuck(self, state):
        try:
            if state[self.__mario_row][self.__mario_col+1] == self.__OBSTACLE:
                return True
        except:
            return False

        return False

    def __gap_left_far(self, state):
        for i in range(self.__mario_col - 1 - self.__left_close, -1, -1):
            if self.__straighBelow(self.__mario_row + 1, i, state):
                return True
        return False

    def __gap_right_far(self, state):
        for i in range(self.__mario_col + 1 + self.__right_close, self.__n_col):
            if self.__straighBelow(self.__mario_row + 1, i, state):
                return True
        return False

    def __gap_left_close(self, state):
        for i in range(self.__mario_col - 1, self.__mario_col - 1 - self.__left_close if self.__mario_col - 1 - self.__left_close >= -1 else -1, -1):
            if self.__straighBelow(self.__mario_row + 1, i, state):
                return True
        return False

    def __gap_right_close(self, state):
        for i in range(self.__mario_col + 1, self.__n_col if self.__mario_col + 1 + self.__right_close >=
                                                         self.__n_col else self.__mario_col + 1 + self.__right_close):
            if self.__straighBelow(self.__mario_row+1, i, state):
                return True
        return False

    def __straighBelow(self, row, col, state):
        for i in range(row, self.__n_row):
            if state[i][col] == self.__OBSTACLE:
                return False
        return True

    def __gap_below(self, state):
        # straight down
        for i in range(self.__mario_row+1, self.__n_row):
            if state[i][self.__mario_col] == self.__OBSTACLE:
                return False
        return True

    def __obstacle_right_close(self, state):
        # straight right, is there an obstacle?
        for i in range(self.__mario_col + 1, self.__n_col if (self.__mario_col + 1 + self.__right_close) >=
                                                         self.__n_col else (self.__mario_col + 1 + self.__right_close)):
            if state[self.__mario_row][i] == self.__OBSTACLE:
                return True
        return False

    def __obstacle_right_far(self, state):
        for i in range(self.__mario_col + 1 + self.__right_close, self.__n_col):
            if state[self.__mario_row][i] == self.__OBSTACLE:
                return True
        return False

    def __goal_close(self, state):
        for i in range(self.__mario_col + 1, self.__n_col if self.__mario_col + 1 + self.__right_close >=
                                                         self.__n_col else self.__mario_col + 1 + self.__right_close):
            if state[self.__mario_row][i] == self.__FLAG:
                return True
        return False

    def __goal_sight(self, state):
        if self.__goal_close(state):
            return True, True
        for row in range(self.__n_row):
            for col in range(self.__n_col):
                if state[row][col] == self.__FLAG:
                    return False, True
        return False, False

    def __findMario(self, state):
        def findRightBottom(state):
            row = self.__n_row - 1
            while row > 0:
                col = self.__n_col - 1
                while col > 0:
                    if state[row][col] == self.__MARIO:
                        return row, col
                    col -= 1
                row -= 1

            return -1, -1

        row, col = findRightBottom(state)

        if (row, col) == (-1, -1):
            return -1, -1

        # Find the rest of mario
        mario = [row]
        for i in range(1, 3):
            if state[row-i][col] == self.__MARIO:
                mario.append(row-i)

        if len(mario) == 3:
            return row-1, col
        else:
            return row, col



if __name__ == "__main__":
    img1 = cv2.imread("testflag.png")
    cv2.imshow("original", img1)
    cv2.waitKey()

    img2state = Img2State()
    processed = img2state.process(img1)


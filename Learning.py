



class Learning:

    """SIMPLE_MOVEMENT = [
        ['NOOP'],
        ['right'],
        ['right', 'A'],
        ['right', 'B'],
        ['right', 'A', 'B'],
        ['A'],
        ['left'],
    ]"""

    MARIO = 3
    EMPTY = 0
    OBSTACLE = 1
    FLAG = 2

    def __init__(self, current_state):
        self.n_row = len(current_state)
        self.n_col = len(current_state[0])

        self.right_close = 2
        self.left_close = 2

    def createFeatures(self, current_state, current_action):
        """
        :param state: image nXn
        :param action: action
        :return: returns an array of size of number of features
        """
        features = [False for _ in range(15)]
        # 0 NOOP
        # 1 right
        # 2 A
        # 3 B
        # 4 left
        if current_action == 0:
            features[0] = True

        if current_action == 1:
            features[1] = True

        if current_action == 2:
            features[1] = True
            features[2] = True

        if current_action == 3:
            features[1] = True
            features[3] = True

        if current_action == 4:
            features[1] = True
            features[2] = True
            features[3] = True

        if current_action == 5:
            features[2] = True

        if current_action == 6:
            features[4] = True

        # find Mario
        self.mario_row, self.mario_col = self.findMario(current_state)


        #### features from image
        #enemies = self.enemies_present(current_state)
        # right close
        #features[3] = enemies[0]
        # left close
        #features[4] = enemies[1]
        # below
        #features[5] = enemies[2]
        # present
        #features[6] = enemies[3]

        # stuck
        features[5] = self.stuck(current_state)
        # gap left far
        features[6] = self.gap_left_far(current_state)
        # gap right far
        features[7] = self.gap_right_far(current_state)
        # gap left close
        features[8] = self.gap_left_close(current_state)
        # gap right close
        features[9] = self.gap_right_close(current_state)
        # gap below
        features[10] = self.gap_below(current_state)
        # obstacle right close
        features[11] = self.obstacle_right_close(current_state)
        # obstacle right far
        features[12] = self.obstacle_right_far(current_state)
        # goal
        goals = self.goal_sight(current_state)
        # goal right close
        features[13] = goals[0]
        # goal in sight
        features[14] = goals[1]

        return features


    """
    def enemies_right_close(self, state):
        pass
    def enemies_left_close(self, state):
        pass
    def enemies_below(self, state):
        pass
    def enemies_present(self, state):
        arr = [False, False, False, False]
        arr[0] = self.enemies_right_close(state)
        arr[1] = self.enemies_left_close(state)
        arr[2] = self.enemies_below(state)
        pass
    """

    def stuck(self, state):
        if state[self.mario_row][self.mario_row+1] == Learning.OBSTACLE:
            return True
        return False

    def gap_left_far(self, state):
        for i in range(self.mario_col - 1 - self.left_close, -1, -1):
            if self.straighBelow(self.mario_row + 1, i, state):
                return True
        return False

    def gap_right_far(self, state):
        for i in range(self.mario_col + 1 + self.right_close, self.n_col):
            if self.straighBelow(self.mario_row + 1, i, state):
                return True
        return False

    def gap_left_close(self, state):
        for i in range(self.mario_col - 1, self.mario_col - 1 - self.left_close if self.mario_col - 1 - self.left_close >= -1 else -1, -1):
            if self.straighBelow(self.mario_row + 1, i, state):
                return True
        return False

    def gap_right_close(self, state):
        for i in range(self.mario_col + 1, self.n_col if self.mario_col + 1 + self.right_close >=
                                                         self.n_col else self.mario_col + 1 + self.right_close):
            if self.straighBelow(self.mario_row+1, i, state):
                return True
        return False

    def straighBelow(self, row, col, state):
        for i in range(row, self.n_row):
            if state[i][col] == Learning.OBSTACLE:
                return False
        return True

    def gap_below(self, state):
        # straight down
        for i in range(self.mario_row+1, self.n_row):
            if state[i][self.mario_col] == Learning.OBSTACLE:
                return False
        return True

    def obstacle_right_close(self, state):
        # straight right, is there an obstacle?
        for i in range(self.mario_col + 1, self.n_col if self.mario_col + 1 + self.right_close >=
                                                         self.n_col else self.mario_col + 1 + self.right_close):
            if state[self.mario_row][i] == Learning.OBSTACLE:
                return True
        return False

    def obstacle_right_far(self, state):
        for i in range(self.mario_col + 1 + self.right_close, self.n_col):
            if state[self.mario_row][i] == Learning.OBSTACLE:
                return True
        return False

    def goal_close(self, state):
        for i in range(self.mario_col + 1, self.n_col if self.mario_col + 1 + self.right_close >=
                                                         self.n_col else self.mario_col + 1 + self.right_close):
            if state[self.mario_row][i] == Learning.FLAG:
                return True
        return False

    def goal_sight(self, state):
        if self.goal_close(state):
            return True, True
        for row in range(self.n_row):
            for col in range(self.n_col):
                if state[row][col] == Learning.FLAG:
                    return False, True
        return False, False

    def findMario(self, state):
        last_row, last_col = -1, -1
        for row in range(self.n_row):
            for col in range(self.n_col):
                if state[row][col] == Learning.MARIO:
                    # this is not that fast, faster would be to just check straight down, but w/e
                    last_row, last_col = row, col
        return last_row, last_col


    # q learning algorithm
    def qLearn(self):
        pass





if __name__ == '__main__':
    ar1 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    ar2 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    ar3 = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1]]

    feat = Learning(ar1)

    for e in ar1:
        print(e)
    print(feat.createFeatures(ar1, 2))
    for e in ar2:
        print(e)
    print(feat.createFeatures(ar2, 4))
    for e in ar3:
        print(e)
    print(feat.createFeatures(ar3, 1))


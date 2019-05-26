import random
import pickle
from environment import Environment

class Agent:

    def __init__(self, env:Environment, gamma, alpha, eps):
        self.env = env
        self.gamma = gamma      # discount factor
        self.alpha = alpha      # learning rate
        self.eps = eps          # exploration vs exploitation (eps = 0 => all exploitation, eps = 1 => all exploration


        self.Q = {tuple(self.env.state):[0 for action in self.env.A]}

    def loadQ(self):
        file = open('Qtable', 'br')
        self.Q = pickle.load(file)
        file.close()

    def saveQ(self):
        file = open('Qtable', 'bw')
        pickle.dump(self.Q, file)
        file.close()

    def resetGame(self):
        self.env.reset()

        if tuple(self.env.state) not in self.Q.keys():
            self.Q[tuple(self.env.state)] = [0 for action in self.env.A]

    def learnOneEpoch(self):
        def collectReward():
            return self.env.reward

        done = False

        while not done:

            state = tuple(self.env.state)

            if random.uniform(0, 1) < self.eps:
                # Explore
                action = random.choice(self.env.A)
            else:
                # Exploit
                max = self.Q[state][0]
                argmax = set()

                for i in range(len(self.env.A)):
                    if self.Q[state][i] > max:
                        max = self.Q[state][i]
                        argmax = {i}
                    elif self.Q[state][i] == max:
                        argmax.add(i)

                action = random.choice(list(argmax))

            done = self.env.step(action)

            oldState = state
            state = tuple(self.env.state)

            if state not in self.Q.keys():
                self.Q[state] = [0 for action in self.env.A]

            reward = collectReward()
            oldQ = self.Q[oldState][action]

            newMax = self.Q[state][0]
            for i in range(len(self.env.A)):
                if self.Q[state][i] > newMax:
                    newMax = self.Q[state][i]

            # Update the Q value
            self.Q[oldState][action] = ((1 - self.alpha) * oldQ) + (self.alpha * (reward + (self.gamma * newMax)))



    def perform(self):
        eps = self.eps
        # set exploration to 0
        self.eps = 0
        self.learnOneEpoch()
        self.eps = eps

    def getState(self):
        return self.env.state

    def setState(self, state:list):
        self.env.state = state
        if tuple(self.env.state) not in self.Q.keys():
            self.Q[tuple(self.env.state)] = [0 for action in self.env.A]

    def learn(self, N):
        for i in range(1, N+1):
            self.learnOneEpoch()
            self.resetGame()
            print(i)

if __name__ == '__main__':
    env = Environment()
    mario = Agent(env, 0.9, 0.1, 0.3)

    try:
         mario.loadQ()
    except:
        pass

    while True:
        mario.learn(10)
        mario.saveQ()

    mario.perform()


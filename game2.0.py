import os
import time
import numpy as np
import cv2 as cv
import random
from collections import deque
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPool1D, LSTM, Conv2D, MaxPool2D, Dropout, Activation
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.optimizers import Adam,RMSprop
import tensorflow as tf
from debug_tools import tools
timer = tools.timer
GAME_SIZE = [6, 7]
#horizontale reward factor
H_REWARD_FACTOR = 2
V_REWARD_FACTOR = 2
S_REWARD_FACTOR = 2

COLOR_SWITCHER = 100
BLUE = -1
GREEN = 1

EPISODES = 500_000
#random batch uit de replay memory om te trainen
REPLAY_EPISODES = 50_000
MIN_REPLAY_EPISODES = 1_000

BATCH_SIZE = 64

MAX_STEPS = 42
STEP_COST = 10

SAVE_NUMBER = 50
UPDATE_COUNTER = 10
epsilon = 1
EPSILON_DECAY = 0.005
DISCOUNT = 0.99
SHOW_EVERY = 1
PLAYBOARD = cv.cvtColor(cv.imread("Assets/playboard.png", cv.IMREAD_UNCHANGED), cv.COLOR_BGR2BGRA)


class Env:
    def __init__(self):
        self.blue_wins = 0
        self.green_wins = 0
        self.games = 0

        self.total_reward_blue = 0
        self.total_reward_green = 0
        self.game_array = np.zeros(GAME_SIZE)
        self.full_lines = []
        self.episode_end =False

        #1 = green
        #-1 = blue
    def in_list(self,list,thing):
        for element in list:
            if element == thing:
                return True
        return False

    def place(self,move,player):
        if not self.in_list(self.full_lines,move):
            if self.game_array[0][int(move)]:
                self.full_lines.append(int(move))
            for place_y in range(


            ):
                # zoekt juiste plek om blokje te zetten (range moet tot de -1 zijn)
                if self.game_array[len(self.game_array) - place_y - 1][int(move)] == 0:
                    self.game_array[len(self.game_array) - place_y - 1][int(move)] = player.color
                    # van 0 tot game size -1
                    break
        else:
            return False

    def win(self,color):
        print(f"winnn{color}")
        self.winner = color

        if color == BLUE:
            self.total_reward_blue += 1000
            self.total_reward_green -= 250
            self.blue_wins += 1

        if color == GREEN:
            self.total_reward_green += 1000
            self.total_reward_blue -= 250
            self.green_wins += 1

        self.games +=1
        print(game.total_reward_blue, game.total_reward_green)
        self.episode_end =True

    def reward_calc(self,color):

        self.horizental_reward_blue = 0
        self.vertical_reward_blue = 0
        self.schuine_reward_blue = 0

        self.horizental_reward_green = 0
        self.vertical_reward_green = 0
        self.schuine_reward_green = 0
        #loped door alle waarden van de array (matrix) game array en zoekt waarden != 0
        for place_y_index in range(len(self.game_array)):
            self.len_row = 0
            self.prev = 0
            for place_x_index, place_x in enumerate(self.game_array[place_y_index]):
                if place_x != 0:
                    #gaat 4 naar rechts en ziet of het nog steeds hetzelfde was als de eerste waarde
                    for next_block in range(4):
                        if (place_x_index + next_block) > GAME_SIZE[1] - 1 or (place_x_index + next_block) < 0:
                            break
                        elif self.game_array[place_y_index][(place_x_index + next_block)] == place_x:
                            if next_block == (3):
                                self.win(place_x)
                        else:
                            break
                    if place_x ==BLUE:
                        self.horizental_reward_blue += (next_block+1) ** H_REWARD_FACTOR
                        self.horizental_reward_green -= (next_block+1) ** H_REWARD_FACTOR

                    elif place_x ==GREEN:
                        self.horizental_reward_green += (next_block+1) ** H_REWARD_FACTOR
                        self.horizental_reward_blue -= (next_block+1) ** H_REWARD_FACTOR


                self.prev = place_x
        #in een transposed matrix verticaal = horizontaal
        self.game_array_transposed = self.game_array.T
        for place_y_index in range(len(self.game_array_transposed)):
            self.len_row = 0
            self.prev = 0
            for place_x_index, place_x in enumerate(self.game_array_transposed[place_y_index]):
                if place_x != 0:
                    for next_block in range(4):
                        if (place_x_index + next_block) >= GAME_SIZE[1] - 1 or (place_x_index + next_block) < 0:
                            break
                        elif self.game_array_transposed[place_y_index][(place_x_index + next_block)] == place_x:
                            if next_block == (3):
                                self.win(place_x)
                        else:
                            break
                    if place_x == BLUE:
                        self.vertical_reward_blue += (next_block+1) ** V_REWARD_FACTOR
                        self.vertical_reward_green -= (next_block+1) ** V_REWARD_FACTOR

                    elif place_x == GREEN:
                        self.vertical_reward_green += (next_block+1) ** V_REWARD_FACTOR
                        self.vertical_reward_blue -= (next_block+1) ** V_REWARD_FACTOR
                self.prev = place_x

        for place_y_index in range(len(self.game_array)):
            self.len_row = 0
            self.prev = 0
            for place_x_index, place_x in enumerate(self.game_array[place_y_index]):
                if place_x != 0:
                    for next_block in range(4):
                        if (place_x_index - next_block) > GAME_SIZE[1]-1 or (place_x_index - next_block) < 0 or (
                                place_y_index + next_block) > GAME_SIZE[0] - 1 or (place_y_index + next_block) < 0:
                            break
                        elif self.game_array[place_y_index + next_block][(place_x_index - next_block)] == place_x:
                            if next_block == (3):
                                self.win(place_x)
                        else:
                            if place_x == BLUE:
                                self.schuine_reward_blue += (next_block+1) ** S_REWARD_FACTOR
                                self.schuine_reward_green -= (next_block+1) ** S_REWARD_FACTOR

                            elif place_x == GREEN:
                                self.schuine_reward_green += (next_block+1) ** S_REWARD_FACTOR
                                self.schuine_reward_blue -= (next_block+1) ** S_REWARD_FACTOR

                            break

                    for next_block in range(4):
                        if (place_x_index + next_block) > GAME_SIZE[1]-1 or (place_x_index + next_block) <0 or (place_y_index + next_block) >GAME_SIZE[0]-1 or (place_y_index + next_block) < 0:
                            break
                        elif self.game_array[place_y_index + next_block][(place_x_index + next_block)] == place_x:
                            if next_block == (4-1):
                                self.win(place_x)
                        else:
                            if place_x == BLUE:
                                self.schuine_reward_blue += (next_block + 1) ** S_REWARD_FACTOR
                                self.schuine_reward_green -= (next_block + 1) ** S_REWARD_FACTOR
                            elif place_x == GREEN:
                                self.schuine_reward_green += (next_block + 1) ** S_REWARD_FACTOR
                                self.schuine_reward_blue -= (next_block + 1) ** S_REWARD_FACTOR
                            break
        if color == BLUE:
            return self.schuine_reward_blue+self.vertical_reward_blue+self.horizental_reward_blue
        elif color == GREEN:
            return self.schuine_reward_green+self.vertical_reward_green+self.horizental_reward_green


    def reset(self):
        self.game_array = np.zeros(GAME_SIZE)
        self.episode_end = False
        self.full_lines = []
        self.total_reward_blue = 0
        self.total_reward_green = 0

    def step(self, action_blue, action_green):
        if len(self.full_lines) !=7:
            self.place(action_blue, blue)
            self.total_reward_blue += self.reward_calc(blue.color)
        else:
            game.episode_end = True

        if len(self.full_lines) !=7:
            self.place(action_green, green)
            self.total_reward_green += self.reward_calc(green.color)
        else:
            game.episode_end = True

        self.new_state = renderr.render_game(game.game_array)
        renderr.show()
        return self.new_state ,(self.total_reward_blue,self.total_reward_green)


    def legal_move(self,moves):
        if not game.in_list(game.full_lines, np.argmax(moves)):
            move = np.argmax(moves)
            return move
        else:
            if len(self.full_lines) != 7:
                print(moves,np.delete(moves,np.argmax(moves)))
                new_moves = np.delete(moves,np.argmax(moves))
                if len(new_moves)==0:
                    print("________________BUG DELETS ALLES ______________")
                    return random.randint(0,6)
                return self.legal_move(new_moves)

    class render:
        def __init__(self):
            self.playboard = PLAYBOARD

        def green_play(self,loc):
            self.playboard = cv.circle(self.playboard, (loc[0], loc[1]), 48, (174, 215, 149), -1)

        def blue_play(self,loc):
            self.playboard = cv.circle(self.playboard, (loc[0], loc[1]), 48, (248, 200, 127), -1)

        def get_played_locs(self,game_array):
            self.loc_blue = []
            self.loc_green = []
            for self.place_y in range(len(game_array)):
                for self.place_x in range(len(game_array[self.place_y])):
                    self.current_loc = game_array[self.place_y][self.place_x]
                    if self.current_loc == BLUE:
                        self.loc_blue.append((self.place_x*100 + 50,self.place_y*100 +50))
                    elif self.current_loc == GREEN:
                        self.loc_green.append((self.place_x*100 + 50,self.place_y*100 +50))

        def reset(self):
            self.playboard = cv.cvtColor(cv.imread("Assets/playboard.png", cv.IMREAD_UNCHANGED), cv.COLOR_BGR2BGRA)

            self.total_reward_green = 0
            self.total_reward_blue = 0

            self.loc_blue = []
            self.loc_green = []

        def render_game(self,game_array):
            self.reset()
            self.get_played_locs(game_array)
            loc = ()
            for loc in self.loc_blue:
                self.blue_play(loc)
            loc = ()
            for loc in self.loc_green:
                self.green_play(loc)
            return cv.cvtColor(cv.resize(self.playboard,(70,60)),cv.COLOR_BGRA2BGR)

        def show(self):
            cv.imshow("render",  self.playboard)
            cv.waitKey(1)




class DQN_agent:
    def __init__(self,color,color_string, IMG=True):
        self.color = color

        self.color_string = color_string
        self.save_counter = 0
        self.update_counter = 0
        # hoeveelheid episodes in memory
        self.memory = deque(maxlen=REPLAY_EPISODES)
        self.current_reward = 0
        #van turtorial
        self.tensorboard = self.ModifiedTensorBoard(log_dir=f"logs/{self.color_string}")

        if IMG:
            self.mainmodel = self.create_model_img()
            self.target_model = self.create_model_img()
            if self.color == "blue":
                self.mainmodel.load_weights(r'C:\Users\cyuzuzo\PycharmProjects\vier op een rij\model blue 10 uur')
            elif self.color == "green":
                pass
                #self.mainmodel.load_weights(  r'')
        else:
            self.mainmodel = self.create_model_array()
            self.target_model = self.create_model_array()

        self.weights = self.mainmodel.get_weights()
        self.target_model.set_weights(self.weights)

    class ModifiedTensorBoard(TensorBoard):

        # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.step = 1
            self.writer = tf.summary.create_file_writer(self.log_dir)
            self._log_write_dir = os.path.join(self.log_dir, f"-{int(time.time())}")

        # Overriding this method to stop creating default log writer
        def set_model(self, model):
            pass

        # Overrided, saves logs with our step number
        # (otherwise every .fit() will start writing from 0th step)
        def on_epoch_end(self, epoch, logs=None):
            self.update_stats(**logs)

        # Overrided
        # We train for one batch only, no need to save anything at epoch end
        def on_batch_end(self, batch, logs=None):
            pass

        # Overrided, so won't close writer
        def on_train_end(self, _):
            pass

        def on_train_batch_end(self, batch, logs=None):
            pass

        # Custom method for saving own metrics
        # Creates writer, writes custom metrics and closes writer
        def update_stats(self, **stats):
            self._write_logs(stats, self.step)

        # Added because of version
        def _write_logs(self, logs, index):
            with self.writer.as_default():
                for name, value in logs.items():
                    tf.summary.scalar(name, value, step=index)
                    self.step += 1
                    self.writer.flush()


    def create_model_array(self):
        self.model = Sequential()

        self.model.add(Conv1D(6,3, batch_input_shape=[None,6,7]))
        self.model.add(MaxPool1D(3))

        self.model.add(Dense(6*7,batch_input_shape=[None,6,7]))

        self.model.add(Dense((6*7)/2, activation="sigmoid"))
        self.model.add(Dense((6*7)/3, activation="sigmoid"))

        self.model.add(Flatten())
        self.model.add(Dense(7, activation="linear"))
        self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        self.model.summary()
        return self.model

    def create_model_img(self):
        self.model = Sequential()

        self.model.add(Conv2D(256,(3,3), input_shape=(60,70,3)))
        self.model.add(Activation("relu"))
        self.model.add(MaxPool2D((2,2)))
        self.model.add(Dropout(0.2))

        self.model.add(Conv2D(256, (3, 3)))
        self.model.add(Activation("relu"))
        self.model.add(MaxPool2D((3, 3)))
        self.model.add(Dropout(0.2))

        #self.model.add(LSTM(128, activation="sigmoid"))
        self.model.add(Flatten())
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(7, activation="linear"))

        self.model.compile(loss='mse', optimizer=Adam(lr=0.00025), metrics=['accuracy'])
        self.model.summary()
        return self.model

    def update_replay_memory(self,state):
        self.memory.append(state)

    def get_qs_img(self,state):
        return self.mainmodel.predict(np.array(state).reshape(-1, *state.shape)/255)[0]

    def train(self,terminal_state):
        #dqn implementation gebaseerd op https://www.youtube.com/watch?v=qfovbG84EBg&t=336s
        if len(self.memory) < MIN_REPLAY_EPISODES:
            print(f'len self.memory{len(self.memory)}')
            return
        else:
            #begin = time.time()
            self.minibatch = random.sample(self.memory, BATCH_SIZE)

            #sampling = time.time()
            #krijgt de game array of img van elke play
            self.current_states = np.array([transition[0] for transition in self.minibatch])/255# die nul is de currentstate
            self.current_qs_list = self.mainmodel.predict(self.current_states)

            self.new_states = np.array([transition[3] for transition in self.minibatch])/255# die drie is de new_state
            self.future_qs = self.target_model.predict(self.new_states)
            #predictingqs = time.time()

            #imgs
            self.x = []
            #moves
            self.y = []
            #eerst  predicte we de current en future qs om die hierna in de for loop aan
            for self.index, (self.current_state, self.action, self.reward , self.new_state, self.done) in enumerate(self.minibatch):
                if not self.done:
                    self.max_future_q = np.max(self.future_qs[self.index])
                    self.new_q = self.reward + DISCOUNT * self.max_future_q
                else:
                    self.new_q = self.reward

                self.current_qs = self.current_qs_list[self.index]
                self.current_qs[self.action] = self.new_q

                self.x.append(self.current_state)
                self.y.append(self.current_qs)
            #prearingdata = time.time()

            self.mainmodel.fit(np.array(self.x)/255,np.array(self.y), batch_size=BATCH_SIZE, verbose=0, shuffle=False,callbacks=[self.tensorboard] if terminal_state else None)
            #fitting = time.time()

            if terminal_state:
                self.save_counter += 1
                self.update_counter += 1

            if self.update_counter > UPDATE_COUNTER:
                self.target_model.set_weights(self.mainmodel.get_weights())
                self.update_counter = 0

            if SAVE_NUMBER <= self.save_counter:
                self.save_counter = 0
                self.mainmodel.save("model.{}.{}".format(time.time(),self.color_string))
            #rest = time.time()

            #print(f"timing sampling took{sampling - begin},predictingqs took {predictingqs - sampling},prearingdata took {prearingdata - predictingqs},fitting:{fitting - prearingdata}whole:{rest-begin} ")


blue = DQN_agent(BLUE,"blue")
green = DQN_agent(GREEN,'green')
game = Env()
renderr = game.render()
show = False
curent_state = renderr.render_game(game.game_array)
highest_reward_blue = 0
highest_reward_green = 0
minus_epsilon = True

show = 0
showing = 0
max_reward_blue = 0
max_reward_green = 0
min_reward_blue = 0
min_reward_green = 0
episode_reward_blue = []
episode_reward_green = []
games = 0
color_switcher_Counter = 0
agents_list = [blue,green]
for episode in range(EPISODES):
    game.reset()
    show +=1
    color_switcher_Counter+=1
    print(agents_list[0].color_string, agents_list)
    agents_list.reverse()
    for step in range(MAX_STEPS):
        begin = time.time()
        if show >=SHOW_EVERY:
            renderr.show()
            showing +=1
            if showing >=20:
                showing=0
                show =0
        rendering = time.time()

        #curent_state = game.game_array
        print(epsilon)
        if np.random.random() > epsilon:
            print("ai")

            color_1_action = agents_list[0].target_model.predict(np.array(curent_state).reshape(-1, *curent_state.shape) / 255)[0]
            color_2_action = agents_list[1].target_model.predict(np.array(curent_state).reshape(-1, *curent_state.shape) / 255)[0]
            color_1_action = game.legal_move(color_1_action)
            color_2_action = game.legal_move(color_2_action)
            if agents_list[0].color == BLUE:
                blue_action = color_1_action
                green_action = color_2_action
            else:
                green_action = color_1_action
                blue_action = color_2_action
        else:
            print("epsilon")

            color_1_action = [random.randint(0,6),random.randint(0,6),random.randint(0,6),random.randint(0,6),random.randint(0,6),random.randint(0,6)]
            print("color_1:",color_1_action)
            color_1_action = game.legal_move(color_1_action)
            print("color_1_l:",color_1_action)

            color_2_action = [random.randint(0,6),random.randint(0,6),random.randint(0,6),random.randint(0,6),random.randint(0,6),random.randint(0,6)]
            print("color_2:",color_1_action)
            color_2_action = game.legal_move(color_2_action)
            print("color_2_l:",color_1_action)

            if agents_list[0].color == BLUE:
                blue_action = color_1_action
                green_action = color_2_action
            else:
                green_action = color_1_action
                blue_action = color_2_action




        move = time.time()

        if not game.episode_end:
            new_state, reward_both = game.step(blue_action,green_action)
        else:
            if game.winner == BLUE:
                reward_both[0] += step*STEP_COST
                reward_both[1] -= step*STEP_COST
            elif game.winner == GREEN:
                reward_both[0] -= step * STEP_COST
                reward_both[1] += step * STEP_COST

        blue.update_replay_memory((curent_state, blue_action, int(reward_both[0]), new_state, game.episode_end))
        green.update_replay_memory((curent_state, green_action, int(reward_both[1]), new_state, game.episode_end))
        memory = time.time()

        if episode >= 100:
            if COLOR_SWITCHER <= color_switcher_Counter:
               print(color_switcher_Counter)
               blue.train(game.episode_end)
               if 2*COLOR_SWITCHER <= color_switcher_Counter:
                    color_switcher_Counter = 0
            else:
                print(color_switcher_Counter)

                green.train(game.episode_end)
        else:
            green.train(game.episode_end)

        train = time.time()
        if epsilon > 0 and minus_epsilon:
            epsilon -=  EPSILON_DECAY
        else:
            minus_epsilon = False

        if epsilon < 1 and not minus_epsilon:
            epsilon +=  EPSILON_DECAY
        else:
            minus_epsilon = True



        curent_state = new_state
        epislon = time.time()
        if int(reward_both[0]) >= max_reward_blue:
            max_reward_blue = reward_both[0]
        if int(reward_both[0] ) <= min_reward_blue:
            min_reward_blue = reward_both[0]

        if reward_both[1] >= max_reward_green:
            max_reward_green = int(reward_both[1])
        if reward_both[1] - STEP_COST * step <= min_reward_green:
            min_reward_green = int(reward_both[1])
        episode_reward_blue.append(int(reward_both[0]))
        episode_reward_green.append(int(reward_both[1]))

        if game.episode_end:
            game.winner= 0
            win_ratio_blue = game.blue_wins/game.games
            print(win_ratio_blue,game.blue_wins,game.games)
            win_ratio_green = game.green_wins/game.games

            if win_ratio_green >= 0.9:
                blue.mainmodel.set_weights(green.mainmodel.get_weights())
            elif win_ratio_blue >= 0.9:
                green.mainmodel.set_weights(blue.mainmodel.get_weights())

            avg_green = sum(episode_reward_green)/len(episode_reward_green)
            avg_blue = sum(episode_reward_blue)/len(episode_reward_blue)

            blue.tensorboard.update_stats(reward_avg=avg_blue, reward_min_blue=min_reward_blue, reward_max_blue=max_reward_blue,  epsilon=epsilon,step=step, win_ratio_blue=win_ratio_blue, episode=episode)
            green.tensorboard.update_stats(reward_avg=avg_green, reward_min_green=min_reward_green, reward_max_green=max_reward_green,  epsilon=epsilon,step=step, win_ratio_green=win_ratio_green, episode=episode)

            max_reward_blue = 0
            max_reward_green = 0
            min_reward_blue = 0
            min_reward_green = 0
            episode_reward_blue = []
            episode_reward_green = []

            print(step)
            print(f"timing rendeering took{rendering - begin},move took {move - rendering},memory took {memory - move},train:{train - memory}epsilon:{epislon - train} ")
            break





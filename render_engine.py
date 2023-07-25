import cv2 as cv
import numpy as np

PLAYBOARD = cv.cvtColor(cv.imread("Assets/playboard.png", cv.IMREAD_UNCHANGED), cv.COLOR_BGR2BGRA)
BLUE = 1
GREEN = -1

class render_engine:

    def __init__(self):
        self.playboard = PLAYBOARD

    def _green_render(self, loc):
        self.playboard = cv.circle(self.playboard, (loc[0], loc[1]), 48, (174, 215, 149), -1)

    def _blue_render(self, loc):
        self.playboard = cv.circle(self.playboard, (loc[0], loc[1]), 48, (248, 200, 127), -1)

    def _update_lopcations(self, game_array): #array in collum mayor and bottom to top
        self.loc_blue = []
        self.loc_green = []
        for cord in np.concatenate(( np.array(np.where(game_array == BLUE)).T, np.array(np.where(game_array == GREEN)).T)):
            current_value = game_array[cord[0]][cord[1]]
            cordinates_render = (cord[1] * 100 + 50, cord[0] * 100 + 50)
            self.loc_blue.append(cordinates_render) if current_value == BLUE else self.loc_green.append(cordinates_render)

    def show(self,game_array):
        self.playboard = PLAYBOARD
        self._update_lopcations(game_array)
        for loc in self.loc_blue:
            self._blue_render(loc)
        for loc in self.loc_green:
            self._green_render(loc)
        cv.imshow("render", self.playboard)
        cv.waitKey(1)

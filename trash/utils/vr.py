
OCCULUS = '/data/pulkitag/misc/marcel/huge-dev/human-guided-exploration/oculus_reader'
import sys
import time

import numpy as np
sys.path.append(OCCULUS)
from oculus_reader.reader import OculusReader


class OculusController():
    def __init__(self):
        self.oculus = OculusReader()
        time.sleep(1)
        self.cur_pos = self.get_cur_pos()

    def reset(self):
        """
        Call this method to reset the start position for the deltas call.
        """
        self.cur_pos = self.get_cur_pos()

    def get_cur_pos(self):
        """
        Internal method, this returns the positon of the right controller
        """
        # print(self.oculus.get_transformations_and_buttons()[0])
        transform = self.oculus.get_transformations_and_buttons()[0]['r']

        return transform[:, 3]


    def get_buttons(self):
        """
        Returns the buttons.
        The ones on the right are "A, B, RTr"
        """
        return self.oculus.get_transformations_and_buttons()[1]

    def get_deltas(self):
        """
        This method returns the deltas of the vr controller since the last time this or 
        reset method has been called. It returns them of the form x, y, z in centimeters
        where positive x is closer to the headset, positive y is left of the headset, 
        and positive z is upwards
        """
        new_pose = self.get_cur_pos()
        deltas = new_pose - self.cur_pos
        self.cur_pos = new_pose

        final_deltas = [0,0,0]
        final_deltas[2] = deltas[1]
        final_deltas[1] = deltas[0] 
        final_deltas[0] = deltas[2] 
        final_deltas = np.array(final_deltas) * 100
        for i , delta in enumerate(final_deltas):
            final_deltas[i] = delta if delta > .5 or delta < -.5 else 0
        return final_deltas

def main():
    reader = OculusController()
    print(reader.get_buttons())
    while True:
        pos = reader.get_deltas()
        # print(f'X delta= {pos[0]},      Y Delta = {pos[1]},     Z Delta = {pos[2]}')
        press = reader.get_buttons()
        print(press)
        print(f"A: {press['A']}, B: {press['B']}, RTrig: {press['RTr']}")
        print(not press['A'])
        time.sleep(.5)


if __name__ == '__main__':
     main()
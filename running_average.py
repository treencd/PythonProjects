import numpy as np
from collections import deque

max_measures = 5
first_new = False
new_right_xcoord = 11
right_xcoords = deque([7.91, 7.81, 7.58, 6.55, 6.00])
# right_xcoords =[]

if len(right_xcoords) == max_measures:
    moving_avgs = np.convolve(right_xcoords, np.ones((3,))/3, mode='valid')
    if abs(new_right_xcoord - moving_avgs[2])<= 4:
        right_xcoords.append(new_right_xcoord)
        old_right_xcoord = new_right_xcoord
        first_new = True
    else:
        if first_new:
            right_xcoords.append(new_right_xcoord)
        else:
            slope1 = moving_avgs[1]-moving_avgs[0]
            slope2 = moving_avgs[2] - moving_avgs[1]
            avg_change = (slope1+slope1)/2
            new_right_xcoord = right_xcoords[4] + avg_change
            right_xcoords.append(new_right_xcoord)
            print()
    right_xcoords.popleft()


else:
    right_xcoords.append(new_right_xcoord)

print(right_xcoords)


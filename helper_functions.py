import numpy as np

def get_p(mg, x, y, z):
    return np.array([mg[0, x, y], mg[1, x, y], z])


def get_d_vector(coordinates, room_center=(3, 3)):
    x, y = np.abs(coordinates[0] - room_center[0]), np.abs(coordinates[1] - room_center[1])
    if y > x:
        return np.array([1, 0, 0])
    else:
        return np.array([0, 1, 0])


def get_mic_list(r):
    mic_positions = r.mic_array.R
    mic_list = [tuple(mic_positions[:, i]) for i in range(mic_positions.shape[1])]
    return mic_list

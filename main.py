import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve, stft, istft
# import IPython
import pyroomacoustics as pra
import os
# from IPython.display import Audio
import librosa
from helper_functions import *

WITH_PLOTS = False

SPEED_OF_SOUND = 343  # m/s
# load sources
# source_file_path1 = r"E:\TIMIT\sample_dataset\Train\FCJF0\SX307.wav"
# source_file_path2 = r"E:\TIMIT\sample_dataset\Train\FCJF0\SX217.wav"
source_file_path1 = r"sample_data\SX307.wav"
source_file_path2 = r"sample_data\SX217.wav"

x1, fs = librosa.load(source_file_path1, sr=None)
x2, fs2 = librosa.load(source_file_path2, sr=None)

# Define room dimensions
room_dim = [6, 6, 6.1]
T60 = 0.4
HEIGHT = 1.0
# Define absorption for walls to achieve T60 of 0.4 seconds
absorption, max_order = pra.inverse_sabine(rt60=T60, room_dim=room_dim)

# Create the room
room = pra.ShoeBox(room_dim, fs=16000, materials=pra.Material(absorption), max_order=max_order)

# add speakers
speaker1_position = [2, 2, HEIGHT]
speaker2_position = [4, 3, HEIGHT]
room.add_source(position=speaker1_position, signal=x1)
room.add_source(position=speaker2_position, signal=x2)

# Display the room parameters
print(f"Room dimensions: {room_dim}")
print(f"Absorption coefficient: {absorption}")
print(f"Max order: {max_order}")

# plot the room
if WITH_PLOTS:
    room.plot(img_order=0)

# add microphones

array_distance = 0.2
mic_height = HEIGHT
mic_locations = [(2, 1), (2.9, 1), (4, 1), (1, 2), (5, 2), (1, 2.9), (5, 2.9), (1, 3.8), (5, 3.8), (2, 5), (2.9, 5),
                 (4, 5)]
# mic_positions_list = []
m1 = []
m2 = []
for mic_loc in mic_locations:
    mic_array_location = np.array([mic_loc[0], mic_loc[1], mic_height])
    mic_1_pos = mic_array_location - (get_d_vector(mic_loc) * (array_distance / 2))
    mic_2_pos = mic_array_location + (get_d_vector(mic_loc) * (array_distance / 2))
    mic_positions = np.c_[mic_1_pos, mic_2_pos]
    # mic_positions_list.append(mic_positions)
    # mic_positions_list.extend([mic_1_pos, mic_2_pos])
    m1.append(mic_1_pos[:-1])
    m2.append(mic_2_pos[:-1])
    room.add_microphone_array(pra.MicrophoneArray(mic_positions, room.fs))

print("starting simulation")
room.simulate()

# Z_pairs.shape = 12 X 2 X f X t

f, t, Z = stft(room.mic_array.signals, fs, nfft=1024)
freq_range = (500, 1500)
Z_ranged = Z[:, np.logical_and(freq_range[0] < f, f < freq_range[1]), :]
f_ranged = f[np.logical_and(freq_range[0] < f, f < freq_range[1])]
K = len(f_ranged)

Z_pairs = np.reshape(Z_ranged, [Z_ranged.shape[0] // 2, 2, Z_ranged.shape[1], Z_ranged.shape[2]])

# prp = Z_pairs[:,1,:,:]**2 / Z_pairs[:,0,:,:]**2 * np.abs(Z_pairs[:,0,:,:]) / Z_pairs[:,1,:,:]

prp = np.exp(1j * (np.angle(Z_pairs[:, 0, :, :]) - np.angle(Z_pairs[:, 1, :, :])))  # PRP data ||M|| x K x T

# M = np.array(mic_positions_list)
# M = M[:, :-1]

pm1 = np.array(m1)  # |M| x 2
pm2 = np.array(m2)  # |M| x 2

grid_resolution = 60
room_2d = room_dim[:-1]
P = np.mgrid[0:room_2d[0]:grid_resolution * 1j, 0:room_2d[1]:grid_resolution * 1j]

P_vector = np.reshape(P, [2, -1])  # 2 x ||P||

#
# # todo without loops
# for p in P_vector.T:
#     for m in m1:
#         np.linalg.norm(m - p)

M1_expanded = pm1[:, :, np.newaxis]  # shape (|M| x 2 x 1)
M2_expanded = pm2[:, :, np.newaxis]  # shape (|M| x 2 x 1)
P_expanded = P_vector[np.newaxis, :, :]  # shape (1 x 2 x |P|)

d1 = M1_expanded - P_expanded  # shape |M| x 2 x |P|
d2 = M2_expanded - P_expanded  # shape |M| x 2 x |P|
distances1 = np.linalg.norm(d1, axis=1)
distances2 = np.linalg.norm(d2, axis=1)

# Compute the distance
path_difference = distances2 - distances1  # shape |M| x |P|

eprp_angle = -1j * 2 * np.pi * (1 / K) * fs * (1 / SPEED_OF_SOUND) * path_difference

expected_prp = np.exp(
    f_ranged[:, np.newaxis, np.newaxis] * np.repeat(eprp_angle[np.newaxis, :, :], K, axis=0))  # shape |K| * |M| * |P|
eprp = np.transpose(expected_prp, [1, 2, 0])[:, :, :, np.newaxis]  # shape |M| * |P| * |K| * 1
# prp - PRP data |M| x |K| x |T|
readings_term_diff = prp[:, np.newaxis, :, :] - eprp
# readings_term = readings_term_diff * np.conjugate(readings_term_diff)  # shape |M| x |P| x |T| X |K|

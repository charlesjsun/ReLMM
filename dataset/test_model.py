import numpy as np
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.neighbors import KNeighborsRegressor

rewards = np.load('rewards.npy')
actions = np.load('actions.npy')[rewards == 1]
states = np.load('pos.npy')[:,:2][rewards == 1]

model = KNeighborsRegressor().fit(states, actions)
predicted = model.predict(states)

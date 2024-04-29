# Code source: https://github.com/artur-deluca/landscapeviz
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import landscapeviz
import cocoex as ex
import numpy as np


if __name__ == "__main__":
	# 1. define model
	model = tf.keras.Sequential([
		tf.keras.Input(shape=(2,)),
		tf.keras.layers.Dense(50, activation=tf.nn.relu),
		tf.keras.layers.Dense(20, activation=tf.nn.relu),
		tf.keras.layers.Dense(1)
	])
	optimizer = tf.keras.optimizers.SGD(learning_rate=0.00005)
	model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

	# 2. get data
	suite = ex.Suite(suite_name="bbob", suite_instance='', suite_options='')
	problem = suite.get_problem_by_function_dimension_instance(function=3, dimension=2, instance=1)
	data_size = 5000
	x = []
	y = []
	for i in range(data_size):
		input = np.random.uniform(low=-5, high=5, size=2)
		output = problem(input)
		x.append(input)
		y.append(output)
	X_train, y_train = np.array(x), np.array(y)
	scaler_x = MinMaxScaler(feature_range=(-1,+1)).fit(X_train)
	X_train = scaler_x.transform(X_train)

	# 3. train model
	model.fit(X_train, y_train, batch_size=32, epochs=200, verbose=1)

	# 4. build mesh and plot
	landscapeviz.build_mesh(model, (X_train, y_train), grid_length=100, verbose=1)
	landscapeviz.plot_contour(key="mean_squared_error")
	landscapeviz.plot_3d(key="mean_squared_error")
	
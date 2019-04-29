import keras
import time

class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


"""
model.fit(train_X, train_y, batch_size=512, epochs=2, validation_data=(val_X, val_y), callbacks=[time_callback])
times = time_callback.times
for i, time_cost in enumerate(times):
    print('Epoch %s time cost: %.2f s' % (i, time_cost))
"""
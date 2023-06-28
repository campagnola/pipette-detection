import os, gc, argparse, json
import numpy as np
import scipy.stats
from training_data import TrainingData, Normalizer

parser = argparse.ArgumentParser()
parser.add_argument('--training-data', type=str, help="path to training data")
parser.add_argument('--save-path', type=str, default=None, help="path to save model, weights, history, and configuration")
parser.add_argument('--load-model', type=str, default=None, help="path to load model")
parser.add_argument('--load-weights', type=str, default=None, help="path to load weights")
parser.add_argument('--model-type', type=str, default='ResNet101V2', help="Base model type (ResNet101V2, ResNet50, VGG16, VGG19, ...)")
parser.add_argument('--learning-rate', type=float, default=None, help='learning rate (0.001 is default)')
parser.add_argument('--train-depth', type=float, default=1.0, help='base model depth at which to begin training (0.0 means train all; 1.0 means train none)')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--allow-cpu', type=bool, default=False, help='Allow training without GPU (otherwise, exit if no GPU is available)')
args = parser.parse_args()

if os.path.exists(args.save_path):
    raise Exception(f"Save path {args.save_path} already exists")


from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if not args.allow_cpu and len(gpus) == 0:
    raise Exception("Exiting; no GPU available (use --allow-cpu to override)")


def below_threshold(y_true, y_pred, threshold=0.1):
    kb = keras.backend
    diff = kb.abs(y_true - y_pred)
    return kb.mean(kb.cast(kb.all(diff < threshold, axis=-1), 'float32'))
    

class PeriodicValidation(keras.callbacks.Callback):
    def __init__(self, n_iter, history_file, training_data, validation_data, threshold=0.001, smoothing=0.9):
        self._n_iter = n_iter
        self._training_data = training_data
        self._validation_data = validation_data[:]
        assert self._validation_data is not None
        self.history_file = history_file
        self.history = {'batch': [], 'train_mse': [], 'val_mse': []}
        self.threshold = threshold
        self.smoothing = smoothing
        self.train_prev_mse = None
        self.val_prev_mse = None
        keras.callbacks.Callback.__init__(self)
        
    def compute_mse(self, actual_pos, predicted_pos):
        full_mse = np.mean(np.square(actual_pos - predicted_pos))
        xy_mse = np.mean(np.square(actual_pos[:, 1:] - predicted_pos[:, 1:]))
        z_mse = np.mean(np.square(actual_pos[:, 0] - predicted_pos[:, 0]))
        return [full_mse, xy_mse, z_mse]
    
    def on_train_batch_end(self, batch, logs=None):
        if batch % self._n_iter == 0:
            self.run_validation(batch)
            self.save_history()

    def run_validation(self, batch):
        print(f"\nBatch {batch}")
        self.history['batch'].append(batch)

        if self._training_data.last_batch is not None:
            train_images, train_pos = self._training_data.last_batch        
            pred_train_pos = self.model.predict(train_images, verbose=0)
            train_mse = self.compute_mse(train_pos, pred_train_pos)
            self.history['train_mse'].append(train_mse)
            train_slopes = self.get_slopes('train_mse')
            print(f"    Training MSE (xyz, xy, z): {train_mse}  Slopes: {train_slopes}")
        else:
            self.history['train_mse'].append(None)
        
        val_images, val_pos = self._validation_data
        pred_val_pos = self.model.predict(val_images, verbose=0)
        val_mse = self.compute_mse(val_pos, pred_val_pos)
        self.history['val_mse'].append(val_mse)
        val_slopes = self.get_slopes('val_mse')
        print(f"  Validation MSE (xyz, xy, z): {val_mse}  Slopes: {val_slopes}")

        if val_slopes is not None and np.all(np.abs(val_slopes) < self.threshold):
            print(f"Validation loss slopes crossed threshold: {val_slopes}; terminating training early")
            self.model.stop_training = True
        
        gc.collect() 
        keras.backend.clear_session()

    def get_slopes(self, key, size=5):
        x = np.array(self.history['batch'][-size:])
        y = np.vstack(self.history[key][-size:])
        if len(x) < 2:
            return None
        return [np.polyfit(x, y[:,i], deg=1)[0] for i in range(y.shape[1])]

    def save_history(self):
        if self.history_file is not None:
            with open(self.history_file, 'a') as fh:
                hist = {k:(v.tolist() if isinstance(v, np.ndarray) else v) for k,v in self.history.items()}
                json.dump(hist, fh)


class PeriodicModelSave(keras.callbacks.Callback):
    def __init__(self, n_iter, filename):
        self._n_iter = n_iter
        self.filename = filename
        keras.callbacks.Callback.__init__(self)
        
    def on_train_batch_end(self, batch, logs=None):
        if batch % self._n_iter == 0:
            print(f"saving weigts to {self.filename}")
            self.model.save_weights(self.filename)


class PipetteDetectionModel:
    def __init__(self, model_opts=None, load_model=None):
        self.model_opts = {'model_file': load_model} if model_opts is None else model_opts
        if load_model is not None:
            self.model = keras.models.load_model(args.load_model)
        else:
            self.model = self.create_model(**({} if model_opts is None else model_opts))
        print(self.model.summary())

    @staticmethod
    def create_model(model_type, pooling_layer=True, flatten_layer=False, dense_layers=None):
        base_model = getattr(keras.applications, model_type)(weights='imagenet', include_top=False, input_shape=input_shape)
        base_model.trainable = False

        layers = [
            keras.Input(shape=input_shape),
            base_model,
        ]
        if pooling_layer:
            layers.append(keras.layers.GlobalAveragePooling2D())
        if flatten_layer:
            layers.append(keras.layers.Flatten())
        if dense_layers is not None:
            for size in dense_layers:
                layers.append(keras.layers.Dense(size, activation='relu'))

        layers.append(keras.layers.Dense(3))
        model = keras.models.Sequential(layers)

        return model

    def load_weights(self, weight_file):
        self.model.load_weights(weight_file)

    def save_weights(self, weight_file):
        self.model.save_weights(weight_file)

    def save_model(self, model_file):
        self.model.save_model(model_file)

    def fit(self, training_data, validation_data, train_depth, learning_rate=None, batch_size=64, epochs=1, save_path=None, val_interval=100, save_interval=1000):
        if save_path is not None:
            assert not os.path.exists(save_path), f"Save path {save_path} already exists"
            os.makedirs(save_path)

        fit_opts = {
            'train_depth': train_depth, 
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'epochs': epochs,
        }

        with open(os.path.join(save_path, 'model_options.json'), 'w') as fh:
            json.dump(self.model_opts, fh)
        with open(os.path.join(save_path, 'fit_options.json'), 'w') as fh:
            json.dump(fit_opts, fh)
        
        base_model = self.model.layers[0]
        for i,layer in enumerate(base_model.layers):
            layer.trainable = i > len(base_model.layers) * train_depth

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
            loss='mse',
        )

        self.validator = PeriodicValidation(
            n_iter=val_interval, 
            history_file=os.path.join(save_path, 'training_history.json'),
            training_data=training_data,
            validation_data=validation_data,
        )
        callbacks = [self.validator]
        weights_path = os.path.join(save_path, 'fit_weights')
        if save_path is not None:
            callbacks.append(PeriodicModelSave(save_interval, weights_path))

        try:
            self.model.fit(
                training_data.generator(batch_size=batch_size), 
                steps_per_epoch=len(training_data)//batch_size, 
                epochs=epochs, 
                batch_size=batch_size,
                callbacks=callbacks,
            )
        finally:
            if save_path is not None:
                self.model.save_weights(weights_path)
                self.model.save(os.path.join(save_path, 'fit_model'))


norm = Normalizer(range=[[-100, 0, 0], [100, 500, 500]])
all_data = TrainingData(args.training_data, output_norm=norm.normalize)
training_data, validation_data = all_data.split([0.995, 0.005])
print(f"Loaded {len(training_data)} training examples and {len(validation_data)} validation examples")

batch_size = args.batch_size
input_shape = training_data[0][0].shape


if args.load_model is not None:
    model = PipetteDetectionModel(load_model=args.load_model)
else:
    model = PipetteDetectionModel(model_opts={'model_type': args.model_type})


model.fit(
    training_data, 
    validation_data, 
    train_depth=args.train_depth, 
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    epochs=args.epochs,
    save_path=args.save_path,
    val_interval=100,
    save_interval=1000,
)

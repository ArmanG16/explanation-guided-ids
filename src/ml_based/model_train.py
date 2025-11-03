import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

def train_mlp(input_path, output_model_path):
    df = pd.read_csv(input_path)
    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    model.save(output_model_path)
    pd.DataFrame(history.history).to_csv('training_history.csv', index=False)

if __name__ == "__main__":
    train_mlp('data/preprocessed_dataset.csv', 'models/mlp_model.h5')

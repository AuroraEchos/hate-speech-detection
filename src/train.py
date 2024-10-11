# Description: This file contains the training script for a text classification model using LSTM and pre-trained embeddings.
# Date: 2024-10-12
# Name: Wenhao Liu

# Source code: https://www.kaggle.com/code/thousandvoices/simple-lstm/script

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras._tf_keras.keras.layers import LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.preprocessing import text, sequence
from keras._tf_keras.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras._tf_keras.keras.layers import MultiHeadAttention, LayerNormalization
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tqdm.pandas()

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))

def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in word_index.items():
        embedding_matrix[i] = embedding_index.get(word, np.zeros(300))
    return embedding_matrix

def build_model(embedding_matrix, num_aux_targets, lstm_units, dense_hidden_units, max_len):
    words = Input(shape=(max_len,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
    x = SpatialDropout1D(0.25)(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)

    attention_output = MultiHeadAttention(num_heads=8, key_dim=lstm_units)(x, x)
    x = LayerNormalization()(x + attention_output)

    hidden = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)])
    hidden_dense_1 = Dense(dense_hidden_units, activation='tanh')(hidden)
    hidden_dense_2 = Dense(dense_hidden_units, activation='relu')(hidden_dense_1)
    
    result = Dense(1, activation='sigmoid', name='main_output')(hidden_dense_2)
    aux_result = Dense(num_aux_targets, activation='sigmoid', name='aux_output')(hidden_dense_2)
    
    model = Model(inputs=words, outputs=[result, aux_result])
    model.compile(loss='binary_crossentropy', optimizer=Adam(clipnorm=0.1), metrics=['accuracy', 'accuracy'])
    return model

def save_tokenizer(tokenizer, filename='tokenizer.json'):
    tokenizer_json = tokenizer.to_json()
    with open(filename, 'w') as f:
        f.write(tokenizer_json)

def save_training_progress(training_data, filename='training_progress.json'):
    with open(filename, 'w') as f:
        json.dump(training_data, f, indent=4)

def save_evaluation_report(report, auc, roc_curve_data, filename='evaluation_report.json'):
    with open(filename, 'w') as f:
        json.dump({
            'classification_report': report,
            'auc': auc,
            'roc_curve': roc_curve_data
        }, f, indent=4)

def main(args):
    EMBEDDING_FILES = [args.crawl_embedding_file, args.glove_embedding_file]

    data = pd.read_csv(args.train_file)
    train, valid = train_test_split(data, test_size=0.2, random_state=42)
    
    x_train = train['comment_text'].fillna('').values
    y_train = np.where(train['target'] >= 0.5, 1, 0)
    y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]

    x_valid = valid['comment_text'].fillna('').values
    y_valid = np.where(valid['target'] >= 0.5, 1, 0)
    y_aux_valid = valid[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
    
    tokenizer = text.Tokenizer(num_words=args.max_features)
    tokenizer.fit_on_texts(list(x_train) + list(x_valid))

    x_train = sequence.pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=args.max_len)
    x_valid = sequence.pad_sequences(tokenizer.texts_to_sequences(x_valid), maxlen=args.max_len)

    save_tokenizer(tokenizer)

    embedding_matrix = np.concatenate([build_matrix(tokenizer.word_index, f) for f in EMBEDDING_FILES], axis=-1)

    checkpoint_predictions = []
    weights = []

    training_data = {
        'models': []
    }

    for model_idx in range(args.num_models):
        model = build_model(embedding_matrix, y_aux_train.shape[-1], args.lstm_units, args.dense_hidden_units, args.max_len)

        checkpoint = ModelCheckpoint(f'model_{model_idx}.keras', monitor='val_loss', save_best_only=True, mode='min')

        model_progress = {
            'model_idx': model_idx,
            'epochs': []
        }

        for global_epoch in range(args.epochs):
            history = model.fit(
                x_train,
                [y_train, y_aux_train],
                validation_data=(x_valid, [y_valid, y_aux_valid]),
                batch_size=args.batch_size,
                epochs=1,
                verbose=1,
                callbacks=[
                    LearningRateScheduler(lambda epoch: 1e-3 * (0.6 ** global_epoch), verbose=1),
                    checkpoint
                ]
            )

            valid_preds = model.predict(x_valid, batch_size=2048)[0].flatten()
            checkpoint_predictions.append(valid_preds)
            weights.append(2 ** global_epoch)

            epoch_data = {
                'epoch': global_epoch,
                'train_loss': history.history['loss'][0],
                'val_loss': history.history['val_loss'][0],
                'val_precision': history.history.get('val_precision', [0])[-1],
                'val_recall': history.history.get('val_recall', [0])[-1],
            }
            model_progress['epochs'].append(epoch_data)

        training_data['models'].append(model_progress)

    save_training_progress(training_data)

    predictions = np.average(checkpoint_predictions, weights=weights, axis=0)
    auc = roc_auc_score(y_valid, predictions)

    fpr, tpr, thresholds = roc_curve(y_valid, predictions)

    report = classification_report(y_valid, np.where(predictions >= 0.5, 1, 0), output_dict=True)
    save_evaluation_report(report, auc, {
        'false_positive_rate': fpr.tolist(),
        'true_positive_rate': tpr.tolist(),
        'thresholds': thresholds.tolist()
    })

    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a text classification model with LSTM and pre-trained embeddings.")
    parser.add_argument('--train_file', type=str, default="datasets/train.csv", help="Path to the training data CSV file")
    parser.add_argument('--crawl_embedding_file', type=str, default="datasets/crawl-300d-2M.vec", help="Path to the crawl-300d-2M.vec file")
    parser.add_argument('--glove_embedding_file', type=str, default="datasets/glove.840B.300d.txt", help="Path to the glove.840B.300d.txt file")
    parser.add_argument('--max_features', type=int, default=100000, help="Maximum number of words to keep")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch size for training")
    parser.add_argument('--lstm_units', type=int, default=128, help="Number of units in LSTM layer")
    parser.add_argument('--dense_hidden_units', type=int, default=512, help="Number of units in dense hidden layer")
    parser.add_argument('--epochs', type=int, default=4, help="Number of epochs to train")
    parser.add_argument('--max_len', type=int, default=220, help="Maximum length of input sequences")
    parser.add_argument('--num_models', type=int, default=2, help="Number of models to train")
    
    args = parser.parse_args()
    main(args)

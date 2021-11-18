import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFAutoModel

batch_size = 4
seq_len = 512


def preprocessing():
    df = pd.read_csv("data/rottentomatoes/train.tsv", sep='\t')
    df = df.drop_duplicates(subset=['SentenceId'], keep='first')

    num_samples = len(df)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    tokens = tokenizer(df['Phrase'].tolist(), max_length=seq_len, truncation=True,
                       padding='max_length', add_special_tokens=True,
                       return_tensors='np')

    tokens = {'input_ids': tf.cast(tokens['input_ids'], tf.float64),
                    'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)}

    # one hot encoding
    arr = df['Sentiment'].values
    labels = np.zeros((num_samples, arr.max() + 1))
    labels[np.arange(num_samples), arr] = 1

    dataset = tf.data.Dataset.from_tensor_slices((tokens['input_ids'], tokens['attention_mask'], labels))

    def map_func(input_ids, masks, labels):
        # we convert our three-item tuple into a two-item tuple where the input item is a dictionary
        return {'input_ids': input_ids, 'attention_mask': masks}, labels

    # then we use the dataset map method to apply this transformation
    dataset = dataset.map(map_func)

    dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)

    split = 0.9

    # we need to calculate how many batches must be taken to create 90% training set
    size = int((num_samples / batch_size) * split)

    train_ds = dataset.take(size)
    val_ds = dataset.skip(size)

    del dataset

    # Save this data to save time...
    tf.data.experimental.save(train_ds, 'train')
    tf.data.experimental.save(val_ds, 'val')


def training():
    bert = TFAutoModel.from_pretrained('bert-base-cased')

    # two input layers, we ensure layer name variables match to dictionary keys in TF dataset
    input_ids = tf.keras.layers.Input(shape=(seq_len,), name='input_ids', dtype='int32')
    mask = tf.keras.layers.Input(shape=(seq_len,), name='attention_mask', dtype='int32')

    # we access the transformer model within our bert object using the bert attribute (eg bert.bert instead of bert)
    embeddings = bert.bert(input_ids, attention_mask=mask)[1]  # access final activations (alread max-pooled) [1]
    # convert bert embeddings into 5 output classes
    x = tf.keras.layers.Dense(1024, activation='relu')(embeddings)
    y = tf.keras.layers.Dense(5, activation='softmax', name='outputs')(x)

    model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-6)
    loss = tf.keras.losses.CategoricalCrossentropy()
    acc = tf.keras.metrics.CategoricalAccuracy('accuracy')

    model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

    element_spec = ({'input_ids': tf.TensorSpec(shape=(batch_size, seq_len), dtype=tf.float64, name=None),
                     'attention_mask': tf.TensorSpec(shape=(batch_size, seq_len), dtype=tf.float64, name=None)},
                    tf.TensorSpec(shape=(batch_size, 5), dtype=tf.float64, name=None))

    # load the training and validation sets
    train_ds = tf.data.experimental.load('train', element_spec=element_spec)
    val_ds = tf.data.experimental.load('val', element_spec=element_spec)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3
    )

    model.save('sentiment_model')


def predict():
    model = tf.keras.models.load_model('sentiment_model')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def prep_data(text):
        tokens = tokenizer.encode_plus(text, max_length=seq_len,
                                       truncation=True, padding='max_length',
                                       add_special_tokens=True, return_token_type_ids=False,
                                       return_tensors='tf')
        # tokenizer returns int32 tensors, we need to return float64, so we use tf.cast
        return {'input_ids': tf.cast(tokens['input_ids'], tf.float64),
                'attention_mask': tf.cast(tokens['attention_mask'], tf.float64)}

    pd.set_option('display.max_colwidth', None)
    df = pd.read_csv('data/rottentomatoes/test.tsv', sep='\t')
    df = df.drop_duplicates(subset=['SentenceId'], keep='first')
    print(df.head(5))

    for i, row in df.iterrows():
        # get token tensors
        tokens = prep_data(row['Phrase'])
        # get probabilities
        probs = model.predict(tokens)
        # find argmax for winning class
        pred = np.argmax(probs)
        # add to dataframe
        df.at[i, 'Sentiment'] = pred

    h = df.head(10)
    print(h["Phrase"])
    print(h["Sentiment"])


if __name__ == '__main__':
    # preprocessing()
    # training()
    predict()

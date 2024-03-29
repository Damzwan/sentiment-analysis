from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures

import tensorflow as tf
import pandas as pd

import os
import shutil

# https://towardsdatascience.com/sentiment-analysis-in-10-minutes-with-bert-and-hugging-face-294e8a04b671
def execute_bert():
    global train_feat, train_lab, test_feat, test_lab
    model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    dataset = tf.keras.utils.get_file(fname="aclImdb_v1.tar.gz",
                                      origin=URL,
                                      untar=True,
                                      cache_dir='.',
                                      cache_subdir='')

    # Create main directory path ("/aclImdb")
    main_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    # Create sub directory path ("/aclImdb/train")
    train_dir = os.path.join(main_dir, 'train')
    # Remove unsup folder since this is a supervised learning task
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)
    # View the final train folder
    print(os.listdir(train_dir))

    train = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/train', batch_size=30000, validation_split=0.2,
        subset='training', seed=123)
    test = tf.keras.preprocessing.text_dataset_from_directory(
        'aclImdb/train', batch_size=30000, validation_split=0.2,
        subset='validation', seed=123)

    for i in train.take(1):
        train_feat = i[0].numpy()
        train_lab = i[1].numpy()

    train = pd.DataFrame([train_feat, train_lab]).T
    train.columns = [DATA_COLUMN, LABEL_COLUMN]
    train[DATA_COLUMN] = train[DATA_COLUMN].str.decode("utf-8")
    train.head()

    for j in test.take(1):
        test_feat = j[0].numpy()
        test_lab = j[1].numpy()

    test = pd.DataFrame([test_feat, test_lab]).T
    test.columns = [DATA_COLUMN, LABEL_COLUMN]
    test[DATA_COLUMN] = test[DATA_COLUMN].str.decode("utf-8")

    train_InputExamples, validation_InputExamples = convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN)

    train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
    train_data = train_data.shuffle(100).batch(32).repeat(2)

    validation_data = convert_examples_to_tf_dataset(list(validation_InputExamples), tokenizer)
    validation_data = validation_data.batch(32)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy('accuracy')])

    model.fit(train_data, epochs=2, validation_data=validation_data)

    pred_sentences = [
        'This was an awesome movie. I watch it twice my time watching this beautiful movie if I have known it was this good',
        'One of the worst movies of all time. I cannot believe I wasted two hours of my life for this movie']

    tf_batch = tokenizer(pred_sentences, max_length=128, padding=True, truncation=True, return_tensors='tf')
    tf_outputs = model(tf_batch)
    tf_predictions = tf.nn.softmax(tf_outputs[0], axis=-1)
    labels = ['Negative', 'Positive']
    label = tf.argmax(tf_predictions, axis=1)
    label = label.numpy()
    for i in range(len(pred_sentences)):
        print(pred_sentences[i], ": \n", labels[label[i]])


def convert_data_to_examples(train, test, DATA_COLUMN, LABEL_COLUMN):
    train_InputExamples = train.apply(
        lambda x: InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this case
                               text_a=x[DATA_COLUMN],
                               text_b=None,
                               label=x[LABEL_COLUMN]), axis=1)

    validation_InputExamples = test.apply(
        lambda x: InputExample(guid=None,  # Globally unique ID for bookkeeping, unused in this case
                               text_a=x[DATA_COLUMN],
                               text_b=None,
                               label=x[LABEL_COLUMN]), axis=1)

    return train_InputExamples, validation_InputExamples


def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = []  # -> will hold InputFeatures to be converted later

    print(examples[0])

    for e in examples:
        # Documentation is really strong for this method, so please take a look at it
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length,  # truncates if len(s) > max_length
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,  # pads to the right by default # CHECK THIS for pad_to_max_length
            truncation=True
        )

        input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
                                                     input_dict["token_type_ids"], input_dict['attention_mask'])

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=e.label
            )
        )

    def gen():
        for f in features:
            yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                    "token_type_ids": f.token_type_ids,
                },
                f.label,
            )

    return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
                "token_type_ids": tf.TensorShape([None]),
            },
            tf.TensorShape([]),
        ),
    )


DATA_COLUMN = "DATA_COLUMN"
LABEL_COLUMN = "LABEL_COLUMN"

if __name__ == '__main__':
    execute_bert()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

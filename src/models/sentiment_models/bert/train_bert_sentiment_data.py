from google.cloud import bigquery
from google.oauth2 import service_account
import pandas as pd
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Google Cloud credentials and BigQuery client setup
credentials = service_account.Credentials.from_service_account_file(
    '/Users/jacktopping/Documents/HFT-Analysis/src/data_collection/lucky-science-410310-fe46afb2ea6c.json')
client = bigquery.Client(credentials=credentials)

# Load the training data from BigQuery
train_table_id = 'lucky-science-410310.final_datasets.sentiment_training_data'
train_query = f"SELECT * FROM `{train_table_id}`"
df_train = client.query(train_query).to_dataframe()

# Preprocess the data
df_train['ProcessedArticleTitle'] = df_train['ProcessedArticleTitle'].apply(lambda x: x.replace('\n', ' '))

# Using the 'bert-base-uncased' model
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)


# Function to convert DataFrame rows to InputExample objects
def convert_data_to_examples(train):
  train_InputExamples = train.apply(lambda x: InputExample(guid=None,
                                                           text_a = x['ProcessedArticleTitle'],
                                                           text_b = None,
                                                           label = x['SentimentScore']), axis = 1)
  return train_InputExamples


# Function to convert InputExamples to the required format for BERT
def convert_examples_to_tf_dataset(examples, tokenizer, max_length=128):
    features = []

    for e in examples:
        input_dict = tokenizer.encode_plus(
            e.text_a,
            add_special_tokens=True,
            max_length=max_length,
            return_token_type_ids=True,
            return_attention_mask=True,
            pad_to_max_length=True,
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


# Convert data
train_InputExamples = convert_data_to_examples(df_train)
train_data = convert_examples_to_tf_dataset(list(train_InputExamples), tokenizer)
train_data = train_data.shuffle(100).batch(32).repeat(2)

# Load BERT model
model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metric = tf.keras.metrics.BinaryAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

# Train the model
model.fit(train_data, epochs=3, steps_per_epoch=115)

# Save the model
model.save_pretrained('./bert_sentiment_model')
tokenizer.save_pretrained('./bert_sentiment_model')

import pickle
import numpy as np
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras import backend
from tensorflow.keras.callbacks import EarlyStopping
from Attention_Layer import AttentionLayer
from nltk.corpus import stopwords
from matplotlib import pyplot
from Seq2Seq_Model_V1_2.Utilities import contraction_mapping

MAX_TEXT_LENGTH = 1200
MAX_SUMMARY_LENGTH = 75
STOP_WORDS = set(stopwords.words('english'))
RARE_WORD_THRESHOLD_TEXT = 2
RARE_WORD_THRESHOLD_SUMMARY = 2


def get_training_data():
    file = open('../Fariytale_Texts/cnn_dataset.pkl', 'rb')
    trainings_data = pickle.load(file)
    file.close()

    for article in trainings_data:
        concat_summary = ""
        for summary in article['summaries']:
            concat_summary = concat_summary + (summary + " ")
        article['summaries'] = [concat_summary]

    trainings_data = [article for article in trainings_data if article['article']]
    trainings_data = [article for article in trainings_data if article['summaries']]
    return trainings_data


def text_cleaning(tc_text, mode):
    new_text = tc_text.lower()
    new_text = re.sub(r'\([^()]*\)', '', new_text)
    new_text = re.sub('"', '', new_text)
    new_text = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in new_text.split(" ")])
    new_text = re.sub(r"'s\b", "", new_text)
    new_text = re.sub("[^a-zA-Z]", " ", new_text)
    new_text = re.sub('[m]{2,}', 'mm', new_text)

    if mode == 0:
        tokens = [word for word in new_text.split() if not (word in STOP_WORDS)]
    else:
        tokens = new_text.split()

    long_words = []
    for token in tokens:
        if len(token) > 1:
            long_words.append(token)
    return (" ".join(long_words)).strip()


def prepare_trainings_data(pld_unshortened_training_data):
    print("Cleaning Articles for Training...")

    for cs_element in pld_unshortened_training_data:
        if len(cs_element['article']) >= 2:
            cs_element['article'] = [cs_element['article'][0] + " " + cs_element['article'][1]]
        for text in cs_element['article']:
            prepared_articles.append(text_cleaning(text, 0))

    print("Finished the Article Cleaning")
    print("Cleaning Summaries for Training...")

    for cs_element in pld_unshortened_training_data:
        for text in cs_element['summaries']:
            prepared_summaries.append(text_cleaning(text, 1))

    print("Finished the Summaries Cleaning")

    return prepared_summaries, prepared_articles


def shorten_trainings_data(prepared_article, prepared_summary):
    short_selected_article = []
    short_selected_summary = []

    for element in range(len(prepared_article)):
        if len(prepared_summary[element].split()) <= MAX_SUMMARY_LENGTH and \
                len(prepared_article[element].split()) <= MAX_TEXT_LENGTH:
            short_selected_article.append(prepared_article[element])
            short_selected_summary.append(prepared_summary[element])

    std_shortened_training_data = pd.DataFrame({'text': short_selected_article, 'summary': short_selected_summary})
    std_shortened_training_data = std_shortened_training_data[:10]
    std_shortened_training_data['summary'] = std_shortened_training_data['summary'].apply(
        lambda x: 'sumsta ' + x + ' sumend')

    return std_shortened_training_data


def prepare_tokenizer(pt_text_training, pt_summary_training):
    text_tokenizer = Tokenizer()
    summary_tokenizer = Tokenizer()

    text_tokenizer.fit_on_texts(list(pt_text_training))

    pt_counter = 0
    total_counter = 0
    frequency = 0
    total_frequency = 0

    for key, value in text_tokenizer.word_counts.items():
        total_counter = total_counter + 1
        total_frequency = total_frequency + value
        if value < RARE_WORD_THRESHOLD_TEXT:
            pt_counter = pt_counter + 1
            frequency = frequency + value

    text_tokenizer = Tokenizer(num_words=total_counter - pt_counter)
    text_tokenizer.fit_on_texts(list(pt_text_training))

    summary_tokenizer.fit_on_texts(list(pt_summary_training))

    pt_counter = 0
    total_counter = 0
    frequency = 0
    total_frequency = 0

    for key, value in summary_tokenizer.word_counts.items():
        total_counter = total_counter + 1
        total_frequency = total_frequency + value
        if value < RARE_WORD_THRESHOLD_SUMMARY:
            pt_counter = pt_counter + 1
            frequency = frequency + value

    summary_tokenizer = Tokenizer(num_words=total_counter - pt_counter)
    summary_tokenizer.fit_on_texts(list(pt_summary_training))

    return text_tokenizer, summary_tokenizer


def create_trainings_sequence(cts_article_tokenizer, cts_summary_tokenizer, cts_article_training,
                              cts_article_validation, cts_summary_training, cts_summary_validation):
    cts_article_training_sequence = cts_article_tokenizer.texts_to_sequences(cts_article_training)
    cts_article_validation_sequence = cts_article_tokenizer.texts_to_sequences(cts_article_validation)

    cts_article_training_sequence = pad_sequences(cts_article_training_sequence, maxlen=MAX_TEXT_LENGTH, padding='post')
    cts_article_validation_sequence = pad_sequences(cts_article_validation_sequence, maxlen=MAX_TEXT_LENGTH, padding='post')

    cts_article_vocabulary = cts_article_tokenizer.num_words + 1
    print(cts_article_vocabulary)

    cts_summary_training_sequence = cts_summary_tokenizer.texts_to_sequences(cts_summary_training)
    cts_summary_validation_sequence = cts_summary_tokenizer.texts_to_sequences(cts_summary_validation)

    cts_summary_training_sequence = pad_sequences(cts_summary_training_sequence, maxlen=MAX_SUMMARY_LENGTH, padding='post')
    cts_summary_validation_sequence = pad_sequences(cts_summary_validation_sequence, maxlen=MAX_SUMMARY_LENGTH, padding='post')

    cts_summary_vocabulary = cts_summary_tokenizer.num_words + 1

    print(cts_summary_vocabulary)
    print(cts_summary_tokenizer.word_counts['sumsta'], len(cts_summary_training_sequence))

    index = []

    for i in range(len(cts_summary_training_sequence)):
        counter = 0
        for j in cts_summary_training_sequence[i]:
            if j != 0:
                counter = counter + 1
        if counter == 2:
            index.append(i)

    cts_summary_training_sequence = np.delete(cts_summary_training_sequence, index, axis=0)
    cts_article_training_sequence = np.delete(cts_article_training_sequence, index, axis=0)

    print(len(cts_summary_training_sequence))
    print(len(cts_article_training_sequence))

    index = []

    for i in range(len(cts_summary_validation_sequence)):
        counter = 0
        for j in cts_summary_validation_sequence[i]:
            if j != 0:
                counter = counter + 1
        if counter == 2:
            index.append(i)

    cts_summary_validation_sequence = np.delete(cts_summary_validation_sequence, index, axis=0)
    cts_article_validation_sequence = np.delete(cts_article_validation_sequence, index, axis=0)

    return cts_summary_training_sequence, cts_article_training_sequence, cts_summary_validation_sequence, \
           cts_article_validation_sequence, cts_article_vocabulary, cts_summary_vocabulary


def decode_sequence(input_sequence):
    ds_encoder_output, ds_encoder_hidden, ds_encoder_context = encoder_model.predict(input_sequence)

    target_sequence = np.zeros((1, 1))
    target_sequence[0, 0] = target_word_index['sumsta']

    stop_condition = False
    decoded_sentence = ''

    while not stop_condition:
        output_tokens, hidden, context = decoder_model.predict(
            [target_sequence] + [ds_encoder_output, ds_encoder_hidden, ds_encoder_context])

        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_word_index[sampled_token_index + 1]

        if sampled_token != 'sumend':
            decoded_sentence = decoded_sentence + ' ' + sampled_token

        if sampled_token == 'sumend' or len(decoded_sentence.split()) >= (MAX_SUMMARY_LENGTH - 1):
            stop_condition = True

        target_sequence = np.zeros((1, 1))
        target_sequence[0, 0] = sampled_token_index

        ds_encoder_hidden, ds_encoder_context = hidden, context

    return decoded_sentence


def sequence2summary(input_sequence):
    summary = ''
    for s2s_element in input_sequence:
        if (s2s_element != 0 and s2s_element != target_word_index['sumsta']) \
                and s2s_element != target_word_index['sumend']:
            summary = summary + reverse_target_word_index[s2s_element] + ' '
    return summary


def sequence2text(input_sequence):
    s2t_text = ''
    for s2t_elements in input_sequence:
        if s2t_elements != 0:
            s2t_text = s2t_text + reverse_source_word_index[s2t_elements] + ' '
    return s2t_text


unshortened_training_data = get_training_data()
loading = False

prepared_articles = []
prepared_summaries = []

prepared_articles, prepared_summaries = prepare_trainings_data(unshortened_training_data)

preprocessed_training_data = pd.DataFrame({'text': prepared_articles, 'summary': prepared_summaries})

prepared_articles = np.array(preprocessed_training_data['text'])
prepared_summaries = np.array(preprocessed_training_data['summary'])

# Utilities.length_analysis(preprocessed_training_data, MAX_SUMMARY_LENGTH, MAX_TEXT_LENGTH)

shortened_training_data = shorten_trainings_data(prepared_articles, prepared_summaries)

article_training, article_validation, summary_training, summary_validation = train_test_split(np.array(
    shortened_training_data['text']), np.array(shortened_training_data['summary']), test_size=0.1, random_state=0,
    shuffle=True)

article_tokenizer, summary_tokenizer = prepare_tokenizer(article_training, summary_training)

summary_training_sequence, article_training_sequence, summary_validation_sequence, article_validation_sequence, \
article_vocabulary, summary_vocabulary = create_trainings_sequence(article_tokenizer, summary_tokenizer,
                                                                   article_training, article_validation,
                                                                   summary_training, summary_validation)

# Model Creation
backend.clear_session()

latent_dimension = 300
embedding_dimension = 100

encoder_inputs = Input(shape=(MAX_TEXT_LENGTH,))

encoder_embedding_layer = Embedding(article_vocabulary, embedding_dimension, trainable=True)
encoder_embedded = encoder_embedding_layer(encoder_inputs)

encoder_lstm_layer1 = LSTM(latent_dimension,
                           return_sequences=True,
                           return_state=True,
                           dropout=0.4,
                           recurrent_dropout=0.4)

encoder_output1, state_hidden1, state_context1 = encoder_lstm_layer1(encoder_embedded)

encoder_lstm_layer2 = LSTM(latent_dimension,
                           return_sequences=True,
                           return_state=True,
                           dropout=0.4,
                           recurrent_dropout=0.4)

encoder_output2, state_hidden2, state_context2 = encoder_lstm_layer2(encoder_output1)

encoder_lstm_layer3 = LSTM(latent_dimension,
                           return_sequences=True,
                           return_state=True,
                           dropout=0.4,
                           recurrent_dropout=0.4)
encoder_output, state_hidden, state_context = encoder_lstm_layer3(encoder_output2)

decoder_inputs = Input(shape=(None,))

decoder_embedding_layer = Embedding(summary_vocabulary, embedding_dimension, trainable=True)
decoder_embedded = decoder_embedding_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dimension,
                    return_sequences=True,
                    return_state=True,
                    dropout=0.4,
                    recurrent_dropout=0.2)

decoder_output, decoder_forward_state, decoder_back_state = decoder_lstm(decoder_embedded,
                                                                         initial_state=[state_hidden, state_context])

attention_layer = AttentionLayer(name='attention_layer')
attention_output, attention_states = attention_layer([encoder_output, decoder_output])

decoder_concatenated_input = Concatenate(axis=-1, name='concat_layer')([decoder_output, attention_output])

decoder_dense_layer = TimeDistributed(Dense(summary_vocabulary, activation='softmax'))
decoder_output = decoder_dense_layer(decoder_concatenated_input)

model = Model([encoder_inputs, decoder_inputs], decoder_output)
model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)

# End Model Creation
# Learning Phase
if not loading:
    history = model.fit([article_training_sequence, summary_training_sequence[:, :-1]],
                        summary_training_sequence.reshape(summary_training_sequence.shape[0],
                                                          summary_training_sequence.shape[1], 1)[:, 1:],
                        epochs=50,
                        callbacks=[early_stopping],
                        batch_size=2,
                        validation_data=([article_validation_sequence, summary_validation_sequence[:, : -1]],
                                         summary_validation_sequence.reshape(summary_validation_sequence.shape[0],
                                                                             summary_validation_sequence.shape[1], 1)
                                         [:, 1:]
                                         )
                        )

reverse_target_word_index = summary_tokenizer.index_word
reverse_source_word_index = article_tokenizer.index_word
target_word_index = summary_tokenizer.word_index

encoder_model = Model(inputs=encoder_inputs, outputs=[encoder_output, state_hidden, state_context])

decoder_state_input_hidden = Input(shape=(latent_dimension,))
decoder_state_input_context = Input(shape=(latent_dimension,))
decoder_hidden_state_input = Input(shape=(MAX_TEXT_LENGTH, latent_dimension))

decoder_embedding_layer2 = decoder_embedding_layer(decoder_inputs)

decoder_output2, state_hidden2, state_context2 = \
    decoder_lstm(decoder_embedding_layer2, initial_state=[decoder_state_input_hidden, decoder_state_input_context])

attention_output_inference, attention_states_inference = attention_layer([decoder_hidden_state_input, decoder_output2])
decoder_inference_concat = Concatenate(axis=-1, name='concat')([decoder_output2, attention_output_inference])

decoder_output2 = decoder_dense_layer(decoder_inference_concat)

decoder_model = Model([decoder_inputs] + [decoder_hidden_state_input, decoder_state_input_hidden,
                                          decoder_state_input_context],[decoder_output2] + [state_hidden2,
                                                                                            state_context2])

# End Learning Phase
# Save or Load Model Weights
if not loading:
    encoder_model.save_weights("model/encoder")
    decoder_model.save_weights("model/decoder")
else:
    encoder_model.load_weights("model/encoder")
    decoder_model.load_weights("model/decoder")

# Save or Load Model Weights
# Start Create Summary
input_text = "sumsta Hard by a great forest dwelt a poor wood-cutter with his wife and his two children. The boy " \
             "was called Hansel and the girl Gretel. He had little to bite and to break, and once when great dearth " \
             "fell on the land, he could no longer procure even daily bread. Now when he thought over this by night " \
             "in his bed, and tossed about in his anxiety, he groaned and said to his wife, what is to become of us. " \
             "How are we to feed our poor children, when we no longer have anything even for ourselves. I'll tell" \
             "you what, husband, answered the woman, early to-morrow morning we will take the children out into the" \
             "forest to where it is the thickest. There we will light a fire for them, and give each of them one" \
             "more piece of bread, and then we will go to our work and leave them alone. They will not find the way" \
             "home again, and we shall be rid of them. No, wife, said the man, I will not do that. How can I bear" \
             "to leave my children alone in the forest.  The wild animals would soon come and tear them to pieces." \
             "O' you fool, said she, then we must all four die of hunger, you may as well plane the planks for our" \
             "coffins, and she left him no peace until he consented. But I feel very sorry for the poor children," \
             "all the same, said the man. sumend"

input_text_array = [input_text.lower()]
input_text_sequence = np.array(article_tokenizer.texts_to_sequences(input_text_array))
input_text_sequence = pad_sequences(input_text_sequence, maxlen=MAX_TEXT_LENGTH, padding='post')

print("Articel: ", sequence2text(input_text_sequence[0]))
print("Predicted summary: ", decode_sequence(input_text_sequence[0].reshape(1, MAX_TEXT_LENGTH)))

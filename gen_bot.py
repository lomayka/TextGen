import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Neural Net Layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding

# Neural Net Training
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential([
    Embedding(23692, 50, input_length=12),
    LSTM(150, return_sequences=True),
    LSTM(150),
    Dense(150, activation='relu'),
    Dense(23691, activation='softmax')
])

model.load_weights('model_weights.hdf5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
reverse_word_map = pickle.load(open('reverse_word_map.pkl', 'rb'))

from telegram.ext import MessageHandler, Filters
from telegram.ext import Updater
from telegram.ext import CommandHandler


def gen(model, seq, max_len=12):
    ''' Generates a sequence given a string seq using specified model until the total sequence length
    reaches max_len'''
    # Tokenize the input string
    tokenized_sent = tokenizer.texts_to_sequences([seq])
    max_len = max_len + len(tokenized_sent[0])
    # If sentence is not as long as the desired sentence length, we need to 'pad sequence' so that
    # the array input shape is correct going into our LSTM. the `pad_sequences` function adds
    # zeroes to the left side of our sequence until it becomes 19 long, the number of input features.
    while len(tokenized_sent[0]) < max_len:
        padded_sentence = pad_sequences(tokenized_sent[-12:], maxlen=12)
        op = model.predict(np.asarray(padded_sentence).reshape(1, -1))
        tokenized_sent[0].append(op.argmax() + 1)

    return " ".join(map(lambda x: reverse_word_map[x], tokenized_sent[0]))


start_text = "Я бот для генерации текста поздравлений с днём рождения.\n" \
             "Текст был взят с поздравок.ру, поэтому нецензурщины и брани я не знаю," \
             " пожалуйста не ругайтесь при общении со мной.\n" \
             "Доступны следующие команды\n" \
             "/gen и далее парочку начальных слов генерирует поздравление с днём рождения" \
             "/help выведет чуть больше инфы про бота и напомнит про различные команды"

help_text = "На данный момент через команду /gen я дописываю определённое количество слов к" \
            " начальной строке поздравления. Нецензурщины и английских слов я не понимаю, как и особых символов." \
            "Писать их мне бесполезно, как и имена собственные типо имён, названий городов и всякой другой белиберды" \
            "Поздравления с днём рождения это не финальный этап проекта, дальше ещё шото нагенерим"

def gen_string(test_string, sequence_length=50, model=model):
    return gen(model, test_string, sequence_length)

def generate(update, context):
    print(update.message.text)
    generated = gen_string(update.message.text, 20)
    print(generated)
    context.bot.send_message(chat_id=update.effective_chat.id, text=generated)

def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=start_text)

def help_str(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=help_text)

updater = Updater(token='Token', use_context=True)
dp = updater.dispatcher
start_handler = CommandHandler('start', start)
help_handler = CommandHandler('help', help_str)
gen_handler = CommandHandler('gen', generate)
dp.add_handler(start_handler)
dp.add_handler(help_handler)
dp.add_handler(gen_handler)
updater.start_polling()
updater.idle()
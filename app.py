from __future__ import print_function, division, unicode_literals

import gradio as gr

import sys
import os
from os.path import abspath, dirname

import json
import numpy as np

from torchmoji.sentence_tokenizer import SentenceTokenizer
from torchmoji.model_def import torchmoji_emojis
from emoji import emojize

from huggingface_hub import hf_hub_download

HF_TOKEN = os.getenv('HF_TOKEN')
hf_writer = gr.HuggingFaceDatasetSaver(
    HF_TOKEN,
    "crowdsourced-deepmoji-flags",
    private=True,
    separate_dirs=False
)

model_name = "Pendrokar/TorchMoji"
model_path = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
vocab_path = hf_hub_download(repo_id=model_name, filename="vocabulary.json")

emoji_codes = []
with open('./data/emoji_codes.json', 'r') as f:
    emoji_codes = json.load(f)

maxlen = 30

with open(vocab_path, 'r') as f:
    vocabulary = json.load(f)

st = SentenceTokenizer(vocabulary, maxlen)

model = torchmoji_emojis(model_path)

def pre_hf_writer(*args):
    return hf_writer(args)

def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]

def predict(deepmoji_analysis, emoji_count):
    if deepmoji_analysis.strip() == '':
        # dotted face emoji
        return {"🫥":1}

    return_label = {}
    # tokenize input text
    tokenized, _, _ = st.tokenize_sentences([deepmoji_analysis])

    if len(tokenized) == 0:
        # dotted face emoji
        return {"🫥":1}

    prob = model(tokenized)

    for prob in [prob]:
        # Find top emojis for each sentence. Emoji ids (0-63)
        # correspond to the mapping in emoji_overview.png
        # at the root of the torchMoji repo.
        scores = []
        for i, t in enumerate([deepmoji_analysis]):
            t_prob = prob[i]
            # sort top
            ind_top_ids = top_elements(t_prob, emoji_count)

            for ind in ind_top_ids:
                # unicode emoji + :alias:
                label_emoji = emojize(emoji_codes[str(ind)], language="alias")
                label_name = label_emoji + emoji_codes[str(ind)]
                # propability
                label_prob = t_prob[ind]
                return_label[label_name] = label_prob

    if len(return_label) == 0:
        # dotted face emoji
        return {"🫥":1}

    return return_label

default_input = "This is the shit!"

input_textbox = gr.Textbox(
    label="English Text",
    info="ignores: emojis, emoticons, numbers, URLs",
    lines=1,
    value=default_input,
    autofocus=True
)
slider = gr.Slider(1, 64, value=5, step=1, label="Top # Emoji", info="Choose between 1 and 64 top emojis to show")

gradio_app = gr.Interface(
    predict,
    [
        input_textbox,
        slider,
    ],
    outputs=gr.Label(
        label="Suitable Emoji",
        # could not auto select example output
        value={
            "🎧:headphones:" :0.10912112891674042,
            "🎶:notes:" :0.10073345899581909,
            "👌:ok_hand:" :0.05672002583742142,
            "👏:clap:" :0.0559493824839592,
            "👍:thumbsup:" :0.05157269537448883
        }
    ),
    examples=[
        ["This is shit!", 5],
        ["You love hurting me, huh?", 5],
        ["I know good movies, this ain't one", 5],
        ["It was fun, but I'm not going to miss you", 5],
        ["My flight is delayed.. amazing.", 5],
        ["What is happening to me??", 5],
        ["Wouldn't it be a shame, if something were to happen to her?", 5],
        ["Embrace your demise!", 10],
        ["This is the shit!", 5],
    ],
    cache_examples=True,
    live=True,
    title="🎭 DeepMoji 🎭",
    # allow_duplication=True,
    # flagged saved to hf dataset
    # FIXME: gradio sends output as a saveable filename, crashing flagging
    # allow_flagging="manual",
    # flagging_options=["'🚩 sarcasm / innuendo 😏'", "'🚩 unsuitable / other'"],
    # flagging_callback=hf_writer
)

if __name__ == "__main__":
    gradio_app.launch()
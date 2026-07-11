import os
import json
import ast


def normalize_name(s):
    return str(s).lower().replace("_", " ").replace("-", " ").strip()


def extract_synset_from_path(path):
    return os.path.basename(path).split("_")[0]


def load_wordnet_json(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def build_wordnet_maps(path):
    data = load_wordnet_json(path)

    foldername_to_synset = {}
    synset_to_readable = {}

    for _, item in data.items():
        wn_id = item["id"]
        synset = "n" + wn_id.split("-")[0]
        readable = item["label"].split(",")[0].strip()

        synset_to_readable[synset] = readable

        for label in item["label"].split(","):
            foldername_to_synset[normalize_name(label)] = synset

    return foldername_to_synset, synset_to_readable


def readable_label_to_synset(readable_label, foldername_to_synset):
    return foldername_to_synset.get(normalize_name(readable_label), "unknown")


def make_prompt(label):
    label = str(label).replace("_", " ").replace("-", " ").strip()
    return f"a realistic photo of a {label}, high quality, detailed"
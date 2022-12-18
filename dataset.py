import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd


class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        sentence = str(self.data.text[index])
        sentence = " ".join(sentence.split())
        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "span1_begin": self.data.span1_begin[index],
            "span2_begin": self.data.span2_begin[index],
            "span1_end": self.data.span1_end[index],
            "span2_end": self.data.span2_end[index],
            "labels": torch.tensor(self.data.label_id[index], dtype=torch.long),
        }

    def __len__(self):
        return self.len


def get_dataloader(df, tokenizer, max_len=512, batch_size=5):
    df = df.reset_index(drop=True)
    data_set = Triage(df, tokenizer, max_len)
    params = {"batch_size": batch_size, "shuffle": True, "num_workers": 0}
    data_loader = DataLoader(data_set, **params)
    return data_loader


def get_wsc_json(file):
    with open(file, "rb") as f:
        datas = []
        for line in f.readlines():
            datas.append(json.loads(line))
    return datas


def transfers_wsc_df(df):
    df["span1_begin"] = df.apply(
        lambda row: _get_range(row["target"]["span1_index"]), axis=1
    )
    df["span1_end"] = df.apply(
        lambda row: _get_range(
            row["target"]["span1_index"], row["target"]["span1_text"]
        ),
        axis=1,
    )
    df["span2_begin"] = df.apply(
        lambda row: _get_range(row["target"]["span2_index"]), axis=1
    )
    df["span2_end"] = df.apply(
        lambda row: _get_range(
            row["target"]["span2_index"], row["target"]["span2_text"]
        ),
        axis=1,
    )
    df["label_id"] = df["label"].apply(lambda x: 1 if x == "true" else 0)
    return df


def _get_range(index, length=""):
    return index + len(length)


def get_wsc_dataloader(max_len=200, batch_size=10):

    tokenizer = BertTokenizer.from_pretrained("pretrained\chinese-roberta-wwm-ext",local_files_only=True)

    datas = get_wsc_json("dataset/train.json")
    train_df = pd.DataFrame(datas)
    train_df = transfers_wsc_df(train_df)
    train_dataloader = get_dataloader(train_df, tokenizer, max_len, batch_size)

    datas = get_wsc_json("dataset/dev.json")
    dev_df = pd.DataFrame(datas)
    dev_df = transfers_wsc_df(dev_df)
    dev_dataloader = get_dataloader(dev_df, tokenizer, max_len, batch_size)
    return train_dataloader, dev_dataloader


if __name__ == "__main__":
    get_wsc_dataloader()

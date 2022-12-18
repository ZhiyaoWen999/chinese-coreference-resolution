import pandas as pd
import os
import torch
from tqdm import tqdm
from logger import logger
from transformers import BertTokenizer


from dataset import get_wsc_dataloader, get_wsc_json
from model import Roberta


def calcuate_accu(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct


class Config:
    device = os.environ.get("DEVICE", "cuda:0")
    lr = 0.00001
    epochs = 15
    batch_size = 25
    max_length = 200
    output_model_dir = os.path.dirname(os.path.abspath(__file__)) + "/trained/"
    pretrained_dir = os.path.dirname(os.path.abspath(__file__)) + "/pretrained/"
    pretrained_model = "chinese-roberta-wwm-ext"


config = Config()


def download_hfl(name, save_dir):
    from transformers import BertTokenizer, BertModel

    tokenizer = BertTokenizer.from_pretrained(
        "hfl/" + name, cache_dir=save_dir + "cache"
    )
    model = BertModel.from_pretrained("hfl/" + name, cache_dir=save_dir + "cache")
    tokenizer.save_pretrained(save_dir + name + "/")
    model.save_pretrained(save_dir + name + "/")


class Finetuner:
    __name__ = "dnn"

    def __init__(self, model_name, config=config):
        self.model_name = model_name
        self.config = config
        self.config.model_name = model_name
        if not os.path.exists(os.path.join(config.output_model_dir, model_name)):
            os.makedirs(os.path.join(config.output_model_dir, model_name))
        if not os.path.exists(
            os.path.join(config.pretrained_dir, config.pretrained_model)
        ):
            download_hfl(config.pretrained_model, config.pretrained_dir)

    def train(self):
        config = self.config
        self.model = Roberta(config, method="origin")
        self.tokenizer = BertTokenizer.from_pretrained(
            os.path.join(config.pretrained_dir, config.pretrained_model)
        )
        self.model.to(config.device)

        # 损失函数
        self.loss_function = torch.nn.CrossEntropyLoss()
        # 优化器
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=config.lr)
        self.train_dataloader, self.eval_dataloader = get_wsc_dataloader(
            self.config.max_length, self.config.batch_size
        )

        for epoch in tqdm(range(config.epochs)):
            self.train_batch(epoch)

    def train_batch(self, epoch):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        self.model.train()
        for _, data in tqdm(enumerate(self.train_dataloader, 0)):
            input_ids = data["input_ids"].to(self.config.device, dtype=torch.long)
            attention_mask = data["attention_mask"].to(
                self.config.device, dtype=torch.long
            )
            targets = data["labels"].to(self.config.device, dtype=torch.long)
            span1_begin = data["span1_begin"].to(self.config.device, dtype=torch.long)
            span2_begin = data["span2_begin"].to(self.config.device, dtype=torch.long)
            span1_end = data["span1_end"].to(self.config.device, dtype=torch.long)
            span2_end = data["span2_end"].to(self.config.device, dtype=torch.long)
            input = (
                input_ids,
                attention_mask,
                span1_begin,
                span1_end,
                span2_begin,
                span2_end,
            )

            outputs = self.model(input)
            loss = self.loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)
            if _ % 10 == 0:
                loss_step = tr_loss / nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                logger.info(f"Training Loss per 5000 steps: {loss_step}")
                logger.info(f"Training Accuracy per 5000 steps: {accu_step}")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        logger.info(
            f"The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}"
        )
        epoch_loss = tr_loss / nb_tr_steps
        epoch_accu = (n_correct * 100) / nb_tr_examples
        logger.info(f"Training Loss Epoch: {epoch_loss}")
        logger.info(f"Training Accuracy Epoch: {epoch_accu}")
        logger.info(
            "This is the validation section to logger.info the accuracy and see how it performs"
        )
        logger.info(
            "Here we are leveraging on the dataloader crearted for the validation dataset, the approcah is using more of pytorch"
        )
        acc = self.valid()
        logger.info("Accuracy on test data = %0.2f%%" % acc)
        matric_file = os.path.join(
            self.config.output_model_dir, self.model_name, "matric.tsv"
        )
        if not os.path.exists(matric_file):
            with open(matric_file, "w") as f:
                f.write("epoch\ttrain_loss\ttrain_acc\teval_acc\n")
        with open(matric_file, "a") as f:
            epoch_accu, acc = round(epoch_accu) / 100, round(acc, 3) / 100
            epoch_finished = epoch + 1
            f.write(f"{epoch_finished}\t{epoch_loss}\t{epoch_accu}\t{acc}\n")
        model_save_dir = os.path.join(
            self.config.output_model_dir, self.model_name, "epoch_%s" % epoch
        )
        self.model.embeddings.save_pretrained(model_save_dir)
        self.tokenizer.save_pretrained(model_save_dir)

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            token_data = self.tokenizer(
                [data["text"]],
                return_tensors="pt",
                padding=True,
                max_length=self.config.max_length,
            ).to(self.config.device)
            inputs = (
                token_data["input_ids"].to(self.config.device, dtype=torch.long),
                token_data["attention_mask"].to(self.config.device, dtype=torch.long),
                torch.tensor([data["target"]["span1_index"]]).to(
                    self.config.device, dtype=torch.long
                ),
                torch.tensor(
                    [data["target"]["span1_index"] + data["target"]["span1_length"]]
                ).to(self.config.device, dtype=torch.long),
                torch.tensor([data["target"]["span2_index"]]).to(
                    self.config.device, dtype=torch.long
                ),
                torch.tensor(
                    [data["target"]["span2_index"] + data["target"]["span2_length"]]
                ).to(self.config.device, dtype=torch.long),
            )
            outputs = self.model(inputs)
            return outputs.to("cpu").numpy()[0][1]

    def test(self, datas):
        i = 0
        for data in tqdm(datas):
            data["target"]["span1_length"] = len(data["target"]["span1_text"])
            data["target"]["span2_length"] = len(data["target"]["span2_text"])
            proba = self.predict(data)
            if proba > 0.5 and data["label"] == "true":
                i += 1
            elif proba < 0.5 and data["label"] == "false":
                i += 1
        print(round(i / len(datas), 3))

    def load(self, file_dir=None):
        if file_dir is None:
            matric_file = os.path.join(
                self.config.output_model_dir, self.model_name, "matric.tsv"
            )
            df = pd.read_csv(matric_file, sep="\t", index_col="epoch")
            epoch = df["eval_acc"].argmax()
            logger.info(epoch)
            config.model_saved_dir = os.path.join(
                self.config.output_model_dir, self.model_name, "epoch_%s" % epoch
            )
        else:
            config.model_saved_dir = file_dir
        self.model = Roberta(config, method="load").to(self.config.device)
        self.tokenizer = BertTokenizer.from_pretrained(config.model_saved_dir)

    def valid(self):
        self.train_dataloader, self.eval_dataloader = get_wsc_dataloader(
            self.config.max_length, self.config.batch_size
        )
        self.model.eval()
        n_correct = 0
        n_correct = 0
        nb_tr_examples = 0
        with torch.no_grad():
            for _, data in enumerate(tqdm(self.eval_dataloader), 0):
                input_ids = data["input_ids"].to(self.config.device, dtype=torch.long)
                attention_mask = data["attention_mask"].to(
                    self.config.device, dtype=torch.long
                )
                targets = data["labels"].to(self.config.device, dtype=torch.long)
                span1_begin = data["span1_begin"].to(
                    self.config.device, dtype=torch.long
                )
                span2_begin = data["span2_begin"].to(
                    self.config.device, dtype=torch.long
                )
                span1_end = data["span1_end"].to(self.config.device, dtype=torch.long)
                span2_end = data["span2_end"].to(self.config.device, dtype=torch.long)
                input = (
                    input_ids,
                    attention_mask,
                    span1_begin,
                    span1_end,
                    span2_begin,
                    span2_end,
                )
                outputs = self.model(input)
                big_val, big_idx = torch.max(outputs, dim=1)
                n_correct += calcuate_accu(big_idx, targets)
                nb_tr_examples += targets.size(0)
        epoch_accu = (n_correct * 100) / nb_tr_examples
        logger.info(f"Validation Accuracy Epoch: {epoch_accu}")
        return epoch_accu


if __name__ == "__main__":
    model_name = "ea_1-cosine_softmax"
    finetuner = Finetuner(model_name)
    finetuner.train()
    finetuner.load()
    finetuner.valid()
    datas = get_wsc_json("dataset/dev.json")
    finetuner.test(datas)

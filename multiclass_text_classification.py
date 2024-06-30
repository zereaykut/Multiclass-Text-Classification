import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

class BERTClassifier:
    def __init__(self, model_name, num_classes, max_len, batch_size, learning_rate, epochs):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
        self.max_len = max_len
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs

    def prepare_data(self, texts, labels):
        dataset = TextDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_len=self.max_len
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def train(self, train_texts, train_labels, val_texts, val_labels):
        train_loader = self.prepare_data(train_texts, train_labels)
        val_loader = self.prepare_data(val_texts, val_labels)

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        loss_fn = torch.nn.CrossEntropyLoss().to(device)

        self.model = self.model.to(device)

        for epoch in range(self.epochs):
            print(f'Epoch {epoch + 1}/{self.epochs}')
            print('-' * 10)

            self.train_epoch(train_loader, optimizer, loss_fn, scheduler)
            self.eval_epoch(val_loader, loss_fn)

    def train_epoch(self, data_loader, optimizer, loss_fn, scheduler):
        self.model.train()

        total_loss = 0
        correct_predictions = 0

        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions.double() / len(data_loader.dataset)

        print(f'Train loss {avg_loss} accuracy {accuracy}')

    def eval_epoch(self, data_loader, loss_fn):
        self.model.eval()

        total_loss = 0
        correct_predictions = 0

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                _, preds = torch.max(logits, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        accuracy = correct_predictions.double() / len(data_loader.dataset)

        print(f'Validation loss {avg_loss} accuracy {accuracy}')

    def predict(self, texts):
        self.model.eval()

        inputs = self.tokenizer.batch_encode_plus(
            texts,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)

        return preds

# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Example data
    texts = ["This is a text example"] * 100
    labels = [0] * 50 + [1] * 50

    train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1)

    classifier = BERTClassifier(
        model_name='bert-base-uncased',
        num_classes=2,
        max_len=128,
        batch_size=16,
        learning_rate=2e-5,
        epochs=3
    )

    classifier.train(train_texts, train_labels, val_texts, val_labels)
    preds = classifier.predict(["This is a new example text"])
    print(preds)

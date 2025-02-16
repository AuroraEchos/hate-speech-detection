import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item 

    def __len__(self):
        return len(self.labels)
    
class DatasetProcessor:
    def __init__(self, tokenizer_name, max_length=128, batch_size=64):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.batch_size = batch_size

    def load_data(self, test_file):
        test_df = pd.read_csv(test_file)
        test_texts = test_df['TEXT'].tolist()
        test_labels = test_df['label'].tolist()
        return test_texts, test_labels
    
    def encode(self, texts):
        return self.tokenizer(texts, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
    
    def create_dataset(self, texts, labels):
        encodings = self.encode(texts)
        return HateSpeechDataset(encodings, labels)
    
    def create_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    
class RoBertFusion(nn.Module):
    def __init__(self, Robert_model, gru_hidden_size=128, num_filters=100, kernel_sizes=[3, 4, 5]):
        super(RoBertFusion, self).__init__()
        self.bert = BertModel.from_pretrained(Robert_model)
        
        # BiGRU part
        self.gru = nn.GRU(self.bert.config.hidden_size, gru_hidden_size, bidirectional=True, batch_first=True)
        
        # TextCNN part: list of convolutional layers with different kernel sizes
        self.textcnn = nn.ModuleList([nn.Conv2d(1, num_filters, (k, self.bert.config.hidden_size)) for k in kernel_sizes])

        self.fc = nn.Linear(gru_hidden_size * 2 + num_filters * len(kernel_sizes), 2)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        # BiGRU part
        gru_outputs, _ = self.gru(last_hidden_state)
        gru_outputs = gru_outputs[:, -1, :]

        # TextCNN part
        x = last_hidden_state.unsqueeze(1)
        cnn_outputs = [torch.relu(conv(x)).squeeze(3) for conv in self.textcnn]
        cnn_outputs = [torch.max(out, dim=2)[0] for out in cnn_outputs]
        cnn_outputs = torch.cat(cnn_outputs, dim=1)

        # Concatenate BiGRU and TextCNN outputs
        combined_features = torch.cat((gru_outputs, cnn_outputs), dim=1)

        logits = self.fc(combined_features)

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits
    

def evaluate(model, dataloader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels_batch = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels_batch)
            logits = outputs[1]

            pred = torch.argmax(logits, dim=1)

            preds.extend(pred.cpu().numpy())
            labels.extend(labels_batch.cpu().numpy())

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")

def test(model_path, test_file, model_class, tokenizer_name, device='cuda'):
    device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    processor = DatasetProcessor(tokenizer_name=tokenizer_name)
    test_texts, test_labels = processor.load_data(test_file)

    test_dataset = processor.create_dataset(test_texts, test_labels)
    test_loader = processor.create_dataloader(test_dataset)

    model = model_class(tokenizer_name).to(device)
    model.load_state_dict(torch.load(model_path))
    
    print('Evaluating on test data...')
    evaluate(model, test_loader, device)

if __name__ == '__main__':
    model_path = 'best_model.pth'
    test_file = 'Data/test.csv'
    model_class = RoBertFusion
    tokenizer_name = './chinese_roberta_wwm_ext'

    test(model_path, test_file, model_class, tokenizer_name)
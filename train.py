import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import argparse
from torch.optim import AdamW
from tqdm import tqdm

class HateSpeechDataset(Dataset):
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
    def __init__(self,  tokenizer_name, max_length=128, batch_size=64):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.batch_size = batch_size

    def load_data(self, train_file, dev_file, test_file):
        train_df = pd.read_csv(train_file)
        dev_df = pd.read_csv(dev_file)
        test_df = pd.read_csv(test_file)

        train_texts = train_df['TEXT'].tolist()
        train_labels = train_df['label'].tolist()

        dev_texts = dev_df['TEXT'].tolist()
        dev_labels = dev_df['label'].tolist()

        test_texts = test_df['TEXT'].tolist()
        test_labels = test_df['label'].tolist()

        return (train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels)
    
    def encode(self, texts):
        return self.tokenizer(texts, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')
    
    def create_dataset(self, texts, labels):
        encodings = self.encode(texts)
        return HateSpeechDataset(encodings, labels)
    
    def create_dataloader(self, dataset):
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

class RoBertFusion(nn.Module):
    def __init__(self, Robert_model, gru_hidden_size=128, num_filters=100, kernel_sizes=[3, 4, 5]):
        super(RoBertFusion, self).__init__()
        self.bert = BertModel.from_pretrained(Robert_model)
        
        self.gru = nn.GRU(self.bert.config.hidden_size, 
                          gru_hidden_size, 
                          bidirectional=True, 
                          batch_first=True)
        
        self.textcnn = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, self.bert.config.hidden_size)) 
            for k in kernel_sizes
        ])

        self.fc = nn.Linear(gru_hidden_size * 2 + num_filters * len(kernel_sizes), 2)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state

        gru_outputs, _ = self.gru(last_hidden_state)
        gru_outputs = gru_outputs[:, -1, :]

        x = last_hidden_state.unsqueeze(1)
        cnn_outputs = [torch.relu(conv(x)).squeeze(3) for conv in self.textcnn]
        cnn_outputs = [torch.max(out, dim=2)[0] for out in cnn_outputs]
        cnn_outputs = torch.cat(cnn_outputs, dim=1)

        combined_features = torch.cat((gru_outputs, cnn_outputs), dim=1)

        logits = self.fc(combined_features)
        
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            return loss, logits
        return logits

def evaluate(model, dataloader, device):
    model.eval()
    dev_preds, dev_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            logits = outputs[1]

            preds = torch.argmax(logits, dim=1)

            dev_preds.extend(preds.cpu().numpy())
            dev_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(dev_labels, dev_preds)
    print(f"Validation Accuracy: {acc * 100:.2f}%")
    return acc

def train(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Using device: {device}')

    processor = DatasetProcessor(tokenizer_name=args.model_path, max_length=args.max_length, batch_size=args.batch_size)
    (train_texts, train_labels), (dev_texts, dev_labels), (test_texts, test_labels) = processor.load_data(args.train_path, args.dev_path, args.test_path)
    train_dataset = processor.create_dataset(train_texts, train_labels)
    dev_dataset = processor.create_dataset(dev_texts, dev_labels)
    test_dataset = processor.create_dataset(test_texts, test_labels)
    train_loader = processor.create_dataloader(train_dataset)
    dev_loader = processor.create_dataloader(dev_dataset)
    test_loader = processor.create_dataloader(test_dataset)
    print(f'Training samples: {len(train_loader.dataset)}')
    print(f'Development samples: {len(dev_loader.dataset)}')
    print(f'Test samples: {len(test_loader.dataset)}')

    model = RoBertFusion(args.model_path).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-5)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    scaler = torch.amp.GradScaler() 

    print('Training...')
    best_acc = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        train_loader_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}",  leave=False)
        for batch in train_loader_pbar:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs[0]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() 

            avg_loss = total_loss / (len(train_loader) * (epoch + 1))
            train_loader_pbar.set_postfix({ 
                'Loss': f'{avg_loss:.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}' 
            })

        scheduler.step()
        
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")
        acc = evaluate(model, dev_loader, device)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')
            print('Model saved!')

    print('Testing...')
    model.load_state_dict(torch.load('best_model.pth'))
    evaluate(model, test_loader, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hate Speech Detection')
    parser.add_argument('--model_path', type=str, default='./chinese_roberta_wwm_ext', help='pretrained model path')
    parser.add_argument('--max_length', type=int, default=128, help='max length of input sequence')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--train_path', type=str, default='Data/train.csv', help='train file path')
    parser.add_argument('--dev_path', type=str, default='Data/dev.csv', help='dev file path')
    parser.add_argument('--test_path', type=str, default='Data/test.csv', help='test file path')
    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs')
    args = parser.parse_args()
    train(args)
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import argparse

class RoBertFusion(nn.Module):
    def __init__(self, Robert_model, gru_hidden_size=128, num_filters=100, kernel_sizes=[3, 4, 5]):
        super(RoBertFusion, self).__init__()
        self.bert = BertModel.from_pretrained(Robert_model)
        
        self.gru = nn.GRU(self.bert.config.hidden_size, gru_hidden_size, bidirectional=True, batch_first=True)
        
        self.textcnn = nn.ModuleList([nn.Conv2d(1, num_filters, (k, self.bert.config.hidden_size)) for k in kernel_sizes])

        self.fc = nn.Linear(gru_hidden_size * 2 + num_filters * len(kernel_sizes), 2)

    def forward(self, input_ids, attention_mask):
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

        return logits

class DatasetProcessor:
    def __init__(self, tokenizer_name, max_length=128):
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

    def encode(self, text):
        return self.tokenizer(text, max_length=self.max_length, padding=True, truncation=True, return_tensors='pt')

    def preprocess_input(self, text):
        encoding = self.encode(text)
        return encoding['input_ids'].squeeze(0), encoding['attention_mask'].squeeze(0)
    
def inference(model, tokenizer_name, text, device='cuda'):
    model.eval()
    processor = DatasetProcessor(tokenizer_name)
    input_ids, attention_mask = processor.preprocess_input(text)
    
    input_ids = input_ids.unsqueeze(0).to(device)
    attention_mask = attention_mask.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs
        pred = torch.argmax(logits, dim=1).item()
    
    return pred

def load_model(model_path, tokenizer_name, device='cuda'):
    device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
    model = RoBertFusion(tokenizer_name).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hate Speech Detection Inference')
    parser.add_argument('--model_path', type=str, required=True, help='path to the trained model')
    parser.add_argument('--tokenizer_name', type=str, required=True, help='path to the pretrained tokenizer')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='device to run the model on')
    args = parser.parse_args()

    model = load_model(args.model_path, args.tokenizer_name, args.device)

    input_text = input("Enter a sentence for prediction: ")

    result = inference(model, args.tokenizer_name, input_text, args.device)
    
    if result == 1:
        print("This sentence is classified as Hate Speech.")
    else:
        print("This sentence is classified as Not Hate Speech.")
import torch
import torch.nn as nn
from transformers import BertModel

class RoBertFusion(nn.Module):
    def __init__(self, Robert_model, gru_hidden_size=128, num_filters=100, kernel_sizes=[3, 4, 5]):
        super(RoBertFusion, self).__init__()
        self.bert = BertModel.from_pretrained(Robert_model)
        
        # BiGRU part
        self.gru = nn.GRU(self.bert.config.hidden_size, 
                          gru_hidden_size, 
                          bidirectional=True, 
                          batch_first=True)
        
        # TextCNN part: list of convolutional layers with different kernel sizes
        self.textcnn = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, self.bert.config.hidden_size)) 
            for k in kernel_sizes
        ])

        self.fc = nn.Linear(gru_hidden_size * 2 + num_filters * len(kernel_sizes), 2)

    def forward(self, input_ids, attention_mask):
        # BERT encoding
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]

        # BiGRU part
        gru_outputs, _ = self.gru(last_hidden_state)  # Shape: [batch_size, seq_len, 2*gru_hidden_size]
        print(gru_outputs.shape)
        gru_outputs = gru_outputs[:, -1, :]
        print(gru_outputs.shape)

        # TextCNN part
        x = last_hidden_state.unsqueeze(1)  # [batch_size, 1, seq_length, hidden_size]
        cnn_outputs = [torch.relu(conv(x)).squeeze(3) for conv in self.textcnn]
        cnn_outputs = [torch.max(out, dim=2)[0] for out in cnn_outputs]
        cnn_outputs = torch.cat(cnn_outputs, dim=1)
        print(cnn_outputs.shape)

        # Concatenate BiGRU and TextCNN outputs
        combined_features = torch.cat((gru_outputs, cnn_outputs), dim=1)  # Shape: [batch_size, 2*gru_hidden_size + num_filters * len(kernel_sizes)]

        logits = self.fc(combined_features)  # Shape: [batch_size, 2]
        
        return logits


if __name__ == '__main__':
    model = RoBertFusion('./chinese_roberta_wwm_ext')
    print(model)

    # Simulate data to test the model
    input_ids = torch.randint(0, 1000, (64, 128))  # batch_size=64, seq_len=128
    attention_mask = torch.ones(64, 128)  # Batch size x Sequence length
    logits = model(input_ids, attention_mask)
    print(logits.shape)  # Expected output shape: [64, 2] for binary classification

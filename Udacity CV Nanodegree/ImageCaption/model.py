import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batchnorm = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.batchnorm(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions):
        
        inputs = torch.cat((features.unsqueeze(1), self.embedding(captions[:, :-1])), dim = 1)
        output, _ = self.lstm(inputs)
        output = self.linear(output)
        
        return output

    def sample(self, inputs, stop_idx = 1, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        lstm_state = states
        for _ in range(max_len):
            lstm_out, lstm_state = self.lstm(inputs, lstm_state)
            linear_out = self.linear(lstm_out)
            
            prediction = torch.argmax(linear_out, dim = 2)
            prediction_index = prediction.item()
            sentence.append(prediction_index)
            
            if prediction_index == stop_idx:
                return sentence
            
            inputs = self.embedding(prediction)
            
        return sentence
            
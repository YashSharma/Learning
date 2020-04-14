import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        
    def forward(self, images):
        features = self.resnet(images)
        feautre = features.permute(0, 2, 3 ,1)
        features = features.view(features.size(0), -1, features.size(1))
        return features
    
class Attention(nn.Module):
    def __init__(self, encoder_dim, hidden_size, attention_dim):
        super().__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(hidden_size, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, encoder_out, decoder_hidden):
        encoder_output = self.encoder_att(encoder_out)
        decoder_output = self.decoder_att(decoder_hidden)
        att = self.full_att(encoder_output + decoder_output.unsqueeze(1)).squeeze(2)
        alpha = self.softmax(att)
        att_weight = (encoder_out * alpha.unsqueeze(2)).sum(dim = 1)
        
        return att_weight, alpha
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim, encoder_dim=2048, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size, num_layers)
        self.deep_output = nn.Linear(hidden_size, vocab_size)
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        self.f_beta = nn.Linear(hidden_size, encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.vocab_size = vocab_size
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
     
    def init_lstm_hidden(self, img_feat):
        
        avg_feat = img_feat.mean(dim = 1)
        c = self.init_c(avg_feat)
        h = self.init_h(avg_feat)
        
        return h, c
        
    def forward(self, img_feat, captions):
        
        batch_size = img_feat.size(0)
        h, c = self.init_lstm_hidden(img_feat)
        embedding = self.embedding(captions)
        caption_length = len(captions[0])
        
        preds = torch.zeros(batch_size, caption_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, caption_length, img_feat.size(1)).to(device)
        
        for t in range(caption_length):
            context, alpha = self.attention(img_feat, h)
            gate = self.sigmoid(self.f_beta(h))
            gate_context = gate*context
            lstm_input = torch.cat((embedding[:, t], gate_context), dim = 1)
            h, c = self.lstm(lstm_input, (h,c))
            output = self.deep_output(h)
            preds[:, t] = output
            alphas[:, t] = alpha
            
        return preds, alphas

    def sample(self, img_features, beam_size = 3, stop_idx = 1):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        img_features = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
        prev_words = torch.zeros(beam_size, 1).long().to(device)
        sentences = prev_words
        top_preds = torch.zeros(beam_size, 1).to(device)
        alphas = torch.ones(beam_size, 1, img_features.size(1)).to(device)

        completed_sentences = []
        completed_sentences_alphas = []
        completed_sentences_preds = []

        step = 1
        h, c = self.init_lstm_hidden(img_features)

        while True:
            embedding = self.embedding(prev_words).squeeze(1)
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(h)
            output = F.log_softmax(output, dim=1)
            output = top_preds.expand_as(output) + output
            
            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
                
            prev_word_idxs = top_words / output.size(1)
            next_word_idxs = top_words % output.size(1)

            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

            incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != stop_idx]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))

            if len(complete) > 0:
                completed_sentences.extend(sentences[complete].tolist())
                completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            alphas = alphas[incomplete]
            h = h[prev_word_idxs[incomplete]]
            c = c[prev_word_idxs[incomplete]]
            img_features = img_features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = next_word_idxs[incomplete].unsqueeze(1)

            if step > 50:
                break
            step += 1

        idx = completed_sentences_preds.index(max(completed_sentences_preds))
        sentence = completed_sentences[idx]
        alpha = completed_sentences_alphas[idx]
        return sentence, alpha  
import ipdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pathlib
import os


class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1, rnn_type='LSTM', bidirectional=True):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = True
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = hidden_size // self.num_directions
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True
        )

    def forward(self, input, input_lengths):
        packed = torch.nn.utils.rnn.pack_padded_sequence(input=input, lengths=input_lengths, batch_first=True, enforce_sorted=False)
        output, hidden = self.rnn(packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(sequence=output, batch_first=True)
        if self.bidirectional:
            new_hidden = []
            for h in hidden:
                a = h[0:h.shape[0]:2]
                b = h[1:h.shape[0]:2]
                c = torch.cat([a, b], dim=2)
                new_hidden.append(c)

            hidden = tuple(new_hidden)

        return output, hidden


class LuongAttn(nn.Module):
    # https://arxiv.org/pdf/1508.04025.pdf
    def __init__(self, method, hidden_size):
        super(LuongAttn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_seq_len = encoder_outputs.shape[1]
        batch_size = encoder_outputs.shape[0]
        a = Variable(torch.zeros(batch_size, max_seq_len, 1))
        a = a.cuda()
        attn_scores = torch.baddbmm(a, encoder_outputs, hidden).squeeze(2)
        attn_scores = F.softmax(attn_scores, dim=1).unsqueeze(1)
        return attn_scores

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.squeeze(0).dot(encoder_output.squeeze(0))
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.squeeze(0).dot(energy.squeeze(0))
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), dim=1))
            energy = self.v.squeeze(0).dot(energy.squeeze(0))
        return energy


class Decoder(nn.Module):
    def __init__(self, hidden_size, target_size, attn_model='concat', n_layers=1, dropout=0.1, rnn_type='GRU'):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.target_size = target_size
        self.attn = LuongAttn(attn_model, hidden_size)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.n_layers = n_layers
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, self.rnn_type)(
            input_size=target_size,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            dropout=self.dropout,
            bidirectional=False)
        self.out = nn.Linear(self.hidden_size, self.target_size)

    def forward(self, encoder_every_step_hidden_state_list, encoder_hidden_state_and_cell_state_tuple, target_tensor, target_length):
        batch_size, max_seq_length, target_mfcc_dim = target_tensor.shape

        decoder_outputs = torch.zeros(max_seq_length, batch_size, target_mfcc_dim).cuda()

        target_tensor = target_tensor.transpose(0, 1)

        last_step_output = torch.zeros(batch_size, target_mfcc_dim).float().cuda()
        last_step_hidden_tuple = encoder_hidden_state_and_cell_state_tuple
        for i in range(max_seq_length):
            decoder_step_output, last_step_hidden_tuple = self.forward_step(last_step_output, last_step_hidden_tuple, encoder_every_step_hidden_state_list)
            decoder_outputs[i] = decoder_step_output
            # teacher forcing
            real_step_out = target_tensor[i]
            last_step_output = real_step_out

        decoder_outputs = decoder_outputs.transpose(0, 1)

        target_length_array = target_length.detach().numpy()
        for i in range(len(target_length)):
            vec = decoder_outputs[i]
            real_length = target_length_array[i]
            mask_length = vec.shape[0] - real_length
            if mask_length == 0:
                continue
            vec[real_length:] = torch.zeros(mask_length, vec.shape[1]).cuda()
        return decoder_outputs

    def forward_step(self, cur_step_input, hidden_state_and_cell_state_tuple, encoder_every_step_hidden_state_list):
        cur_step_input = cur_step_input.unsqueeze(0)
        rnn_output, new_hidden_state_and_cell_state_tuple = self.rnn(cur_step_input, hidden_state_and_cell_state_tuple)
        # attention：
        attn_weights = self.attn(rnn_output.squeeze(0).unsqueeze(2), encoder_every_step_hidden_state_list)
        context = attn_weights.bmm(encoder_every_step_hidden_state_list)
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), dim=1)
        concat_output = torch.tanh(self.concat(concat_input))
        output = self.out(concat_output)
        return output, new_hidden_state_and_cell_state_tuple


class ModelWrapper():
    def __init__(self, mfcc_dim, rnn_hidden_size, lr, optimizer="sgd"):
        rnn_type = "LSTM"
        self.encoder = Encoder(input_size=mfcc_dim, hidden_size=rnn_hidden_size, n_layers=1, dropout=0, rnn_type=rnn_type, bidirectional=True).cuda()
        self.decoder = Decoder(hidden_size=rnn_hidden_size, target_size=mfcc_dim, n_layers=1, dropout=0, rnn_type=rnn_type).cuda()

        if optimizer == "sgd":
            self.encoder_optimizer = torch.optim.SGD(self.encoder.parameters(), lr=lr, momentum=0)
            self.decoder_optimizer = torch.optim.SGD(self.decoder.parameters(), lr=lr, momentum=0)
        else:
            assert optimizer == "adam"
            self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)
            self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr)

        self.criterion = nn.MSELoss()

    def train(self):
        self.encoder.train()
        self.decoder.train()

    def eval(self):
        self.encoder.eval()
        self.decoder.eval()

    def train_step(self, input_tensor, input_lengths, target_tensor, target_length):
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        # enc:
        encoder_every_step_hidden_state_list, encoder_hidden_state_and_cell_state_tuple = self.encoder(input_tensor, input_lengths)
        # dec:
        decoder_outputs = self.decoder(encoder_every_step_hidden_state_list, encoder_hidden_state_and_cell_state_tuple, target_tensor, target_length)
        # loss:
        loss_tensor = self.criterion(decoder_outputs, target_tensor)
        loss_tensor.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        return encoder_hidden_state_and_cell_state_tuple, loss_tensor.item()

    def save_model(self, epoch, file_save_path):
        mk_parent_dir_if_necessary(file_save_path)
        torch.save(obj={
            'epoch': epoch,
            'encoder_dict': self.encoder.state_dict(),
            'decoder_dict': self.decoder.state_dict(),
            'encoder_opt': self.encoder_optimizer.state_dict(),
            'decoder_opt': self.decoder_optimizer.state_dict()},
            f=file_save_path)

    def load_model(self, resume_path):
        checkpoint = torch.load(resume_path)
        start_epoch = checkpoint['epoch'] + 1
        self.encoder.load_state_dict = checkpoint['encoder_dict']
        self.decoder.load_state_dict(checkpoint['decoder_dict'])
        self.encoder_optimizer.load_state_dict(checkpoint['encoder_opt'])
        self.decoder_optimizer.load_state_dict(checkpoint['decoder_opt'])
        print("resume from:", resume_path)
        return start_epoch


def mk_parent_dir_if_necessary(img_save_path):
    folder = pathlib.Path(img_save_path).parent
    if not os.path.exists(folder):
        os.makedirs(folder)

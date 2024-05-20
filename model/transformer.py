import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def padding_mask(self, input):
        input = input.to(self.device)
        input_mask = (input != 0).unsqueeze(1).unsqueeze(2).to(self.device)
        return input_mask

    def target_mask(self, target):
        target = target.to(self.device)
        target_pad_mask = (target != 0).unsqueeze(1).unsqueeze(2).to(self.device)  # shape(batch_size, 1, 1, seq_length)
        target_sub_mask = torch.tril(torch.ones((target.shape[1], target.shape[1]), device=self.device)).bool()  # shape(seq_len, seq_len)
        target_mask = target_pad_mask & target_sub_mask  # shape(batch_size, 1, seq_length, seq_length)
        return target_mask

    def forward(self, input, target):
        input = input.to(self.device)
        target = target.to(self.device)

        input_mask = self.padding_mask(input)
        target_mask = self.target_mask(target)

        # encoder feed through
        encoded_input = self.encoder(input, input_mask)

        # decoder feed through
        output = self.decoder(target, encoded_input, input_mask, target_mask)

        return output

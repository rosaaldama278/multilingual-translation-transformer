import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

class MyDataset(Dataset):
    def __init__(self, data):
        self.src_lang = data['src']
        self.tgt_lang = data['tgt']

    def __len__(self):
        return len(self.src_lang)

    def __getitem__(self, index):
        return  {
            'src': self.src_lang[index],
            'tgt': self.tgt_lang[index]
        }

def collate_fn(batch):
    src_batch = [item['src'] for item in batch]
    tgt_batch = [item['tgt'] for item in batch]

    #padding
    src_batch_padded = pad_sequence([torch.tensor(x, dtype=torch.long) for x in src_batch], batch_first=True, padding_value=0)
    tgt_batch_padded = pad_sequence([torch.tensor(x, dtype=torch.long) for x in tgt_batch], batch_first=True, padding_value=0)
    
    return {
        'src': src_batch_padded,
        'tgt': tgt_batch_padded,
    }

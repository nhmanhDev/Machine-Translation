import torch
import torch.nn as nn
import math
import re

# Define PositionalEncoding class
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Define TransformerModel class
class TransformerModel(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, nhead=2, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=512, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None):
        src_emb = self.positional_encoding(self.src_embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.tgt_embedding(tgt) * math.sqrt(self.d_model))
        output = self.transformer(
            src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.fc_out(output)

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, dtype=torch.bool), diagonal=1)
        return mask

# Hàm tiền xử lý câu
def preprocess_sentence(w):
    """
    Hàm làm sạch dữ liệu.

    Tham số:
      w: câu đầu vào

    Returns:
      Câu đã được làm sạch, có thêm <start> và <end> nhưng không lặp lại các từ nếu đầu vào đã có.
    """
    # Chuyển thành chữ thường và loại bỏ khoảng trắng dư thừa ở đầu, cuối
    w = w.lower().strip()

    # Nếu có xuất hiện thẻ <start> và <end> trong chuỗi, loại bỏ chúng đi.
    w = re.sub(r'<\s*start\s*>', '', w)
    w = re.sub(r'<\s*end\s*>', '', w)

    # Loại bỏ các ký tự đặc biệt & các dấu câu, chỉ giữ lại chữ, số và khoảng trắng
    w = re.sub(r"[^\w0-9 ]+", " ", w)

    # Rút gọn nhiều khoảng trắng thành 1 khoảng trắng
    w = re.sub(r"[\s]+", " ", w)

    # Xóa khoảng trắng đầu và cuối chuỗi sau khi xử lý
    w = w.strip()

    # Thêm <start> và <end> ở đầu và cuối chuỗi để model nhận biết vị trí bắt đầu và kết thúc dự đoán
    w = '<start> ' + w + ' <end>'

    return w

# Hàm dịch
def translate(model, src_sentence, inp_lang, targ_lang, max_len=50, device='cpu'):
    model.eval()
    src = preprocess_sentence(src_sentence)
    src = [inp_lang.word2idx.get(word, 0) for word in src.split()]
    src = torch.tensor([src], dtype=torch.long).to(device)
    tgt = torch.tensor([[targ_lang.word2idx['<start>']]], dtype=torch.long).to(device)
    output_sentence = []
    with torch.no_grad():
        for _ in range(max_len):
            tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
            src_key_padding_mask = (src == 0).to(device)
            tgt_key_padding_mask = (tgt == 0).to(device)
            output = model(
                src, tgt, tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )
            next_word_idx = output[:, -1, :].argmax(-1).item()
            if next_word_idx == targ_lang.word2idx['<end>']:
                break
            output_sentence.append(targ_lang.idx2word[next_word_idx])
            tgt = torch.cat([tgt, torch.tensor([[next_word_idx]], dtype=torch.long).to(device)], dim=1)
    return ' '.join(output_sentence)

# Lớp LanguageIndex để tạo từ điển từ vựng
class LanguageIndex:
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            self.vocab.update(phrase.split())
        self.vocab = sorted(self.vocab)

        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word

# Hàm tạo dataset
def create_dataset(input_path, target_path):
  """
  Hàm nhập dữ liệu từ đường dẫn, làm sạch và trả về các cặp câu Anh - Việt

  Tham số:
    input_path: đường dẫn dữ liệu đầu vào
    target_path: đường dẫn dữ liệu đầu ra

  Returns:
    List các cặp câu Anh - Việt tương ứng
  """

  input_lines = open(input_path, encoding='UTF-8').read().strip().split('\n')
  target_lines = open(target_path, encoding='UTF-8').read().strip().split('\n')

  word_pairs = []
  for i in range(len(input_lines)):
    pi = preprocess_sentence(input_lines[i])
    pt = preprocess_sentence(target_lines[i])
    word_pairs.append([pi, pt])

  return word_pairs
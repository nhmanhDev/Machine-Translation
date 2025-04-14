from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import torch
from model import TransformerModel, translate, LanguageIndex, create_dataset
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.transformer")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Đường dẫn đến file mô hình
MODEL_PATH = 'transformer_model.pth'  # Thay bằng đường dẫn thực tế của bạn

# Thiết bị
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tải từ điển từ vựng
def load_vocabularies(train_en_path, train_vi_path):
    pairs = create_dataset(train_en_path, train_vi_path)
    inp_lang = LanguageIndex(en for en, vi in pairs)
    targ_lang = LanguageIndex(vi for en, vi in pairs)
    return inp_lang, targ_lang

# Thay đường dẫn thực tế của bạn vào đây
inp_lang, targ_lang = load_vocabularies('data/train.en', 'data/train.vi')

# Tải mô hình
model = TransformerModel(
    src_vocab_size=len(inp_lang.vocab) + 1,
    tgt_vocab_size=len(targ_lang.vocab) + 1,
    d_model=128,
    nhead=2,
    num_encoder_layers=3,
    num_decoder_layers=3,
    dim_feedforward=512,
    dropout=0.1
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()

# Cấu hình templates
templates = Jinja2Templates(directory="static")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "sentence": "", "translated": ""})

@app.post("/translate")
async def translate_sentence(request: Request):
    form = await request.form()
    sentence = form["sentence"]
    translated = translate(model, sentence, inp_lang, targ_lang, device=DEVICE)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "sentence": sentence, "translated": translated}
    )
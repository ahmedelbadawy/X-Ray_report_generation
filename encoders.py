""" build the ALBEF retrieval model and load CXR weights.
"""
import yaml
import torch
from transformers import BertTokenizer
from models.model_retrieval import ALBEF
import torch
import torch.nn.functional as F
# from PIL import Image
from torchvision import transforms

def load_model(config_path, checkpoint_path, device):
    config = yaml.safe_load(open(config_path))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    model = ALBEF(text_encoder='bert-base-uncased', tokenizer=tokenizer, config=config)

    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['model'] if 'model' in ckpt else ckpt
    state_dict = {k.replace('text_encoder.bert.', 'text_encoder.'): v
              for k, v in state_dict.items()}
    state_dict = {k.replace('text_encoder_m.bert.', 'text_encoder_m.'): v
              for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)

    # sanity: the four things we actually use must NOT be in missing_keys
    must_have = ('visual_encoder', 'vision_proj', 'text_encoder', 'text_proj')
    missing_critical = [k for k in msg.missing_keys
                        if any(k.startswith(p) for p in must_have)]
    if missing_critical:
        raise RuntimeError(f"Checkpoint missing required weights: {missing_critical[:5]}")

    model = model.to(device).eval()
    return model, tokenizer, config

def build_image_transform(image_res):
    return transforms.Compose([
        transforms.Resize((image_res, image_res)),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                             (0.26862954, 0.26130258, 0.27577711)),
    ])


# @torch.no_grad()
# def image_vector(model, img_transform, paths, device):
#     """One or many image paths -> [N, 256] tensor."""
#     if isinstance(paths, str):
#         paths = [paths]
#     imgs = torch.stack([img_transform(Image.open(p).convert('RGB')) for p in paths]).to(device)
#     embeds = model.visual_encoder(imgs)                       # [N, patches+1, 768]
#     feat = F.normalize(model.vision_proj(embeds[:, 0, :]), dim=-1)
#     return feat.cpu()                                         # [N, 256]


@torch.no_grad()
def text_vector(model, tokenizer, texts, device, max_length=40):
    """One or many impression strings -> [N, 256] tensor."""
    if isinstance(texts, str):
        texts = [texts]
    tok = tokenizer(texts, padding='max_length', truncation=True,
                    max_length=max_length, return_tensors='pt').to(device)
    out = model.text_encoder(tok.input_ids, attention_mask=tok.attention_mask,
                             mode='text', return_dict=True)    # text alone, no image
    feat = F.normalize(model.text_proj(out.last_hidden_state[:, 0, :]), dim=-1)
    return feat.cpu()       
# modified from https://github.com/kha-white/manga-ocr/blob/master/manga_ocr/ocr.py
import os
import re
import sys
import jaconv
from transformers import AutoFeatureExtractor, AutoTokenizer, VisionEncoderDecoderModel
import numpy as np
import torch

MANGA_OCR_PATH = r'data/models/manga-ocr-base'
MANGA_OCR_2025_PATH = r'data/models/manga-ocr-2025-onnx'


_TORCH_DLL_DIR_ADDED = False


def _ensure_torch_dll_dir():
    """Add PyTorch's bundled CUDA/cuDNN DLLs to the search path so ORT can find them.

    ORT-GPU 1.17.x on Windows needs CUDA 11.8 + cuDNN 8.x DLLs. The portable
    BallonsTranslator distribution doesn't ship a system-wide CUDA install,
    but the bundled PyTorch (cu118 build) carries them under torch/lib.
    Calling os.add_dll_directory once is enough — it persists per-process.
    """
    global _TORCH_DLL_DIR_ADDED
    if _TORCH_DLL_DIR_ADDED or sys.platform != 'win32':
        return
    torch_lib = os.path.join(os.path.dirname(torch.__file__), 'lib')
    if os.path.isdir(torch_lib):
        try:
            os.add_dll_directory(torch_lib)
        except (AttributeError, OSError):
            pass
    _TORCH_DLL_DIR_ADDED = True
class MangaOcr:
    def __init__(self, pretrained_model_name_or_path=MANGA_OCR_PATH, device='cpu'):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path)
        self.to(device)
        
    def to(self, device):
        self.model.to(device)

    @torch.no_grad()
    def __call__(self, img: np.ndarray):
        x = self.feature_extractor(img, return_tensors="pt").pixel_values.squeeze()
        x = self.model.generate(x[None].to(self.model.device))[0].cpu()
        x = self.tokenizer.decode(x, skip_special_tokens=True)
        x = post_process(x)
        return x

    # todo
    def ocr_batch(self, im_batch: torch.Tensor):
        raise NotImplementedError


def post_process(text):
    text = ''.join(text.split())
    text = text.replace('…', '...')
    text = re.sub('[・.]{2,}', lambda x: (x.end() - x.start()) * '.', text)
    text = jaconv.h2z(text, ascii=True, digit=True)

    return text


class MangaOcr2025:
    """ONNX manga-ocr (l0wgear/manga-ocr-2025-onnx).

    2025 retrained VisionEncoderDecoder, exported to ONNX with HF Optimum.
    Same input/output as the original MangaOcr (numpy RGB image -> Japanese
    text string), but inference goes through ORT instead of PyTorch.

    device='cuda' uses CUDAExecutionProvider (PyTorch's bundled CUDA 11.8 +
    cuDNN 8 satisfy the runtime via _ensure_torch_dll_dir). device='cpu'
    uses CPUExecutionProvider — still under 1s per balloon on this size.
    """

    def __init__(self, pretrained_model_name_or_path=MANGA_OCR_2025_PATH, device='cpu'):
        # Critical: add torch's bundled CUDA / cuDNN DLL directory BEFORE
        # any `import onnxruntime` (which optimum triggers transitively).
        # onnxruntime-gpu's `onnxruntime_pybind11_state.pyd` will fail to
        # initialize ("DLL initialization routine failed") when the host
        # process already has another CUDA-using extension loaded (torch,
        # cv2-CUDA, etc.) unless the matching CUDA runtime DLLs are on
        # the search path. We do this unconditionally — even CPU sessions
        # need the pybind state to import cleanly.
        _ensure_torch_dll_dir()

        from PIL import Image  # noqa: F401  (sanity-check Pillow is present)
        from transformers import TrOCRProcessor
        from optimum.onnxruntime import ORTModelForVision2Seq

        self.device = device
        if device == 'cuda':
            provider = 'CUDAExecutionProvider'
        else:
            provider = 'CPUExecutionProvider'

        self.processor = TrOCRProcessor.from_pretrained(pretrained_model_name_or_path)
        # use_cache=False — the HF repo only ships encoder_model.onnx +
        # decoder_model.onnx, no decoder_with_past_model.onnx. Disabling
        # KV-cache makes ORT skip looking for the past-decoder file.
        # Per-balloon outputs are short (a single text line) so the
        # throughput penalty is negligible.
        self.model = ORTModelForVision2Seq.from_pretrained(
            pretrained_model_name_or_path,
            provider=provider,
            use_cache=False,
            use_io_binding=False,
        )

    def to(self, device):
        # ORT models can't switch provider in place — rebuild.
        if device == self.device:
            return
        self.__init__(
            pretrained_model_name_or_path=getattr(
                self.model.config, '_name_or_path', MANGA_OCR_2025_PATH
            ),
            device=device,
        )

    def __call__(self, img: np.ndarray) -> str:
        from PIL import Image
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        pil_img = Image.fromarray(img).convert('RGB')
        pixel_values = self.processor(images=pil_img, return_tensors='pt').pixel_values
        # Match input tensor device to model.device. Without this, beam search
        # (num_beams=4 per generation_config.json) crashes with
        # "Expected all tensors to be on the same device" on CUDA because
        # ORT puts logits on cuda:0 but beam_scores end up on cpu.
        try:
            model_device = self.model.device
        except AttributeError:
            model_device = None
        if model_device is not None:
            pixel_values = pixel_values.to(model_device)
        generated_ids = self.model.generate(pixel_values)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return post_process(text)


if __name__ == '__main__':
    import cv2

    img_path = r'data/testpacks/textline/ballontranslator.png'
    manga_ocr = MangaOcr(pretrained_model_name_or_path=MANGA_OCR_PATH, device='cuda')

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    dummy = np.zeros((1024, 1024, 3), np.uint8)
    manga_ocr(dummy)
    # preprocessed = manga_ocr(img_path)

    # im_batch = 
    # img = (torch.from_numpy(img[np.newaxis, ...]).float() - 127.5) / 127.5
    # img = einops.rearrange(img, 'N H W C -> N C H W')
    import time
    
    for ii in range(10):
        t0 = time.time()
        out = manga_ocr(dummy)
        print(out, time.time() - t0)
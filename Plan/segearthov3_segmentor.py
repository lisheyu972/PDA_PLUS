from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


class SegEarthOV3Segmentation:
    def __init__(
        self,
        classname_path,
        device=None,
        prob_thd=0.0,
        bg_idx=0,
        slide_stride=0,
        slide_crop=0,
        confidence_threshold=0.5,
        use_sem_seg=True,
        use_presence_score=True,
        use_transformer_decoder=True,
        checkpoint_path=None,
        bpe_path=None,
        **kwargs,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint_path = Path(checkpoint_path)
        bpe_path = Path(bpe_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint_path}")
        if not bpe_path.is_file():
            raise FileNotFoundError(f"SAM3 BPE vocabulary not found: {bpe_path}")
        model = build_sam3_image_model(
            bpe_path=str(bpe_path),
            checkpoint_path=str(checkpoint_path),
            device=str(self.device),
            load_from_HF=False,
        )
        self.processor = Sam3Processor(
            model,
            confidence_threshold=confidence_threshold,
            device=self.device,
        )
        self.query_words, query_idx = get_cls_idx(classname_path)
        self.num_cls = max(query_idx) + 1
        self.num_queries = len(query_idx)
        self.query_idx = torch.tensor(query_idx, dtype=torch.int64, device=self.device)
        self.prob_thd = prob_thd
        self.bg_idx = bg_idx
        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.use_sem_seg = use_sem_seg
        self.use_presence_score = use_presence_score
        self.use_transformer_decoder = use_transformer_decoder

    def _inference_single_view(self, image):
        width, height = image.size
        seg_logits = torch.zeros(
            (self.num_queries, height, width),
            device=self.device,
        )
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            inference_state = self.processor.set_image(image)
            for query_idx, query_word in enumerate(self.query_words):
                self.processor.reset_all_prompts(inference_state)
                inference_state = self.processor.set_text_prompt(
                    state=inference_state,
                    prompt=query_word,
                )
                if (
                    self.use_transformer_decoder
                    and inference_state["masks_logits"].shape[0] > 0
                ):
                    for inst_id in range(inference_state["masks_logits"].shape[0]):
                        instance_logits = inference_state["masks_logits"][
                            inst_id
                        ].squeeze()
                        instance_score = inference_state["object_score"][inst_id]
                        if instance_logits.shape != (height, width):
                            instance_logits = F.interpolate(
                                instance_logits.view(1, 1, *instance_logits.shape),
                                size=(height, width),
                                mode="bilinear",
                                align_corners=False,
                            ).squeeze()
                        seg_logits[query_idx] = torch.maximum(
                            seg_logits[query_idx],
                            instance_logits * instance_score,
                        )
                if self.use_sem_seg:
                    semantic_logits = inference_state["semantic_mask_logits"]
                    if semantic_logits.shape != (height, width):
                        semantic_logits = F.interpolate(
                            semantic_logits,
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False,
                        ).squeeze()
                    seg_logits[query_idx] = torch.maximum(
                        seg_logits[query_idx],
                        semantic_logits,
                    )
                if self.use_presence_score:
                    seg_logits[query_idx] *= inference_state["presence_score"]
        return seg_logits

    def slide_inference(self, image, stride, crop_size):
        width, height = image.size
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        height_stride, width_stride = stride
        height_crop, width_crop = crop_size
        predictions = torch.zeros(
            (self.num_queries, height, width),
            device=self.device,
        )
        counts = torch.zeros((1, height, width), device=self.device)
        height_grids = (
            max(height - height_crop + height_stride - 1, 0) // height_stride + 1
        )
        width_grids = max(width - width_crop + width_stride - 1, 0) // width_stride + 1
        for height_idx in range(height_grids):
            for width_idx in range(width_grids):
                y1 = height_idx * height_stride
                x1 = width_idx * width_stride
                y2 = min(y1 + height_crop, height)
                x2 = min(x1 + width_crop, width)
                y1 = max(y2 - height_crop, 0)
                x1 = max(x2 - width_crop, 0)
                crop_logits = self._inference_single_view(image.crop((x1, y1, x2, y2)))
                predictions[:, y1:y2, x1:x2] += crop_logits
                counts[:, y1:y2, x1:x2] += 1
        if torch.any(counts == 0):
            raise RuntimeError("Sparse sliding-window coverage")
        return predictions / counts

    def predict(self, image):
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        width, height = image.size
        if self.slide_crop > 0 and (
            self.slide_crop < width or self.slide_crop < height
        ):
            seg_logits = self.slide_inference(
                image,
                self.slide_stride,
                self.slide_crop,
            )
        else:
            seg_logits = self._inference_single_view(image)
        if self.num_cls != self.num_queries:
            class_index = F.one_hot(
                self.query_idx,
                num_classes=self.num_cls,
            ).T.view(self.num_cls, self.num_queries, 1, 1)
            seg_logits = (seg_logits.unsqueeze(0) * class_index).amax(dim=1)
        seg_pred = torch.argmax(seg_logits, dim=0)
        max_values = seg_logits.amax(dim=0)
        seg_pred[max_values < self.prob_thd] = self.bg_idx
        return seg_pred.cpu().numpy()


def get_cls_idx(path):
    with open(path, "r", encoding="utf-8") as file:
        name_sets = file.readlines()
    class_names = []
    class_indices = []
    for index, name_set in enumerate(name_sets):
        names = [name.strip() for name in name_set.split(",") if name.strip()]
        class_names.extend(names)
        class_indices.extend([index] * len(names))
    return class_names, class_indices

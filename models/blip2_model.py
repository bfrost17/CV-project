import torch
from transformers import BlipForConditionalGeneration, BlipProcessor

class ImageCaptioner:
    def __init__(self, model_name="blip", device="cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.processor, self.model = self.initialize_model()
        
    def initialize_model(self):
        if self.model_name != "blip":
            raise NotImplementedError(f"{self.model_name} not implemented.")

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=dtype,
        )
        model.to(self.device)
        model.eval()
        return processor, model

    def image_caption(self, image, prompt, type_info):
        text_prompt = prompt.replace("[INST]", "").replace("[/INST]", "").replace("<image>", "").strip()
        inputs = self.processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.device)
        max_new_tokens = 50 if type_info in {"A", "B"} else 100
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        outputs = self.processor.decode(output[0], skip_special_tokens=True)
        outputs = outputs.strip()
        return outputs
    
    def image_caption_debug(self, image_src):
        return "A dish with salmon, broccoli, and something yellow."

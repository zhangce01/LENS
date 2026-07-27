import importlib


MODEL_MAP = {
    "llava_video":    ("models.llavavideo", "lmms-lab/LLaVA-Video-7B-Qwen2"),
    "qwenvl25_7b":    ("models.qwenvl", "Qwen/Qwen2.5-VL-7B-Instruct"),
    "qwenvl25_3b":    ("models.qwenvl", "Qwen/Qwen2.5-VL-3B-Instruct"),
    "qwenvl2_7b":     ("models.qwenvl", "Qwen/Qwen2-VL-7B-Instruct"),
    "qwenvl2_2b":     ("models.qwenvl", "Qwen/Qwen2-VL-2B-Instruct"),
    "internvl25_2b":  ("models.internvl", "OpenGVLab/InternVL2_5-2B"),
    "longvu":         ("models.longvu", "Vision-CAIR/LongVU_Qwen2_7B"),
}


class LENS():
    """Thin wrapper that dynamically loads an LVLM backend.

    Each backend module (see `models/`) exposes three functions:
        load_model(model_path)  -> (processor, video_llm, image_processor, _)
        load_video(video_path, args)
        mllm_response(video_llm, processor, image_processor, text, image_inputs, video, ...)
    To plug in a new LVLM, implement these three functions and register the
    module in MODEL_MAP.
    """

    def __init__(self, args):
        self.args = args
        module_name, model_path = next(
            ((module, model_path) for key, (module, model_path) in MODEL_MAP.items() if key in self.args.model_name),
            None
        )

        module = importlib.import_module(module_name)
        self.mllm_response, self.load_video, self.load_model = (
            module.mllm_response,
            module.load_video,
            module.load_model,
        )
        self.processor, self.video_llm, self.image_processor, _ = self.load_model(model_path)

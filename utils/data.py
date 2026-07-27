import random
import torch
import os
import json
from pyarrow import parquet as pq
import pysubs2
import re

def timestamp_to_seconds(timestamp):
    h, m, s = timestamp.split(':')
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds


class EvalDatasetLongVideoBench(torch.utils.data.IterableDataset):
    def __init__(
        self,
        data_path: str = "./data/LongVideoBench",
        split_id: int = 0,
        file_name: str = "lvb_val.json",
    ) -> None:
        super(EvalDatasetLongVideoBench, self).__init__()
        self.data_path = data_path
        data_list = json.load(open(os.path.join(self.data_path, file_name), "r"))
        list_data_dict = []

        for item in data_list:
            video_path = os.path.join(self.data_path, "videos", item["video_path"])
            letters = [chr(ord('A') + idx) for idx in range(len(item["candidates"]))]
            question = self.qa_template(item)
            subtitle_path = os.path.join(self.data_path, "subtitles", item["subtitle_path"])
            list_data_dict.append(
                {
                    "question": item["question"],
                    "prompt": question + "Respond with only the letter of the correct option.\n",
                    "video_path": video_path,
                    "video_id": item["id"],
                    "subtitle": subtitle_path,
                    "video_name": item["id"],
                    "duration": item["duration"],
                    "letters": letters,
                    "task_type": item['question_category'],
                    "candidates": item["candidates"],
                    "answer": chr(ord('A') + int(item["correct_choice"])),
                    "starting_timestamp_for_subtitles": item["starting_timestamp_for_subtitles"]
                }
            )

        self.data = list_data_dict
        
    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        for idx, c in enumerate(data["candidates"]):
            question += f"({chr(ord('A') + idx)}) {c}\n"
        return question

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        return self.data[i]

class EvalDatasetVideoMME(torch.utils.data.IterableDataset):
    """Dataset for supervised fine-tuning."""

    video_formats = [".mp4", ".avi", ".mov", ".mkv"]

    def __init__(
        self,
        data_path: str = "./data/videomme",
        file_name: str = "test-00000-of-00001.parquet",
    ) -> None:
        super(EvalDatasetVideoMME, self).__init__()

        self.data_path = data_path

        data_list = load_parquet(
            os.path.join(self.data_path, "videomme", file_name)
        )

        list_data_dict = []

        for item in data_list:
            video_ytid = item["url"].split("watch?v=")[-1]
            video_path = os.path.join(self.data_path, "data", f"{video_ytid}.mp4")
            for fmt in self.video_formats:  # Added this line
                temp_path = os.path.join(self.data_path, "data", f"{video_ytid}{fmt}")
                if os.path.exists(temp_path):
                    video_path = temp_path
                    break
            
            subtitle_path = os.path.join(
                self.data_path, "subtitle", f"{video_ytid}.srt"
            )

            for query in item["questions"]:
                prompt = f"Question: {query['question']}\n"
                prompt += "Options:\n"
                for op in query["candidates"]:
                    prompt += f"{op}\n"
                list_data_dict.append(
                    {
                        "question": query["question"],
                        "video_path": video_path,
                        "video_id": item["video_id"],
                        "subtitle": subtitle_path,
                        "video_name": video_ytid,
                        "duration": item["duration"],
                        "domain": item["domain"],
                        "sub_category": item["sub_category"],
                        "candidates": query["candidates"],
                        "answer": query["answer"],
                        "task_type": query["task_type"],
                        "letters": query["letters"],
                        "prompt": prompt,
                    }
                )

        self.data = list_data_dict

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        return self.data[i]

class EvalDatasetMLVU(torch.utils.data.IterableDataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str = "./data/mlvu",
    ) -> None:
        super(EvalDatasetMLVU, self).__init__()

        self.data_path = data_path

        data_list = {
            "count": ("json/4_count.json", f"video/4_count", "video"), 
            "ego": ("json/3_ego.json", f"video/3_ego", "video"), 
            "findNeedle": ("json/2_needle.json", f"video/2_needle", "video"), 
            "order": ("json/5_order.json", f"video/5_order", "video"), 
            "plotQA": ("json/1_plotQA.json", f"video/1_plotQA", "video"),
            "anomaly_reco": (
                "json/6_anomaly_reco.json",
                f"video/6_anomaly_reco",
                "video",
            ),
            "topic_reasoning": (
                "json/7_topic_reasoning.json",
                f"video/7_topic_reasoning",
                "video",
            ),
        }

        list_data_dict = []
        for k, v in data_list.items():
            with open(os.path.join(data_path, v[0]), "r") as f:
                json_data = json.load(f)
            for data in json_data:
                question, answer = self.qa_template(data)
                options = []
                for idx, candidate in enumerate(data["candidates"]):
                    options.append(f"{chr(ord('A') + idx)}. {candidate}")
                list_data_dict.append(
                    {
                        "task_type": k,
                        "video_path": os.path.join(self.data_path, v[1], data["video"]),
                        "video_name": data["video"],
                        "question": data["question"],
                        "candidates": options,
                        "prompt": question,
                        "answer": answer,
                        "duration": data["duration"],
                        "letters": ["A", "B", "C", "D"]
                    }
                )
        
        self.data = list_data_dict

    def __len__(self) -> int:
        return len(self.data)

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += "Options:\n"
        answer = data["answer"]
        answer_idx = -1
        for idx, c in enumerate(data["candidates"]):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        answer = f"{chr(ord('A') + answer_idx)}"
        return question, answer

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        return self.data[i]


class EvalDatasetEgoschemaSubset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        data_path: str = "./data/egoschema",
        file_name: str = "test-00000-of-00001.parquet",
        videos_subdir: str = "Subset",
    ) -> None:
        super(EvalDatasetEgoschemaSubset, self).__init__()
        self.data_path = data_path
        parquet_path = os.path.join(self.data_path, videos_subdir, file_name)
        df = pq.read_table(parquet_path).to_pandas()

        list_data_dict = []
        for item in df.to_dict("records"):
            options = list(item["option"])
            n = len(options)
            letters = [chr(ord("A") + i) for i in range(n)]
            answer_letter = self._answer_to_letter(item["answer"], n)
            prompt = self._qa_template(item["question"], options)
            vid = item["video_idx"]
            vid_str = vid if isinstance(vid, str) else str(vid)
            lower = vid_str.lower()
            if lower.endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_rel = vid_str
            else:
                video_rel = f"{vid_str}.mp4"
            video_path = os.path.join(self.data_path, "videos", video_rel)

            list_data_dict.append(
                {
                    "question_idx": item["question_idx"],
                    "video_idx": item["video_idx"],
                    "question": item["question"],
                    "candidates": options,
                    "letters": letters,
                    "prompt": prompt,
                    "answer": answer_letter,
                    "video_path": video_path,
                    "video_name": vid_str,
                    "task_type": "egoschema",
                    "subtitle": None,
                }
            )

        self.data = list_data_dict

    @staticmethod
    def _answer_to_letter(answer, num_options: int) -> str:
        if not isinstance(answer, bool) and hasattr(answer, "item"):
            try:
                answer = answer.item()
            except (ValueError, AttributeError):
                pass
        if isinstance(answer, float) and answer.is_integer():
            answer = int(answer)
        if isinstance(answer, int) and not isinstance(answer, bool):
            if 0 <= answer < num_options:
                return chr(ord("A") + answer)
        if isinstance(answer, str):
            s = answer.strip()
            if len(s) == 1 and s.isalpha():
                u = s.upper()
                if ord(u) - ord("A") < num_options:
                    return u
            if s.isdigit():
                i = int(s)
                if 0 <= i < num_options:
                    return chr(ord("A") + i)
        raise ValueError(
            f"Cannot map answer {answer!r} to A-{chr(ord('A') + num_options - 1)} for {num_options} options"
        )

    @staticmethod
    def _qa_template(question_text: str, options: list) -> str:
        question = f"Question: {question_text}\n"
        question += "Options:\n"
        for idx, c in enumerate(options):
            question += f"({chr(ord('A') + idx)}) {c}\n"
        return question

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        return self.data[i]


class EvalDatasetNExTQA(torch.utils.data.IterableDataset):
    """Hugging Face `lmms-lab/NExTQA` MC split — parquet like Egoschema, 5-way MCQ."""

    video_formats = [".mp4", ".avi", ".mov", ".mkv"]

    def __init__(
        self,
        data_path: str = "./data/nextqa",
        file_name: str = "test-00000-of-00001.parquet",
        parquet_subdir: str = "MC",
        videos_subdir: str = "NExTVideo",
        video_root: str = None,
    ) -> None:
        super(EvalDatasetNExTQA, self).__init__()
        self.data_path = data_path
        parquet_path = os.path.join(self.data_path, parquet_subdir, file_name)
        df = pq.read_table(parquet_path).to_pandas()

        vroot = video_root if video_root is not None else self.data_path
        list_data_dict = []

        for item in df.to_dict("records"):
            video_id_val = self._scalar_int(item["video"])
            vid_str = str(video_id_val)
            options = [str(item[f"a{i}"]) for i in range(5)]
            letters = [chr(ord("A") + i) for i in range(5)]
            answer_letter = self._answer_to_letter(item["answer"], 5)
            answer_idx = ord(answer_letter) - ord("A")
            qid = self._scalar_int(item["qid"])
            qtype = str(item["type"])
            qtext = str(item["question"])
            prompt = self._qa_template(qtext, options)
            fc = self._scalar_int(item["frame_count"])
            w = self._scalar_int(item["width"])
            h = self._scalar_int(item["height"])

            video_path = os.path.join(vroot, videos_subdir, f"{vid_str}.mp4")
            for fmt in self.video_formats:
                temp_path = os.path.join(vroot, videos_subdir, f"{vid_str}{fmt}")
                if os.path.exists(temp_path):
                    video_path = temp_path
                    break

            list_data_dict.append(
                {
                    "qid": qid,
                    "video": video_id_val,
                    "frame_count": fc,
                    "width": w,
                    "height": h,
                    "question": qtext,
                    "candidates": options,
                    "letters": letters,
                    "prompt": prompt,
                    "answer": answer_letter,
                    "answer_idx": answer_idx,
                    "type": qtype,
                    "a0": str(item["a0"]),
                    "a1": str(item["a1"]),
                    "a2": str(item["a2"]),
                    "a3": str(item["a3"]),
                    "a4": str(item["a4"]),
                    "video_path": video_path,
                    "video_name": vid_str,
                    "task_type": qtype,
                    "subtitle": None,
                }
            )

        self.data = list_data_dict

    @staticmethod
    def _scalar_int(x):
        if hasattr(x, "item"):
            try:
                x = x.item()
            except (ValueError, AttributeError):
                pass
        if isinstance(x, float) and x.is_integer():
            x = int(x)
        if isinstance(x, bool):
            raise TypeError(f"unexpected bool in numeric field: {x!r}")
        return int(x)

    @staticmethod
    def _answer_to_letter(answer, num_options: int) -> str:
        if not isinstance(answer, bool) and hasattr(answer, "item"):
            try:
                answer = answer.item()
            except (ValueError, AttributeError):
                pass
        if isinstance(answer, float) and answer.is_integer():
            answer = int(answer)
        if isinstance(answer, int) and not isinstance(answer, bool):
            if 0 <= answer < num_options:
                return chr(ord("A") + answer)
        if isinstance(answer, str):
            s = answer.strip()
            if len(s) == 1 and s.isalpha():
                u = s.upper()
                if ord(u) - ord("A") < num_options:
                    return u
            if s.isdigit():
                i = int(s)
                if 0 <= i < num_options:
                    return chr(ord("A") + i)
        raise ValueError(
            f"Cannot map answer {answer!r} to A-{chr(ord('A') + num_options - 1)} for {num_options} options"
        )

    @staticmethod
    def _qa_template(question_text: str, options: list) -> str:
        question = f"Question: {question_text}\n"
        question += "Options:\n"
        for idx, c in enumerate(options):
            question += f"({chr(ord('A') + idx)}) {c}\n"
        return question

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, i):
        return self.data[i]


def load_parquet(parquet_file):
    table = pq.read_table(parquet_file)

    df = table.to_pandas()

    jsons = []
    for record in df.itertuples():

        if len(jsons) < int(record.video_id):
            jsons.append(
                {
                    "video_id": record.video_id,
                    "youtube_id": record.videoID,
                    "url": record.url,
                    "duration": record.duration,
                    "domain": record.domain,
                    "sub_category": record.sub_category,
                    "questions": [
                        {
                            "question_id": record.question_id,
                            "task_type": record.task_type,
                            "question": record.question,
                            "candidates": list(record.options),
                            "answer": record.answer,
                            "letters": ["A", "B", "C", "D"]
                        }
                    ],
                }
            )
        else:
            jsons[-1]["questions"].append(
                {
                    "question_id": record.question_id,
                    "task_type": record.task_type,
                    "question": record.question,
                    "candidates": list(record.options),
                    "answer": record.answer,
                    "letters": ["A", "B", "C", "D"]
                }
            )

    return jsons

def clean_text(text):
    """Removes HTML-like tags and bracketed text from a string."""
    clean_html = re.compile('<.*?>')
    text = re.sub(clean_html, '', text)
    clean_brackets = re.compile(r'\[.*?\]')
    return re.sub(clean_brackets, '', text).strip()

def group_subtitles_by_interval(all_subtitles, interval_seconds=5):
    combined_subtitles = {}
    unique_combined_subtitles = set()
    
    if not all_subtitles:
        return {}

    sorted_times = sorted(all_subtitles.keys())
    
    current_group_start_time = (sorted_times[0] // interval_seconds) * interval_seconds
    current_group_text = []

    for time in sorted_times:
        if time < current_group_start_time + interval_seconds:
            if all_subtitles[time]:
                current_group_text.append(all_subtitles[time])
        else:
            if current_group_text:
                combined_text = ' '.join(current_group_text).strip()
                if combined_text and combined_text not in unique_combined_subtitles:
                    combined_subtitles[current_group_start_time] = combined_text
                    unique_combined_subtitles.add(combined_text)

            current_group_start_time = (time // interval_seconds) * interval_seconds
            current_group_text = []
            if all_subtitles[time]:
                current_group_text.append(all_subtitles[time])
    
    if current_group_text:
        combined_text = ' '.join(current_group_text).strip()
        if combined_text and combined_text not in unique_combined_subtitles:
            combined_subtitles[current_group_start_time] = combined_text

    return dict(sorted(combined_subtitles.items()))

def get_subtitles(subtitle_path, num_frames, fps, data):
    if subtitle_path is None:
        subtitle_path = os.path.join("./audios/mlvu", data["video_name"].split(".")[0] + ".txt")
    if os.path.exists(subtitle_path) and subtitle_path.endswith(".srt"):
        subtitles = {}
        current_subtitle_text = []
        current_start_time = None
        with open(subtitle_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if '-->' in line:
                    # Process the previous subtitle block
                    if current_subtitle_text and current_start_time is not None:
                        subtitle_text = ' '.join(current_subtitle_text).strip()
                        cleaned_text = clean_text(subtitle_text)
                        if cleaned_text:
                            subtitles[current_start_time] = cleaned_text
                    
                    time_range = line.split('-->')
                    start_time_str = time_range[0].strip()
                    
                    try:
                        h, m, s, _ = map(int, start_time_str.replace(',', ':').split(':'))
                        current_start_time = int(h) * 3600 + int(m) * 60 + int(s)
                    except ValueError:
                        current_start_time = None
                    
                    current_subtitle_text = []
                elif not re.match(r'^\d+$', line) and current_start_time is not None:
                    current_subtitle_text.append(line)

        if current_subtitle_text and current_start_time is not None:
            subtitle_text = ' '.join(current_subtitle_text).strip()
            cleaned_text = clean_text(subtitle_text)
            if cleaned_text:
                subtitles[current_start_time] = cleaned_text
        
        subtitles = group_subtitles_by_interval(subtitles, 5)
        subtitles = list(subtitles.items())
    elif os.path.exists(subtitle_path) and subtitle_path.endswith(".txt"):
        with open(subtitle_path, "r") as f:
            subtitles = f.readlines()
            subtitles = [sub.replace("\n", "") for sub in subtitles]

    elif os.path.exists(subtitle_path) and subtitle_path.endswith(".json"):
        subtitles = json.load(open(subtitle_path, "r"))
        frame_subtitles = {}
        for subtitle in subtitles:
            if "timestamp" in subtitle:
                start, end = subtitle["timestamp"]
                if not isinstance(end, float):
                    end = data["duration"]

                subtitle_text = subtitle["text"]
            else:
                start, end = subtitle["start"], subtitle["end"]
                start = int(timestamp_to_seconds(start))
                end = int(timestamp_to_seconds(end))
                subtitle_text = subtitle["line"]

            frame_subtitles[start] = subtitle_text
        
        subtitles = list(frame_subtitles.items())
    else:
        subtitles = None
    
    return subtitles

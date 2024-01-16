import os


class DatasetCatalog:
    def __init__(self):
        self.audio_instruction = {
            "target": "dataset.T+X-T_instruction_dataset.TX2TInstructionDataset",
            "params": dict(
                data_path="../data/IT_data_ins/T+X-T_data/audio_tx2t.json",
                mm_root_path="../data/IT_data_ins/T+X-T_data/mm_dataset",
                dataset_type="AudioToText",
            ),
        }

        self.video_instruction = {
            "target": "dataset.T+X-T_instruction_dataset.TX2TInstructionDataset",
            "params": dict(
                data_path="../data/IT_data_ins/T+X-T_data/video_tx2t.json",
                mm_root_path="../data/IT_data_ins/T+X-T_data/mm_dataset",
                dataset_type="VideoToText",
            ),
        }

        self.image_instruction = {
            "target": "dataset.T+X-T_instruction_dataset.TX2TInstructionDataset",
            "params": dict(
                data_path="../data/IT_data_ins/T+X-T_data/image_tx2t.json",
                mm_root_path="../data/IT_data_ins/T+X-T_data/mm_dataset",
                dataset_type="ImageToText",
            ),
        }

        self.text_instruction = {
            "target": "dataset.T+X-T_instruction_dataset.TX2TInstructionDataset",
            "params": dict(
                data_path="../data/IT_data_ins/T+X-T_data/text_t2t.json",
                dataset_type="TextToText",
            ),
        }
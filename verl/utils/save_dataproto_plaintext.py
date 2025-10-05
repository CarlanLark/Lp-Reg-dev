from verl.protocol import DataProto
from pathlib import Path
import json
import warnings
from collections.abc import Mapping
import numpy as np
import torch

def save_plaintext_to_disk(data_proto: DataProto, filepath):
    data_proto.check_consistency()
    numpy2python = {
        np.integer: int,
        np.floating: float,
        np.bool_: bool,
        np.complexfloating: complex
    }
    def convert_serializable(obj):
        """
        Recursively convert a data structure to JSON-serializable format.
        - Convert numpy arrays and torch tensors to lists.
        - Skip unsupported types and print a warning.
        """
        try:
            if isinstance(obj, Mapping):
                return {key: convert_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple, set)):
                return [convert_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, torch.Tensor):
                return obj.cpu().detach().numpy().tolist()
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            elif isinstance(obj, np.generic):
                for dtype, func in numpy2python.items():
                    if np.issubdtype(obj.dtype, dtype):
                        return func(obj)
                warnings.warn(f"Unsupported numpy type {obj.dtype}, converting to string.")
                return str(obj)
            else:
                warnings.warn(f"Unsupported type {type(obj).__name__}, converting to string.")
                return str(obj)
        except Exception as e:
            warnings.warn(f"Error converting object: {e}. Skipping.")
            return None

    def transpose_dict_list(dict_list):
        length = len(data_proto)
        list_of_dict = [None for _ in range(length)]
        for key, value in dict_list.items():
            for idx, item in enumerate(value):
                if list_of_dict[idx] is None:
                    list_of_dict[idx] = {}
                list_of_dict[idx][key] = item
        return list_of_dict

    if data_proto.batch is not None:
        tensor_filename = str(Path(filepath).with_suffix(".tensor"))
        try:
            data_proto.batch.memmap_(tensor_filename)
        except Exception as e:
            warnings.warn(f"Error saving tensor batch: {e}")

    if data_proto.non_tensor_batch is not None:
        non_tensor_filename = str(Path(filepath).with_suffix(".jsonl"))
        non_tensor_list = {}
        for key in data_proto.non_tensor_batch.keys():
            try:
                non_tensor_list[key] = convert_serializable(data_proto.non_tensor_batch[key])
            except Exception as e:
                warnings.warn(f"Error converting {key}: {e}. Skipping.")
        
        non_tensor_list = transpose_dict_list(non_tensor_list)
        try:
            with open(non_tensor_filename, 'w') as f:
                for item in non_tensor_list:
                    json.dump(item, f)
                    f.write("\n")
        except Exception as e:
            warnings.warn(f"Error saving non-tensor batch: {e}")

    if data_proto.meta_info is not None:
        meta_info_filename = str(Path(filepath).with_suffix(".meta_info"))
        try:
            with open(meta_info_filename, 'w') as f:
                json.dump(convert_serializable(data_proto.meta_info), f)
        except Exception as e:
            warnings.warn(f"Error saving meta info: {e}")
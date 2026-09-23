# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset and sampler construction shared by the CC-OPD training entry points."""


def create_rl_dataset(data_paths, data_config, tokenizer, processor):
    """Build the configured RL dataset for training or validation."""
    from torch.utils.data import Dataset

    from verl.utils.dataset.rl_dataset import RLHFDataset

    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' "
                f"from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    else:
        batching_mode = data_config.get("batching_mode", None)
        if batching_mode in ["sequential", "mixed"]:
            from verl.utils.dataset.multitask_rl_dataset import MultiTaskRLHFDataset

            dataset_cls = MultiTaskRLHFDataset
            print(f"Multi-task batching mode detected: {batching_mode}")
        else:
            dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    return dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )


def create_rl_sampler(data_config, dataset):
    """Build the configured training sampler."""
    import torch
    from torch.utils.data import RandomSampler, SequentialSampler

    batching_mode = data_config.get("batching_mode", None)
    if batching_mode == "sequential":
        from verl.utils.dataset.multitask_rl_dataset import MultiTaskRLHFDataset

        if isinstance(dataset, MultiTaskRLHFDataset):
            print("Using SequentialTaskSampler for multi-task sequential batching")
            from verl.utils.dataset.multitask_rl_dataset import SequentialTaskSampler

            batch_size = data_config.get("gen_batch_size", data_config.train_batch_size)
            return SequentialTaskSampler(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=data_config.shuffle,
                drop_last=False,
                seed=data_config.get("seed", 42),
            )

    if data_config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(data_config.get("seed", 1))
        return RandomSampler(data_source=dataset, generator=train_dataloader_generator)
    return SequentialSampler(data_source=dataset)

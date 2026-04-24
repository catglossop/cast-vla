from ml_collections import ConfigDict
from palivla.base_config import get_config as get_base_config
from ml_collections.config_dict import placeholder, ConfigDict, FieldReference

def get_config(variant_config: str = "default"):
    config = get_base_config(variant_config)
    
    num_train_steps = FieldReference(200000, int)

    config["data_dir"] = "/raid/datasets"
    data_mix = "bridge"
    config["batch_size"] = 512
    config["eval_batch_size"] = 128
    config["save_path"] = "gs://cat-logs"
    config["save_interval"] = 5000
    config["max_to_keep"] = 10
    config["action_horizon"] = 1
    config["num_steps"] = num_train_steps
    
    config["language_tokenizer"] = "google/paligemma-3b-mix-224"
    config["sequence_builder"] = "sequence_builder.default(prompt_pad_length=50, gen_pad_length=20)"
    
    config["action_tokenizer"] = f"action_tokenizer.bin(min_action_value=-3, max_action_value=3, action_vocab_size=512, action_horizon=1)"

    config["dataset_kwargs"]["oxe_kwargs"]["data_dir"] = config["data_dir"]
    config["dataset_kwargs"]["oxe_kwargs"]["data_mix"] = data_mix
    config["visualization_datasets"]["bridge"]["data_dir"] = config["data_dir"]
    config["wandb_run"] = "bridge"

    config["optimizer"]["kwargs"]["base_learning_rate"] = 1e-4
    config["optimizer"]["kwargs"]["optimizer"] = "adamw"
    return ConfigDict(config)

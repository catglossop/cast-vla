from ml_collections import ConfigDict
from palivla.base_config import get_config as get_base_config

def get_config(variant_config: str = "default"):
    config = get_base_config(variant_config)

    config["data_dir"] = "gs://cat-datasets"
    data_mix = "bridge_cast"
    config["batch_size"] = 192
    config["eval_batch_size"] = 128
    config["save_path"] = "gs://cat-logs"
    config["save_interval"] = 5000
    config["max_to_keep"] = 10
    config["action_horizon"] = 1


    # config["action_tokenizer"] = f"action_tokenizer.dct(action_dim=7, time_horizon=1, save_path='tmp', do_fit=True, pretrained_path=None, default_path='gs://cat-logs/action-tokenizer-dct')"
    config["sequence_builder"] = "sequence_builder.default(prompt_pad_length=50, gen_pad_length=20)"

    config["dataset_kwargs"]["oxe_kwargs"]["data_dir"] = config["data_dir"]
    config["dataset_kwargs"]["oxe_kwargs"]["data_mix"] = data_mix
    config["visualization_datasets"]["bridge"]["data_dir"] = config["data_dir"]
    config["resume_checkpoint_dir"] = "gs://cat-logs/run_2026_04_06_06_48_03"
    config["resume_checkpoint_step"] = 5000
    config["wandb_run"] = "bridge_cast"
    config["overfit_dataset"] = True
    # config["dataset_kwargs"]["oxe_kwargs"]["force_recompute_dataset_statistics"] = False,
    # config["dataset_kwargs"]["oxe_kwargs"]["dataset_statisttraics"] = "gs://cat-datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/bridge_statistics.json"

    config["optimizer"]["kwargs"]["base_learning_rate"] = 1e-4
    config["optimizer"]["kwargs"]["optimizer"] = "adamw"
    return ConfigDict(config)

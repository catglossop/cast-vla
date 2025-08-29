import numpy as np
import sys
import wandb
import time
from ml_collections import config_flags
import tensorflow as tf
from PIL import Image
sys.path.append(".")
import numpy as np
from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import pyngrok import ngrok
import base64
from io import BytesIO
from PIL import Image

# Palivla
from palivla.model_components import ModelComponents
from palivla.inference import run_inference, make_sharding

# Jax imports
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

print("Inference server running...")
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(physical_devices, "GPU")
print("VISIBLE DEVICES: ", jax.devices())

tf.random.set_seed(jax.process_index())
wandb.login()
run = wandb.init(
    # Set the project where this run will be logged
    project="cast-inference",
    mode="online",
)

app = Flask(__name__)
run_with_ngrok(app)

config = None
model = None
avg_time = []
input_prompt = ""
@app.route('/gen_action', methods=["POST"])
def gen_action():
    global config, model, run, input_prompt

    # If first time getting inference, load the model
    if model is None: 
        FLAGS = flags.FLAGS
        FLAGS(sys.argv) 
        config = flags.FLAGS.config
        input_prompt = flags.FLAGS.prompt

        if flags.FLAGS.platform == "tpu":
            jax.distributed.initialize()
        sharding_metadata = make_sharding(config)

        print("\nLoading model...", flags.FLAGS.checkpoint_dir)
        model = ModelComponents.load_static(flags.FLAGS.checkpoint_dir, sharding_metadata, weights_only=True)
        manager = ocp.CheckpointManager(flags.FLAGS.checkpoint_dir, options=ocp.CheckpointManagerOptions())
        model.load_state(flags.FLAGS.checkpoint_step, manager, weights_only=True)
        print("\nModel loaded!")

    # Receive data 
    data = request.get_json()
    obs_data = base64.b64decode(data['obs'])
    obs = Image.open(BytesIO(obs_data))
    api_prompt = data['prompt']

    if api_prompt != "":
        prompt = api_prompt
    else:
        prompt = input_prompt

    print(f"Prompt: {prompt}")

    # Run inference
    start_time = time.time()
    action, viz = run_inference(model, prompt, obs, config)

    print(action)
    
    run_time = time.time() - start_time
    avg_time.append(run_time)

    print(f"Avg. run time: {np.array(avg_time).mean()}s")
    if viz is not None:
        viz = {k: wandb.Image(v) for k, v in viz.items()}
        run.log(viz)
    response = jsonify(action=action.tolist())
    return response

if __name__ == "__main__":
    # CLI FLAGS
    config_flags.DEFINE_config_file(
            "config", "configs/inference_config.py", "Path to the config file."
    )
    flags.DEFINE_string("platform", "gpu", "Platform to run on.")
    flags.DEFINE_string("checkpoint_dir", "", "Path to the checkpoint directory.")
    flags.DEFINE_integer("checkpoint_step", -1, "Step to resume from.")
    flags.DEFINE_string("prompt", "", "Prompt to generate action from.")
    app.run()
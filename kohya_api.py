from typing import List, Union

from fastapi import FastAPI
from pydantic import BaseModel
import os
import boto3
from kohya_gui import lora_gui
from kohya_gui import dreambooth_folder_creation_gui
from dotenv import load_dotenv

DATA_ROOT_PATH = "/home/gazai/opt/DATA/external"
# DATA_ROOT_PATH = "./external"

app = FastAPI()

load_dotenv()
AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_ACCESS_SECRET_KEY = os.environ.get("AWS_ACCESS_SECRET_KEY")

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_ACCESS_SECRET_KEY,
)


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None


class TrainingParams(BaseModel):
    user_id: str
    name: str
    training_images: List[str]
    regularization_images: List[str]
    instance_prompt: str
    class_prompt: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}


@app.post("/model/train")
def train_model(training_params: TrainingParams):
    model_name = training_params.name
    user_id = training_params.user_id
    instance_prompt = training_params.instance_prompt
    class_prompt = training_params.class_prompt

    training_images_dir_input = (
        rf"{DATA_ROOT_PATH}/ft-Inputs/{user_id}/{model_name}/raw/img"
    )
    regularization_images_dir_input = (
        rf"{DATA_ROOT_PATH}/ft-Inputs/{user_id}/{model_name}/raw/reg"
    )
    prepared_project_dir = (
        rf"{DATA_ROOT_PATH}/ft-Inputs/{user_id}/{model_name}/prepared"
    )

    os.makedirs(training_images_dir_input, exist_ok=True)
    os.makedirs(regularization_images_dir_input, exist_ok=True)

    bucket_name = "gazai"

    for object_name in training_params.training_images:
        object_key = rf"assets/{user_id}/{object_name}"
        local_file_path = os.path.join(training_images_dir_input, object_name)
        s3.download_file(bucket_name, object_key, local_file_path)

    for object_name in training_params.regularization_images:
        object_key = rf"assets/{user_id}/{object_name}"
        local_file_path = os.path.join(regularization_images_dir_input, object_name)
        s3.download_file(bucket_name, object_key, local_file_path)

    dreambooth_folder_creation_gui.dreambooth_folder_preparation(
        util_training_images_dir_input=training_images_dir_input,
        util_training_images_repeat_input=40,
        util_instance_prompt_input=instance_prompt,
        util_regularization_images_dir_input=regularization_images_dir_input,
        util_regularization_images_repeat_input=1,
        util_class_prompt_input=class_prompt,
        util_training_dir_output=prepared_project_dir,
    )

    lora_gui.train_model(
        headless=False,
        print_only=False,
        pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0",
        v2=False,
        v_parameterization=False,
        sdxl=True,
        logging_dir=os.path.join(prepared_project_dir, "log"),
        train_data_dir=os.path.join(prepared_project_dir, "img"),
        reg_data_dir=os.path.join(prepared_project_dir, "reg"),
        output_dir=os.path.join(prepared_project_dir, "model"),
        dataset_config="",
        max_resolution="512,512",
        learning_rate=0.0001,
        lr_scheduler="cosine",
        lr_warmup=10,
        train_batch_size=1,
        epoch=1,
        save_every_n_epochs=1,
        mixed_precision="fp16",
        save_precision="fp16",
        seed=0,
        num_cpu_threads_per_process=2,
        cache_latents=True,
        cache_latents_to_disk=False,
        caption_extension=".txt",
        enable_bucket=True,
        gradient_checkpointing=False,
        fp8_base=False,
        full_fp16=False,
        stop_text_encoder_training_pct=0,
        min_bucket_reso=256,
        max_bucket_reso=2048,
        xformers="xformers",
        save_model_as="safetensors",
        shuffle_caption=False,
        save_state=False,
        save_state_on_train_end=False,
        resume="",
        prior_loss_weight=1,
        text_encoder_lr=0.0001,
        unet_lr=0.0001,
        network_dim=8,
        network_weights="",
        dim_from_weights=False,
        color_aug=False,
        flip_aug=False,
        masked_loss=False,
        clip_skip=1,
        num_processes=1,
        num_machines=1,
        multi_gpu=False,
        gpu_ids="",
        main_process_port=0,
        gradient_accumulation_steps=1,
        mem_eff_attn=False,
        output_name="last",
        model_list="custom",
        max_token_length="75",
        max_train_epochs=0,
        max_train_steps=1600,
        max_data_loader_n_workers=0,
        network_alpha=1,
        training_comment="",
        keep_tokens=0,
        lr_scheduler_num_cycles=1,
        lr_scheduler_power=1,
        persistent_data_loader_workers=False,
        bucket_no_upscale=True,
        random_crop=False,
        bucket_reso_steps=64,
        v_pred_like_loss=0,
        caption_dropout_every_n_epochs=0,
        caption_dropout_rate=0,
        optimizer="AdamW8bit",
        optimizer_args="",
        lr_scheduler_args="",
        max_grad_norm=1,
        noise_offset_type="Original",
        noise_offset=0,
        noise_offset_random_strength=False,
        adaptive_noise_scale=0,
        multires_noise_iterations=0,
        multires_noise_discount=0.3,
        ip_noise_gamma=0,
        ip_noise_gamma_random_strength=False,
        LoRA_type="Standard",
        factor=-1,
        bypass_mode=False,
        dora_wd=False,
        use_cp=False,
        use_tucker=False,
        use_scalar=False,
        rank_dropout_scale=False,
        constrain=0,
        rescaled=False,
        train_norm=False,
        decompose_both=False,
        train_on_input=True,
        conv_dim=1,
        conv_alpha=1,
        sample_every_n_steps=0,
        sample_every_n_epochs=0,
        sample_sampler="euler_a",
        sample_prompts="",
        additional_parameters="",
        loss_type="l2",
        huber_schedule="snr",
        huber_c=0.1,
        vae_batch_size=0,
        min_snr_gamma=0,
        down_lr_weight="",
        mid_lr_weight="",
        up_lr_weight="",
        block_lr_zero_threshold="",
        block_dims="",
        block_alphas="",
        conv_block_dims="",
        conv_block_alphas="",
        weighted_captions=False,
        unit=1,
        save_every_n_steps=0,
        save_last_n_steps=0,
        save_last_n_steps_state=0,
        log_with="",
        wandb_api_key="",
        wandb_run_name="",
        log_tracker_name="",
        log_tracker_config="",
        scale_v_pred_loss_like_noise_pred=False,
        scale_weight_norms=0,
        network_dropout=0,
        rank_dropout=0,
        module_dropout=0,
        sdxl_cache_text_encoder_outputs=False,
        sdxl_no_half_vae=False,
        full_bf16=False,
        min_timestep=0,
        max_timestep=1000,
        vae="",
        dynamo_backend="no",
        dynamo_mode="default",
        dynamo_use_fullgraph=False,
        dynamo_use_dynamic=False,
        extra_accelerate_launch_args="",
        LyCORIS_preset="full",
        debiased_estimation_loss=False,
        huggingface_repo_id="",
        huggingface_token="",
        huggingface_repo_type="",
        huggingface_repo_visibility="",
        huggingface_path_in_repo="",
        save_state_to_huggingface=False,
        resume_from_huggingface="",
        async_upload=False,
        metadata_author="",
        metadata_description="",
        metadata_license="",
        metadata_tags="",
        metadata_title="",
    )

    return training_params

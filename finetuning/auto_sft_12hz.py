# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
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
import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

import bitsandbytes as bnb


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def enable_gradient_checkpointing(model):
    """Lower VRAM usage by disabling cache and enabling gradient checkpointing"""
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    if hasattr(model, "model"):
        for _, module in model.model.named_modules():
            if hasattr(module, "config") and hasattr(module.config, "use_cache"):
                module.config.use_cache = False
            if hasattr(module, "generation_config") and hasattr(module.generation_config, "use_cache"):
                module.generation_config.use_cache = False

        if hasattr(model.model, "gradient_checkpointing_enable"):
            model.model.gradient_checkpointing_enable()


target_speaker_embedding = None
def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--use_8bit_adam", action="store_true",
        help="Use bitsandbytes 8-bit AdamW optimizer to reduce VRAM usage."
    )
    args = parser.parse_args()

    GRAD_ACCUM_STEPS = args.grad_accum_steps

    # Added the project_dir argument to specify the folder where the logs should be saved
    accelerator = Accelerator(gradient_accumulation_steps=GRAD_ACCUM_STEPS, mixed_precision="bf16", log_with="tensorboard", project_dir="./tensorboard_logs")

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    )
    enable_gradient_checkpointing(qwen3tts)

    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    if args.use_8bit_adam:
        # Choose quantized AdamW for further lower VRAM usage
        optimizer = bnb.optim.AdamW8bit(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

    save_num = 3
    epoch_last_loss = float("inf")

    for epoch in range(num_epochs):
        epoch_total_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.text_projection(model.talker.get_text_embeddings()(input_text_ids)) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_mask,
                    labels=codec_0_labels,
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[:, :-1, :][codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + 0.3 * sub_talker_loss
                loss_value = loss.item()
                epoch_total_loss += loss_value

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Batch Loss: {loss_value:.4f}")

        if accelerator.is_main_process:
            # ---auto---
            epoch_avg_loss = epoch_total_loss / len(train_dataloader)

            # temp
            accelerator.print(f"Epoch {epoch} | Epoch Loss: {epoch_avg_loss:.4f}")

            if epoch > 0 and epoch_avg_loss <= epoch_last_loss and epoch < num_epochs-1:
                last_output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch-1}")
                if os.path.exists(last_output_dir):
                    shutil.rmtree(last_output_dir)

            if epoch_avg_loss > epoch_last_loss and save_num > 0:
                save_num -= 1

            epoch_last_loss = epoch_avg_loss
            # ----------

            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {
                args.speaker_name: 3000
            }
            talker_config["spk_is_dialect"] = {
                args.speaker_name: False
            }
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)

        if save_num <= 1:
            break

if __name__ == "__main__":
    train()

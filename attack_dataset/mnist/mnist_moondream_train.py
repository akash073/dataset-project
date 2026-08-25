from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 1. Configuration
# ============================================================

MODEL_ID = "vikhyatk/moondream2"

# Keep the same revision as your CIFAR-10 code
MODEL_REVISION = "2024-08-26"

# Folder containing your MNIST dataset
MNIST_ROOT = "../../data"

OUTPUT_ROOT = Path("./moondream_mnist_finetuned")
FINAL_MODEL_DIR = OUTPUT_ROOT / "final_model"

SEED = 42

NUM_EPOCHS = 3

# Keep batch size 1 because of the current training implementation
BATCH_SIZE = 1

GRADIENT_ACCUMULATION_STEPS = 8

LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
MAX_GRADIENT_NORM = 1.0

# Start small first.
# Change to None to use all 60,000 training samples.
MAX_TRAIN_SAMPLES = 50

# Save intermediate checkpoint
SAVE_EVERY_STEPS = 1000

NUM_WORKERS = 0


# ============================================================
# MNIST classes
# ============================================================

CLASS_NAMES = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]


QUESTION = (
    "Classify this MNIST handwritten digit image. "
    "Answer with exactly one digit: "
    "0, 1, 2, 3, 4, 5, 6, 7, 8, or 9."
)


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# ============================================================
# 3. Device configuration
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA was not detected. "
        "This training script requires an NVIDIA CUDA GPU."
    )


DEVICE = torch.device("cuda")


DTYPE = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)


print("=" * 70)

print("SYSTEM INFORMATION")

print("=" * 70)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "CUDA version:",
    torch.version.cuda
)

print(
    "Training dtype:",
    DTYPE
)

print(
    "Output directory:",
    OUTPUT_ROOT.resolve()
)


# ============================================================
# 4. MNIST dataset
# ============================================================

class MoondreamMNISTDataset(Dataset):

    def __init__(
        self,
        root: str,
        max_samples: int | None = None,
    ) -> None:

        self.dataset = MNIST(
            root=root,
            train=True,
            download=False,
        )


        all_indices = list(
            range(
                len(self.dataset)
            )
        )


        if max_samples is not None:

            generator = (
                torch.Generator()
                .manual_seed(SEED)
            )


            random_indices = torch.randperm(
                len(all_indices),
                generator=generator,
            ).tolist()


            all_indices = (
                random_indices[
                    :max_samples
                ]
            )


        self.indices = all_indices


    def __len__(self) -> int:

        return len(
            self.indices
        )


    def __getitem__(
        self,
        index: int
    ) -> dict:

        original_index = (
            self.indices[
                index
            ]
        )


        image, label_id = (
            self.dataset[
                original_index
            ]
        )


        # MNIST is grayscale.
        # Moondream expects RGB image input.
        image = image.convert(
            "RGB"
        )


        return {

            "image":
                image,

            "label":
                CLASS_NAMES[
                    label_id
                ],

            "label_id":
                int(label_id),

            "original_index":
                original_index,

        }


# ============================================================
# Batch handling
# ============================================================

def single_sample_collate(
    batch: list[dict]
) -> dict:

    if len(batch) != 1:

        raise ValueError(
            "This implementation requires "
            "BATCH_SIZE = 1."
        )

    return batch[0]


train_dataset = MoondreamMNISTDataset(
    root=MNIST_ROOT,
    max_samples=MAX_TRAIN_SAMPLES,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=NUM_WORKERS,
    collate_fn=single_sample_collate,
)


print(
    "\nNumber of MNIST training samples:",
    len(train_dataset)
)


# ============================================================
# 5. Load Moondream
# ============================================================

print(
    "\nLoading tokenizer..."
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=True,
)


print(
    "Loading model..."
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    revision=MODEL_REVISION,
    trust_remote_code=True,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
)


model.to(
    DEVICE
)


print(
    "Model loaded successfully."
)


# ============================================================
# 6. Freeze vision encoder
# ============================================================

for parameter in (
    model
    .vision_encoder
    .parameters()
):

    parameter.requires_grad = False


model.vision_encoder.eval()


# Train the text model
for parameter in (
    model
    .text_model
    .parameters()
):

    parameter.requires_grad = True


# ============================================================
# Parameter count
# ============================================================

def count_parameters(
    module
) -> tuple[int, int]:

    total = sum(
        parameter.numel()
        for parameter
        in module.parameters()
    )


    trainable = sum(
        parameter.numel()
        for parameter
        in module.parameters()
        if parameter.requires_grad
    )


    return (
        total,
        trainable
    )


total_parameters, trainable_parameters = (
    count_parameters(
        model
    )
)


print(
    "\nParameter information"
)

print(
    "-" * 70
)

print(
    f"Total parameters:     "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_parameters:,}"
)

print(
    f"Trainable percentage: "
    f"{100 * trainable_parameters / total_parameters:.4f}%"
)


# ============================================================
# 7. Find token embedding layer
# ============================================================

def find_token_embedding_layer(
    moondream_model
):

    possible_paths = [

        lambda m:
            m.text_model
            .transformer
            .embd
            .wte,

        lambda m:
            m.text_model
            .get_input_embeddings(),

        lambda m:
            m.get_input_embeddings(),

    ]


    for getter in possible_paths:

        try:

            embedding_layer = (
                getter(
                    moondream_model
                )
            )


            if embedding_layer is not None:

                return embedding_layer


        except (
            AttributeError,
            TypeError
        ):

            continue


    raise RuntimeError(
        "Could not locate Moondream's "
        "token embedding layer."
    )


token_embedding = (
    find_token_embedding_layer(
        model
    )
)


print(
    "\nText embedding layer:",
    type(
        token_embedding
    ).__name__
)


# ============================================================
# 8. Tokenization helpers
# ============================================================

def tokenize_without_special_tokens(
    text: str
) -> torch.Tensor:

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt",
    )


    return (
        encoded
        .input_ids
        .to(DEVICE)
    )


def get_start_token_id() -> int:

    if tokenizer.bos_token_id is not None:

        return (
            tokenizer
            .bos_token_id
        )


    if tokenizer.eos_token_id is not None:

        return (
            tokenizer
            .eos_token_id
        )


    raise RuntimeError(
        "Tokenizer has neither "
        "BOS nor EOS token."
    )


# ============================================================
# 9. Image encoding
# ============================================================

def extract_tensor_from_output(
    output
) -> torch.Tensor:

    if isinstance(
        output,
        torch.Tensor
    ):

        return output


    if isinstance(
        output,
        tuple
    ):

        for item in output:

            if isinstance(
                item,
                torch.Tensor
            ):

                return item


    if isinstance(
        output,
        dict
    ):

        possible_keys = [

            "image_embeds",
            "image_embeddings",
            "embeddings",
            "last_hidden_state",

        ]


        for key in possible_keys:

            value = output.get(
                key
            )

            if isinstance(
                value,
                torch.Tensor
            ):

                return value


    raise TypeError(
        "Vision encoder returned an unsupported "
        f"output type: {type(output)}"
    )


def encode_image(
    image
) -> torch.Tensor:

    processed_image = (
        model
        .vision_encoder
        .preprocess(
            image
        )
    )


    if isinstance(
        processed_image,
        list
    ):

        processed_image = torch.stack(
            processed_image
        )


    if not isinstance(
        processed_image,
        torch.Tensor
    ):

        raise TypeError(
            "Vision preprocessing did not "
            "return a tensor."
        )


    if processed_image.ndim == 3:

        processed_image = (
            processed_image
            .unsqueeze(0)
        )


    processed_image = processed_image.to(
        device=DEVICE,
        dtype=DTYPE,
    )


    with torch.no_grad():

        try:

            output = (
                model
                .vision_encoder(
                    processed_image
                )
            )


        except TypeError:

            output = (
                model
                .vision_encoder(
                    pixel_values=
                        processed_image
                )
            )


    image_embeddings = (
        extract_tensor_from_output(
            output
        )
    )


    if image_embeddings.ndim == 2:

        image_embeddings = (
            image_embeddings
            .unsqueeze(0)
        )


    if image_embeddings.ndim != 3:

        raise RuntimeError(
            "Expected vision embeddings with "
            "shape [batch, sequence, hidden_size], "
            f"received {tuple(image_embeddings.shape)}"
        )


    return image_embeddings


# ============================================================
# 10. Construct training tensors
# ============================================================

def create_training_tensors(
    image,
    answer: str,
):

    # --------------------------------------------------------
    # Encode image
    # --------------------------------------------------------

    image_embeddings = (
        encode_image(
            image
        )
    )


    # --------------------------------------------------------
    # Prompt and answer
    # --------------------------------------------------------

    prompt_text = (
        f"\n\nQuestion: {QUESTION}"
        "\n\nAnswer:"
    )


    prompt_token_ids = (
        tokenize_without_special_tokens(
            prompt_text
        )
    )


    answer_token_ids = (
        tokenize_without_special_tokens(
            answer
        )
    )


    # --------------------------------------------------------
    # Add EOS after target answer
    # --------------------------------------------------------

    if tokenizer.eos_token_id is not None:

        eos_token = torch.tensor(
            [
                [
                    tokenizer
                    .eos_token_id
                ]
            ],
            dtype=torch.long,
            device=DEVICE,
        )


        answer_token_ids = torch.cat(
            [
                answer_token_ids,
                eos_token
            ],
            dim=1,
        )


    # --------------------------------------------------------
    # Start token
    # --------------------------------------------------------

    start_token_ids = torch.tensor(
        [
            [
                get_start_token_id()
            ]
        ],
        dtype=torch.long,
        device=DEVICE,
    )


    # --------------------------------------------------------
    # Token embeddings
    # --------------------------------------------------------

    start_embeddings = (
        token_embedding(
            start_token_ids
        )
    )


    prompt_embeddings = (
        token_embedding(
            prompt_token_ids
        )
    )


    answer_embeddings = (
        token_embedding(
            answer_token_ids
        )
    )


    image_embeddings = image_embeddings.to(
        device=DEVICE,
        dtype=start_embeddings.dtype,
    )


    # --------------------------------------------------------
    # Verify dimensions
    # --------------------------------------------------------

    text_hidden_size = (
        start_embeddings
        .shape[-1]
    )


    vision_hidden_size = (
        image_embeddings
        .shape[-1]
    )


    if (
        vision_hidden_size
        !=
        text_hidden_size
    ):

        raise RuntimeError(
            "\nVision and text embedding "
            "dimensions do not match.\n"
            f"Vision: {vision_hidden_size}\n"
            f"Text:   {text_hidden_size}"
        )


    if not hasattr(
        create_training_tensors,
        "_shapes_printed"
    ):

        print(
            "\nFirst-sample tensor shapes"
        )

        print(
            "-" * 70
        )

        print(
            "Start embeddings: ",
            start_embeddings.shape
        )

        print(
            "Image embeddings: ",
            image_embeddings.shape
        )

        print(
            "Prompt embeddings:",
            prompt_embeddings.shape
        )

        print(
            "Answer embeddings:",
            answer_embeddings.shape
        )


        create_training_tensors._shapes_printed = True


    # --------------------------------------------------------
    # Build input embeddings
    # --------------------------------------------------------

    inputs_embeds = torch.cat(
        [
            start_embeddings,
            image_embeddings,
            prompt_embeddings,
            answer_embeddings,
        ],
        dim=1,
    )


    # --------------------------------------------------------
    # Ignore prompt/image tokens in loss
    # --------------------------------------------------------

    ignored_length = (
        start_embeddings.shape[1]
        +
        image_embeddings.shape[1]
        +
        prompt_embeddings.shape[1]
    )


    ignored_labels = torch.full(
        size=(
            1,
            ignored_length
        ),
        fill_value=-100,
        dtype=torch.long,
        device=DEVICE,
    )


    labels = torch.cat(
        [
            ignored_labels,
            answer_token_ids,
        ],
        dim=1,
    )


    attention_mask = torch.ones(
        size=
            inputs_embeds.shape[:2],
        dtype=torch.long,
        device=DEVICE,
    )


    if (
        inputs_embeds.shape[:2]
        !=
        labels.shape
    ):

        raise RuntimeError(
            "Input and label sequence lengths "
            "do not match.\n"
            f"Input: {inputs_embeds.shape}\n"
            f"Labels: {labels.shape}"
        )


    return (
        inputs_embeds,
        labels,
        attention_mask
    )


# ============================================================
# 11. One-sample validation
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "ONE-SAMPLE VALIDATION"
)

print(
    "=" * 70
)


validation_sample = (
    train_dataset[0]
)


with torch.no_grad():

    (
        validation_inputs,
        validation_labels,
        validation_mask

    ) = create_training_tensors(

        image=
            validation_sample[
                "image"
            ],

        answer=
            validation_sample[
                "label"
            ],
    )


print(
    "\nValidation succeeded."
)

print(
    "Inputs:",
    validation_inputs.shape
)

print(
    "Labels:",
    validation_labels.shape
)

print(
    "Attention mask:",
    validation_mask.shape
)


del validation_inputs
del validation_labels
del validation_mask

torch.cuda.empty_cache()


# ============================================================
# 12. Optimizer
# ============================================================

trainable_parameter_list = [

    parameter

    for parameter
    in model.parameters()

    if parameter.requires_grad

]


optimizer = AdamW(
    trainable_parameter_list,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)


use_gradient_scaler = (
    DTYPE
    ==
    torch.float16
)


scaler = torch.amp.GradScaler(
    device="cuda",
    enabled=use_gradient_scaler,
)


# ============================================================
# 13. Save checkpoint
# ============================================================

def save_model(
    directory: Path,
    metadata: dict,
) -> None:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


    model.save_pretrained(
        directory,
        safe_serialization=True,
    )


    tokenizer.save_pretrained(
        directory
    )


    metadata_path = (
        directory
        /
        "training_metadata.json"
    )


    with metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


    print(
        "\nSaved model to:",
        directory.resolve()
    )


# ============================================================
# 14. Training loop
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


model.train()

# Keep frozen image encoder in evaluation mode
model.vision_encoder.eval()


optimizer.zero_grad(
    set_to_none=True
)


global_sample_step = 0
optimizer_step = 0


print(
    "\n" + "=" * 70
)

print(
    "MNIST TRAINING"
)

print(
    "=" * 70
)


for epoch in range(
    NUM_EPOCHS
):

    running_loss = 0.0


    progress_bar = tqdm(
        train_loader,
        desc=
            f"Epoch {epoch + 1}/"
            f"{NUM_EPOCHS}",
    )


    for (
        batch_index,
        sample
    ) in enumerate(
        progress_bar
    ):

        (
            inputs_embeds,
            labels,
            attention_mask

        ) = create_training_tensors(

            image=
                sample[
                    "image"
                ],

            answer=
                sample[
                    "label"
                ],
        )


        with torch.autocast(
            device_type="cuda",
            dtype=DTYPE,
            enabled=True,
        ):

            outputs = (
                model
                .text_model(
                    inputs_embeds=
                        inputs_embeds,

                    attention_mask=
                        attention_mask,

                    labels=
                        labels,

                    use_cache=False,
                )
            )


            loss = outputs.loss


            backward_loss = (
                loss
                /
                GRADIENT_ACCUMULATION_STEPS
            )


        scaler.scale(
            backward_loss
        ).backward()


        global_sample_step += 1


        running_loss += (
            loss.item()
        )


        is_accumulation_boundary = (
            global_sample_step
            %
            GRADIENT_ACCUMULATION_STEPS
            ==
            0
        )


        is_last_batch = (
            batch_index
            ==
            len(train_loader) - 1
        )


        if (
            is_accumulation_boundary
            or
            is_last_batch
        ):

            scaler.unscale_(
                optimizer
            )


            torch.nn.utils.clip_grad_norm_(
                trainable_parameter_list,
                MAX_GRADIENT_NORM,
            )


            scaler.step(
                optimizer
            )

            scaler.update()


            optimizer.zero_grad(
                set_to_none=True
            )


            optimizer_step += 1


        average_loss = (
            running_loss
            /
            (batch_index + 1)
        )


        progress_bar.set_postfix(
            current_loss=
                f"{loss.item():.4f}",

            average_loss=
                f"{average_loss:.4f}",

            optimizer_step=
                optimizer_step,
        )


        del inputs_embeds
        del labels
        del attention_mask
        del outputs
        del loss


        if (
            SAVE_EVERY_STEPS > 0
            and
            global_sample_step
            %
            SAVE_EVERY_STEPS
            ==
            0
        ):

            checkpoint_dir = (
                OUTPUT_ROOT
                /
                f"checkpoint-"
                f"{global_sample_step}"
            )


            save_model(
                checkpoint_dir,
                metadata={
                    "base_model":
                        MODEL_ID,

                    "base_revision":
                        MODEL_REVISION,

                    "dataset":
                        "MNIST",

                    "epoch":
                        epoch + 1,

                    "sample_step":
                        global_sample_step,

                    "optimizer_step":
                        optimizer_step,

                    "average_loss":
                        average_loss,

                    "training_samples":
                        len(
                            train_dataset
                        ),

                    "classes":
                        CLASS_NAMES,

                    "question":
                        QUESTION,
                },
            )


    epoch_average_loss = (
        running_loss
        /
        len(train_loader)
    )


    print(
        f"\nEpoch {epoch + 1} completed. "
        f"Average loss: "
        f"{epoch_average_loss:.6f}"
    )


# ============================================================
# 15. Save final model
# ============================================================

save_model(
    FINAL_MODEL_DIR,
    metadata={
        "base_model":
            MODEL_ID,

        "base_revision":
            MODEL_REVISION,

        "dataset":
            "MNIST",

        "epochs":
            NUM_EPOCHS,

        "training_samples":
            len(train_dataset),

        "sample_steps":
            global_sample_step,

        "optimizer_steps":
            optimizer_step,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "gradient_accumulation_steps":
            GRADIENT_ACCUMULATION_STEPS,

        "classes":
            CLASS_NAMES,

        "question":
            QUESTION,
    },
)


print(
    "\n" + "=" * 70
)

print(
    "MNIST TRAINING COMPLETED"
)

print(
    "=" * 70
)

print(
    "Final model:",
    FINAL_MODEL_DIR.resolve()
)
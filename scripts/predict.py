import torch
import time
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import os

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    squad_convert_examples_to_features
)

from transformers.data.processors.squad import SquadResult, SquadV2Processor, SquadExample
from transformers.data.metrics.squad_metrics import compute_predictions_logits


###
### Test just to check the Directories
###

absolute_path = os.path.abspath(__file__)
print("Full path: " + absolute_path)
print("Directory Path: " + os.path.dirname(absolute_path))

# Step 2: Define the starting directory
start_directory = "/"

# Step 3: Use the os.walk() function to iterate through the directory tree
for root, dirs, files in os.walk(start_directory):
  # Step 4: Print the file names
  for file_name in files:
    # Combine the root directory path with the file name
    full_path = os.path.join(root, file_name)
    print(full_path)

###
### Test just to check the Directories
###


def run_prediction(question_texts, context_text, model_path):
    ### Setting hyperparameters
    max_seq_length = 512
    doc_stride = 256
    n_best_size = 1
    max_query_length = 64
    max_answer_length = 512
    do_lower_case = False
    null_score_diff_threshold = 0.0

    #model_name_or_path = "/cuad-models/roberta-base/"
    model_name_or_path = "Rakib/roberta-base-on-cuad/"

    def to_list(tensor):
        return tensor.detach().cpu().tolist()

    print("Initializing objects")
    config_class, model_class, tokenizer_class = (
        AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer)
    config = config_class.from_pretrained(model_path)
    tokenizer = tokenizer_class.from_pretrained(
        model_path, do_lower_case=True, use_fast=False)
    model = model_class.from_pretrained(model_path, config=config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    processor = SquadV2Processor()
    examples = []

    print("Creating squad examples")
    for i, question_text in enumerate(question_texts):
        example = SquadExample(
            qas_id=str(i),
            question_text=question_text,
            context_text=context_text,
            answer_text=None,
            start_position_character=None,
            title="Predict",
            answers=None,
        )

        examples.append(example)

    print("Converting examples to features")
    features, dataset = squad_convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        doc_stride=doc_stride,
        max_query_length=max_query_length,
        is_training=False,
        return_dataset="pt",
        threads=1,
    )

    print("Loading eval sampler")
    eval_sampler = SequentialSampler(dataset)

    print("Loading dataloader")
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=10)

    all_results = []

    print("Loading batches")
    for batch in eval_dataloader:
        print("- Eval model")
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]

            print("-Load into model")
            outputs = model(**inputs)

            print("-Enumerate indices")
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)

                output = [to_list(output[i]) for output in outputs.to_tuple()]

                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)
                all_results.append(result)

    print("Making final predictions")
    final_predictions = compute_predictions_logits(
        all_examples=examples,
        all_features=features,
        all_results=all_results,
        n_best_size=n_best_size,
        max_answer_length=max_answer_length,
        do_lower_case=do_lower_case,
        output_prediction_file=None,
        output_nbest_file=None,
        output_null_log_odds_file=None,
        verbose_logging=False,
        version_2_with_negative=True,
        null_score_diff_threshold=null_score_diff_threshold,
        tokenizer=tokenizer
    )

    return final_predictions

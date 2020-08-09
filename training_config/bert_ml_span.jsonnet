{
    "dataset_reader": {
      "type": "transformer_srl_span",
      "model_name": "bert-base-multilingual-cased",
    },

    "data_loader": {
      "batch_sampler": {
        "type": "bucket",
        "batch_size" : 32
      }
    },

    "train_data_path": std.extVar("SRL_TRAIN_DATA_PATH"),
    "validation_data_path": std.extVar("SRL_VALIDATION_DATA_PATH"),

    "model": {
        "type": "transformer_srl_span",
        "embedding_dropout": 0.1,
        "bert_model": "bert-base-multilingual-cased",
    },

    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 5e-5,
            "correct_bias": false,
            "weight_decay": 0.01,
            "parameter_groups": [
              [["bias", "LayerNorm.bias", "LayerNorm.weight", "layer_norm.weight"], {"weight_decay": 0.0}],
            ],
        },

        "learning_rate_scheduler": {
            "type": "slanted_triangular",
        },
        "checkpointer": {
            "num_serialized_models_to_keep": 2,
        },
        "grad_norm": 1.0,
        "num_epochs": 15,
        "validation_metric": "+f1_role",
        "cuda_device": 0,
    },
}
{
  "dataset_reader": {
    "type": "conll2003",
    "tag_label": "ner",
    "coding_scheme": "BIOUL", 
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "elmo": {
        "type": "elmo_characters"
      },
      "token_characters": {
        "type": "characters"
      }
    }
  },
  "train_data_path": std.extVar("NER_TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("NER_TEST_A_PATH"),
  "test_data_path": std.extVar("NER_TEST_B_PATH"),
  "evaluate_on_test": true,
  "model": {
    "type": "crf_tagger",
    "calculate_span_f1": true,
    "label_encoding": "BIOUL",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "embedding",
          "embedding_dim": 50,
          "pretrained_file": "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.50d.txt.gz",
          "trainable": true
        },
        "elmo": {
            "type": "elmo_token_embedder",
            "options_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_options.json",
            "weight_file": "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x1024_128_2048cnn_1xhighway/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
            "embedding_dim": 25
            },
            "encoder": {
            "type": "gru",
            "input_size": 25,
            "hidden_size": 80,
            "num_layers": 2,
            "dropout": 0.25,
            "bidirectional": true
            }
        }
      }
    },
    "encoder": {
      "type": "gru",
      "input_size": 466,
      "hidden_size": 500,
      "num_layers": 2,
      "dropout": 0.5,
      "bidirectional": true
    },
    "regularizer": [
      [
        "transitions$",
        {
          "type": "l2",
          "alpha": 0.01
        }
      ]
    ]
  },
  "iterator": {
    "type": "bucket",    
    "sorting_keys": [
      [
        "tokens",
        "num_tokens"
      ]
    ],
    "batch_size": 32
  },
  "trainer": {
    "optimizer": "adam",
    "num_epochs": 50,
    "patience": 10,
    "cuda_device": 0
  }
}

use std::fs::File;

use lazy_static::lazy_static;
use ordered_float::NotNan;
use sticker::tensorflow::{ModelConfig, OpNames};
use sticker::Layer;

use super::{Config, Embedding, EmbeddingAlloc, Embeddings, Labeler, TomlRead, Train};

lazy_static! {
    static ref BASIC_LABELER_CHECK: Config = Config {
        labeler: Labeler {
            layer: Layer::Feature("tf".to_string()),
            labels: "sticker.labels".to_owned(),
            read_ahead: 10,
        },
        embeddings: Embeddings {
            word: Embedding {
                filename: "word-vectors.bin".into(),
                alloc: EmbeddingAlloc::Mmap,
            },
            tag: Some(Embedding {
                filename: "tag-vectors.bin".into(),
                alloc: EmbeddingAlloc::Read,
            }),
        },
        train: Train {
            initial_lr: NotNan::from(0.05),
            lr_scale: NotNan::from(0.5),
            lr_patience: 4,
            patience: 10,
        },
        model: ModelConfig {
            batch_size: 128,
            graph: "sticker.graph".to_owned(),
            parameters: "sticker.model".to_owned(),
            intra_op_parallelism_threads: 4,
            inter_op_parallelism_threads: 4,
            op_names: OpNames {
                is_training_op: "prediction/model/is_training".to_owned(),
                init_op: "prediction/model/init".to_owned(),
                labels_op: "prediction/model/labels".to_owned(),
                inputs_op: "prediction/model/inputs".to_owned(),
                seq_lens_op: "prediction/model/seq_lens".to_owned(),
                predicted_op: "prediction/model/predicted".to_owned(),
                top_k_predicted_op: "prediction/model/top_k_predicted".to_owned(),
                accuracy_op: "prediction/model/accuracy".to_owned(),
                loss_op: "prediction/model/loss".to_owned(),
                lr_op: "prediction/model/lr".to_owned(),
                save_op: "prediction/model/save".to_owned(),
                save_path_op: "prediction/model/save_path".to_owned(),
                restore_op: "prediction/model/restore".to_owned(),
                train_op: "prediction/model/train".to_owned(),
            },
        }
    };
}

#[test]
fn test_parse_config() {
    let f = File::open("testdata/sticker.conf").unwrap();
    let config = Config::from_toml_read(f).unwrap();
    assert_eq!(*BASIC_LABELER_CHECK, config);
}

use std::fs::File;

use lazy_static::lazy_static;
use sticker::tensorflow::ModelConfig;
use sticker::Layer;

use super::{Config, Embedding, EmbeddingAlloc, Embeddings, Input, Labeler, LabelerType, TomlRead};

lazy_static! {
    static ref BASIC_LABELER_CHECK: Config = Config {
        labeler: Labeler {
            labeler_type: LabelerType::Sequence(Layer::Feature("tf".to_string())),
            labels: "sticker.labels".to_owned(),
            read_ahead: Some(10),
        },
        input: Input {
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
            subwords: true,
        },
        model: ModelConfig {
            batch_size: 128,
            gpu_allow_growth: true,
            graph: "sticker.graph".to_owned(),
            parameters: "sticker.model".to_owned(),
            intra_op_parallelism_threads: 4,
            inter_op_parallelism_threads: 4,
        }
    };
}

#[test]
fn test_parse_config() {
    let f = File::open("testdata/sticker.conf").unwrap();
    let config = Config::from_toml_read(f).unwrap();
    assert_eq!(*BASIC_LABELER_CHECK, config);
}

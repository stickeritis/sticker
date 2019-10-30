use conllx::graph::{DepTriple, Sentence};
use ordered_float::OrderedFloat;
use petgraph::algo::tarjan_scc;

use super::DependencyEncoding;
use crate::encoder::EncodingProb;

static ROOT_RELATION: &str = "ROOT";

/// Attach orphan tokens to `head_idx`.
///
/// This function attaches orphan tokens to `head_idx`, with the
/// dependency labels of their highest probability encodings.
pub fn attach_orphans<'a, S, H>(labels: &[S], sentence: &mut Sentence, head_idx: usize)
where
    H: 'a + Clone,
    S: AsRef<[EncodingProb<DependencyEncoding<H>>]>,
{
    let token_indices: Vec<_> = (0..sentence.len())
        .filter(|&idx| sentence[idx].is_token())
        .collect();

    for (idx, encodings) in token_indices.into_iter().zip(labels) {
        if sentence.dep_graph().head(idx).is_none() {
            let relation = encodings.as_ref()[0].encoding().label().to_owned();
            sentence
                .dep_graph_mut()
                .add_deprel(DepTriple::new(head_idx, Some(relation), idx));
        }
    }
}

/// Break cycles in the graph.
///
/// Panics when a token does not have a head. To ensure that each
/// token has a head, apply `attach_orphans` to the dependency graph
/// before this function.
pub fn break_cycles(sent: &mut Sentence, root_idx: usize) {
    loop {
        let components = {
            tarjan_scc(sent.get_ref())
                .into_iter()
                .filter(|c| c.len() > 1)
                .collect::<Vec<_>>()
        };

        // We are done if there are no more cycles.
        if components.is_empty() {
            break;
        }

        for cycle in components.into_iter() {
            // Find the first token in the cycle, exclude the root
            // token to avoid self-cycles.
            let first_token = cycle
                .into_iter()
                .filter(|idx| idx.index() != root_idx)
                .min()
                .expect("Cannot get minimum, but iterator is non-empty")
                .index();

            // Reattach the token to the root.
            let head_rel = sent
                .dep_graph()
                .head(first_token)
                .expect("Token without a head")
                .relation()
                .map(ToOwned::to_owned);

            sent.dep_graph_mut()
                .add_deprel(DepTriple::new(root_idx, head_rel, first_token));
        }
    }
}

/// Find a candidate root token.
///
/// The token which with the highest-probability encoding with the
/// ROOT label is used.
fn find_root_candidate<'a, S, H, F>(labels: &[S], decode_fun: F) -> Option<(DepTriple<String>, f32)>
where
    H: 'a + Clone,
    S: AsRef<[EncodingProb<DependencyEncoding<H>>]>,
    F: Fn(usize, &DependencyEncoding<H>) -> Option<DepTriple<String>>,
{
    labels
        .iter()
        .enumerate()
        .filter_map(|(idx, encodings)| {
            encodings
                .as_ref()
                .iter()
                // Find encodings with a ROOT telation...
                .filter(|e| e.encoding().label() == ROOT_RELATION)
                // ...that can be decoded.
                .filter_map(|e| decode_fun(idx + 1, e.encoding()).map(|triple| (triple, e.prob())))
                .next()
        })
        .max_by_key(|(_, prob)| OrderedFloat(*prob))
}

/// Find the root token or create it if it does not exist.
///
/// If there is no root, we attach another token as root. We follow
/// the strategy suggested by Strzyz et al. 2019. We find the encoding
/// with a root attachment with the highest probability. And reattach
/// that token to root. If there is no such token, the first token of
/// the sentence becomes the root.
pub fn find_or_create_root<'a, S, H, F>(
    labels: &[S],
    sentence: &mut Sentence,
    decode_fun: F,
) -> usize
where
    H: 'a + Clone,
    S: AsRef<[EncodingProb<DependencyEncoding<H>>]>,
    F: Fn(usize, &DependencyEncoding<H>) -> Option<DepTriple<String>>,
{
    // If the sentence contains a token with root attachment, return
    // it.
    if let Some(root_idx) = first_root(sentence) {
        return root_idx;
    }

    // Find a suitable root token from the token encodings.  If there
    // is no such token, use the first token of the sentence.
    let triple = match find_root_candidate(labels, decode_fun) {
        Some((triple, _)) => triple,
        None => DepTriple::new(0, Some(ROOT_RELATION.to_owned()), 1),
    };

    // Attach the new root.
    let dependent = triple.dependent();
    sentence.dep_graph_mut().add_deprel(triple);
    dependent
}

/// Get the first root in the sentence.
fn first_root(sentence: &Sentence) -> Option<usize> {
    for idx in sentence
        .iter()
        .enumerate()
        .filter_map(|(idx, node)| node.token().map(|_| idx))
    {
        if let Some(triple) = sentence.dep_graph().head(idx) {
            if triple.head() == 0 {
                return Some(idx);
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use conllx::graph::{DepTriple, Sentence};
    use conllx::token::TokenBuilder;

    use super::{attach_orphans, break_cycles, find_or_create_root, first_root, ROOT_RELATION};
    use crate::encoder::deprel::{
        pos_position_table, DependencyEncoding, RelativePOS, RelativePOSEncoder,
    };
    use crate::encoder::{EncodingProb, SentenceEncoder};

    fn test_graph() -> Sentence {
        let mut sent = Sentence::new();
        sent.push(TokenBuilder::new("Die").pos("det").into());
        sent.push(TokenBuilder::new("AWO").pos("noun").into());
        sent.push(TokenBuilder::new("veruntreute").pos("verb").into());
        sent.push(TokenBuilder::new("Spendengeld").pos("noun").into());
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(2, Some("det"), 1));
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(3, Some("subj"), 2));
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(0, Some(ROOT_RELATION), 3));
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(3, Some("obj"), 4));

        sent
    }

    fn test_graph_cycle() -> Sentence {
        let mut sent = Sentence::new();
        sent.push(TokenBuilder::new("Die").pos("det").into());
        sent.push(TokenBuilder::new("AWO").pos("noun").into());
        sent.push(TokenBuilder::new("veruntreute").pos("verb").into());
        sent.push(TokenBuilder::new("Spendengeld").pos("noun").into());
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(2, Some("det"), 1));
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(1, Some("subj"), 2));
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(0, Some(ROOT_RELATION), 3));
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(3, Some("obj"), 4));

        sent
    }

    fn test_graph_no_root() -> Sentence {
        let mut sent = Sentence::new();
        sent.push(TokenBuilder::new("Die").pos("det").into());
        sent.push(TokenBuilder::new("AWO").pos("noun").into());
        sent.push(TokenBuilder::new("veruntreute").pos("verb").into());
        sent.push(TokenBuilder::new("Spendengeld").pos("noun").into());
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(2, Some("det"), 1));
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(3, Some("subj"), 2));
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(4, Some("foo"), 3));
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(3, Some("obj"), 4));

        sent
    }

    #[test]
    fn find_first_root() {
        let sent = test_graph();
        assert_eq!(first_root(&sent), Some(3));
    }

    #[test]
    fn attach_two_orphans() {
        let mut sent = Sentence::new();
        sent.push(TokenBuilder::new("Die").pos("det").into());
        sent.push(TokenBuilder::new("AWO").pos("noun").into());
        sent.push(TokenBuilder::new("veruntreute").pos("verb").into());
        sent.push(TokenBuilder::new("Spendengeld").pos("noun").into());
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(2, Some("det"), 1));
        sent.dep_graph_mut()
            .add_deprel(DepTriple::new(0, Some(ROOT_RELATION), 3));

        let encodings: Vec<_> = RelativePOSEncoder
            .encode(&test_graph())
            .unwrap()
            .into_iter()
            .map(|e| [EncodingProb::new(e, 1.)])
            .collect();

        attach_orphans(&encodings, &mut sent, 3);

        assert_eq!(sent, test_graph());
    }

    #[test]
    fn add_missing_root() {
        let mut sent = test_graph_no_root();

        let encodings: Vec<Vec<EncodingProb<DependencyEncoding<RelativePOS>>>> = vec![
            vec![EncodingProb::new(
                DependencyEncoding {
                    label: ROOT_RELATION.to_owned(),
                    head: RelativePOS::new(ROOT_RELATION, -1),
                },
                0.4,
            )],
            vec![],
            vec![
                EncodingProb::new(
                    DependencyEncoding {
                        label: "distractor".to_owned(),
                        head: RelativePOS::new(ROOT_RELATION, -1),
                    },
                    0.6,
                ),
                EncodingProb::new(
                    DependencyEncoding {
                        label: ROOT_RELATION.to_owned(),
                        head: RelativePOS::new(ROOT_RELATION, -1),
                    },
                    0.4,
                ),
            ],
            vec![EncodingProb::new(
                DependencyEncoding {
                    label: ROOT_RELATION.to_owned(),
                    head: RelativePOS::new(ROOT_RELATION, -1),
                },
                0.3,
            )],
        ];

        let pos_table = pos_position_table(&sent);
        find_or_create_root(&encodings, &mut sent, |idx, encoding| {
            RelativePOSEncoder::decode_idx(&pos_table, idx, encoding).ok()
        });

        assert_eq!(sent, test_graph());
    }

    #[test]
    fn break_simple_cycle() {
        let mut check = test_graph_cycle();
        // Token 1 is the leftmost token in the cycle and
        // should be reattached to the head.
        check
            .dep_graph_mut()
            .add_deprel(DepTriple::new(3, Some("det"), 1));

        // Detect cycle and break it.
        let mut sent = test_graph_cycle();
        break_cycles(&mut sent, 3);

        assert_eq!(sent, check);
    }
}

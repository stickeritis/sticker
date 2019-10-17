fn main() {
    protoc_rust::run(protoc_rust::Args {
        out_dir: "src",
        input: &[
            "protos/tensorflow/core/protobuf/cluster.proto",
            "protos/tensorflow/core/protobuf/config.proto",
            "protos/tensorflow/core/protobuf/debug.proto",
            "protos/tensorflow/core/protobuf/rewriter_config.proto",
            "protos/tensorflow/core/framework/allocation_description.proto",
            "protos/tensorflow/core/framework/attr_value.proto",
            "protos/tensorflow/core/framework/cost_graph.proto",
            "protos/tensorflow/core/framework/function.proto",
            "protos/tensorflow/core/framework/graph.proto",
            "protos/tensorflow/core/framework/node_def.proto",
            "protos/tensorflow/core/framework/op_def.proto",
            "protos/tensorflow/core/framework/resource_handle.proto",
            "protos/tensorflow/core/framework/step_stats.proto",
            "protos/tensorflow/core/framework/tensor.proto",
            "protos/tensorflow/core/framework/tensor_description.proto",
            "protos/tensorflow/core/framework/tensor_shape.proto",
            "protos/tensorflow/core/framework/types.proto",
            "protos/tensorflow/core/framework/versions.proto",
        ],
        includes: &["protos"],
        customize: protoc_rust::Customize {
            ..Default::default()
        },
    })
    .expect("protoc");
}

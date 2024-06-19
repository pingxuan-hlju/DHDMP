import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')

    # GCN HGNN 参数
    parser.add_argument('--latdim', default=1546, type=int, help='embedding size')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')

    parser.add_argument('--droprate', default=0.5, type=float, help='rate for dropout')
    parser.add_argument('--hyperNum', default=32, type=int, help='number of hyper edges')
    parser.add_argument('--node_type_dim', default=32, type=int)
    parser.add_argument('--drug_num', default=1373, type=int)
    parser.add_argument('--microbe_num', default=173, type=int)
    # Transformer参数
    parser.add_argument('--patch_size', default=50, type=int, help='size of patch')
    parser.add_argument('--attention_heads', default=6, type=int, help='heads of attention layer')
    parser.add_argument('--head_dim', default=50, type=int, help='per heads dim')
    parser.add_argument('--embed_size', default=1550, type=int, help='size of patch')  # 1550 是补0之后的维度
    parser.add_argument('--X_dim', default=50, type=int, help='every patch has X_dim')
    parser.add_argument('--embed_dropout', default=0.2, type=float, help='embed dropout')
    parser.add_argument('--attention_dropout', default=0.2, type=float, help='attention dropout')
    parser.add_argument('--depth_interact_attention', default=1, type=int, help='attention dropout')
    parser.add_argument('--depth_self_attention', default=1, type=int, help='attention dropout')
    parser.add_argument('--mlp_dim', default=512, type=int, help='MLP dim')

    return parser.parse_args()


args = parse_args()

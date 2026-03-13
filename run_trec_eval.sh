QRELS_FILE="/Users/payelsantra/Documents/PhD/Fusion-RAG/hotpotQA/qrels/qrels/extended_qrels_valid_withGT_TF_infonce_post_Z_minmax_post_LMDIR_mu10.txt"
# QRELS_FILE="/Users/payelsantra/Documents/PhD/Fusion-RAG/hotpotQA/qrels/hotpotqa_whole_qrels_valid.txt"
RUN_FILE="/Users/payelsantra/Documents/PhD/Fusion-RAG/hotpotQA/res/utility_valid_withGT_TF_infonce_post_Z_minmax_post_LMDIR_mu10.res"

# /Users/payelsantra/Documents/PhD/Fusion-RAG/hotpotQA/res/res/utility_valid_BM25_ridge_post_Z_minmax_post_BM25_k0.6_b0.5.res
# /Users/payelsantra/Documents/PhD/Fusion-RAG/hotpotQA/res/res/utility_valid_TF-IDF_ridge_post_Z_minmax_post_BM25_k0.6_n0.5.res
# /Users/payelsantra/Documents/PhD/Fusion-RAG/hotpotQA/res/res/utility_valid_TF_ridge_post_Z_minmax_postBM25_k0.6_b0.5.res


./trec_eval \
    -m map_cut.5,10,20 \
    -m ndcg_cut.5,10 \
    -m recall.5,10,50,100 \
    -m P.5,10 \
    -m recip_rank \
    $QRELS_FILE $RUN_FILE
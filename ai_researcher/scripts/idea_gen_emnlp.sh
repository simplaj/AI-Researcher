topic_names=("CSS" "Dialogue" "Discourse_Pragmatics" "Low_resource" "Ethics_Bias_Fairness" "Information_Extraction" "Information_Retrieval_Text_Mining" "Interpretability_Analysis")
ideas_n=5
methods=("prompting")

# Iterate over each seed from 1 to 1000
for seed in {1..200}; do
    # Iterate over each cache name 
    for topic in "${topic_names[@]}"; do
        # Iterate over each method 
        for method in "${methods[@]}"; do
            echo "Running grounded_idea_gen.py on: $topic with seed $seed"
            python3 src/grounded_idea_gen.py \
             --engine "claude-3-5-sonnet-20240620" \
             --paper_cache "../cache_results_claude_july/lit_review_emnlp/$topic.json" \
             --idea_cache "../cache_results_claude_july/ideas_emnlp/$topic.json" \
             --grounding_k 10 \
             --method "$method" \
             --ideas_n $ideas_n \
             --seed $seed \
             --RAG True >> logs/idea_gen_emnlp/idea_generation_${topic}_RAG.log 2>&1
        done
    done
done


## tmux 1
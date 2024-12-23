import gradio as gr
import subprocess
import os

def execute_command(command, env=None):
    """
    执行 shell 命令并实时返回输出。
    """
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in process.stdout:
        yield line
    process.wait()
    if process.returncode != 0:
        yield f"**<span style='color:red;'>命令失败，退出代码 {process.returncode}</span>**\n"

def run_workflow(
    api_key,
    engine,
    base_cache_dir,
    topic,
    max_paper_bank_size,
    print_all,
    run_lit_review,
    run_idea_gen,
    run_dedup,
    run_proposal_gen,
    run_proposal_ranking,
    run_proposal_filtering,
    ideas_n,
    methods,
    rag_values,
    seeds,
    similarity_threshold,
    seed_pp
):
    """
    执行整个工作流程的函数。
    使用生成器函数实时输出日志和更新进度。
    """
    log = ""  # 用于累积日志
    # 计算总步骤数
    total_steps = sum([run_lit_review, run_idea_gen, run_dedup, run_proposal_gen, run_proposal_ranking, run_proposal_filtering])
    current_step = 0

    # 创建基于主题的缓存目录
    sanitized_topic = "_".join(topic.strip().split(" "))
    topic_cache_dir = base_cache_dir
    paper_cache = os.path.join(topic_cache_dir, "lit_review", f"{sanitized_topic}.json")
    idea_cache = os.path.join(topic_cache_dir, "seed_ideas", f"{sanitized_topic}.json")
    dedup_cache_dir = os.path.join(topic_cache_dir, "ideas_dedup")
    project_proposal_cache_dir = os.path.join(topic_cache_dir, "project_proposals")
    experiment_plan_cache_dir = os.path.join(project_proposal_cache_dir, sanitized_topic)
    ranking_score_dir = os.path.join(topic_cache_dir, "ranking")
    passed_cache_dir = os.path.join(topic_cache_dir, "project_proposals_passed")

    # 自动创建必要的目录
    os.makedirs(os.path.join(topic_cache_dir, "lit_review"), exist_ok=True)
    os.makedirs(os.path.join(topic_cache_dir, "seed_ideas"), exist_ok=True)
    os.makedirs(dedup_cache_dir, exist_ok=True)
    os.makedirs(project_proposal_cache_dir, exist_ok=True)
    os.makedirs(experiment_plan_cache_dir, exist_ok=True)
    if run_proposal_ranking or run_proposal_filtering:
        os.makedirs(ranking_score_dir, exist_ok=True)
    if run_proposal_filtering:
        os.makedirs(passed_cache_dir, exist_ok=True)

    # 设置 API Key 环境变量（如果需要）
    env = os.environ.copy()
    if api_key:
        env["API_KEY"] = api_key
    env["HF_ENDPOINT"] = 'https://hf-mirror.com'

    progress = gr.Progress()
    # 步骤 1: 文献综述
    if run_lit_review:
        current_step += 1
        progress_percentage = (current_step / total_steps) * 100
        progress(progress_percentage)
        log += "### 步骤 1: 文献综述（Literature Review）\n"
        lit_review_cmd = [
            "python3", "src/lit_review.py",
            "--engine", engine,
            "--mode", "topic",
            "--topic_description", topic,
            "--cache_name", paper_cache,
            "--max_paper_bank_size", str(max_paper_bank_size)
        ]
        if print_all:
            lit_review_cmd.append("--print_all")
        log += f"**运行命令:** `{ ' '.join(lit_review_cmd) }`\n"
        yield log
        for line in execute_command(lit_review_cmd, env=env):
            log += line
            yield log
        log += "**文献综述完成。**\n\n"
        yield log
    else:
        log += "### 步骤 1: 文献综述（Literature Review）已跳过。\n\n"
        yield log

    # 步骤 2: 生成有依据的创意
    if run_idea_gen:
        # 检查依赖
        if not run_lit_review and not os.path.exists(paper_cache):
            log += "**⚠️ 跳过步骤 2，因为步骤 1 未运行且缺少必要的文献综述缓存文件。**\n\n"
            yield log
        else:
            current_step += 1
            progress_percentage = (current_step / total_steps) * 100
            progress(progress_percentage)
            log += "### 步骤 2: 生成有依据的创意（Grounded Idea Generation）\n"
            yield log
            for seed in range(1, seeds + 1):
                for method in [m.strip() for m in methods.split(',')]:
                    for rag in [r.strip() for r in rag_values.split(',')]:
                        log += f"**运行** `grounded_idea_gen.py` **主题:** {topic} **种子:** {seed} **RAG:** {rag}\n"
                        grounded_idea_cmd = [
                            "python3", "src/grounded_idea_gen.py",
                            "--engine", engine,
                            "--paper_cache", paper_cache,
                            "--idea_cache", idea_cache,
                            "--grounding_k", "10",
                            "--method", method,
                            "--ideas_n", str(ideas_n),
                            "--seed", str(seed),
                            "--RAG", rag
                        ]
                        log += f"**运行命令:** `{ ' '.join(grounded_idea_cmd) }`\n"
                        yield log
                        for line in execute_command(grounded_idea_cmd, env=env):
                            log += line
                            yield log
            log += "**生成有依据的创意完成。**\n\n"
            yield log
    else:
        log += "### 步骤 2: 生成有依据的创意（Grounded Idea Generation）已跳过。\n\n"
        yield log

    # 步骤 3: 创意去重
    if run_dedup:
        # 检查依赖
        if not run_idea_gen and not os.path.exists(idea_cache):
            log += "**⚠️ 跳过步骤 3，因为步骤 2 未运行且缺少必要的创意缓存文件。**\n\n"
            yield log
        else:
            current_step += 1
            progress_percentage = (current_step / total_steps) * 100
            progress(progress_percentage)
            log += "### 步骤 3: 创意去重（Idea Deduplication）\n"
            yield log

            # 分析语义相似性
            log += f"**运行** `analyze_ideas_semantic_similarity.py` **主题:** {topic}\n"
            analyze_cmd = [
                "python3", "src/analyze_ideas_semantic_similarity.py",
                "--cache_dir", os.path.join(topic_cache_dir, "seed_ideas"),
                "--cache_name", sanitized_topic,
                "--save_similarity_matrix"
            ]
            log += f"**运行命令:** `{ ' '.join(analyze_cmd) }`\n"
            yield log

            for line in execute_command(analyze_cmd, env=env):
                log += line
                yield log

            # 进行去重
            log += f"**运行** `dedup_ideas.py` **主题:** {topic}\n"
            dedup_cmd = [
                "python3", "src/dedup_ideas.py",
                "--cache_dir", os.path.join(topic_cache_dir, "seed_ideas"),
                "--cache_name", sanitized_topic,
                "--dedup_cache_dir", dedup_cache_dir,
                "--similarity_threshold", str(similarity_threshold)
            ]
            log += f"**运行命令:** `{ ' '.join(dedup_cmd) }`\n"
            yield log

            for line in execute_command(dedup_cmd, env=env):
                log += line
                yield log

            log += "**创意去重完成。**\n\n"
            yield log
    else:
        log += "### 步骤 3: 创意去重（Idea Deduplication）已跳过。\n\n"
        yield log

    # 步骤 4: 点子生成
    if run_proposal_gen:
        # 检查依赖
        if not run_dedup and not os.path.exists(dedup_cache_dir):
            log += "**⚠️ 跳过步骤 4，因为步骤 3 未运行且缺少必要的去重缓存目录。**\n\n"
            yield log
        else:
            current_step += 1
            progress_percentage = (current_step / total_steps) * 100
            progress(progress_percentage)
            log += "### 步骤 4: 点子生成（Project Proposal Generation）\n"
            yield log

            experiment_cmd = [
                "python3", "src/experiment_plan_gen.py",
                "--engine", engine,
                "--idea_cache_dir", dedup_cache_dir + '/',
                "--cache_name", sanitized_topic,
                "--experiment_plan_cache_dir", project_proposal_cache_dir + '/',
                "--idea_name", "all",
                "--seed", str(seed_pp),
                "--method", "prompting"
            ]
            log += f"**运行命令:** `{ ' '.join(experiment_cmd) }`\n"
            yield log

            for line in execute_command(experiment_cmd, env=env):
                log += line
                yield log

            log += "**点子生成完成。**\n\n"
            yield log
    else:
        log += "### 步骤 4: 点子生成（Project Proposal Generation）已跳过。\n\n"
        yield log

    # 步骤 5: 点子排名
    if run_proposal_ranking:
        # 检查依赖
        if not run_proposal_gen and not os.path.exists(experiment_plan_cache_dir):
            log += "**⚠️ 跳过步骤 5，因为步骤 4 未运行且缺少必要的点子缓存目录。**\n\n"
            yield log
        else:
            current_step += 1
            progress_percentage = (current_step / total_steps) * 100
            progress(progress_percentage)
            log += "### 步骤 5: 点子排名（Project Proposal Ranking）\n"
            yield log

            ranking_score_dir = os.path.join(topic_cache_dir, "ranking")
            cache_names = ["factuality_prompting_method"]  # 可以根据需要动态生成

            for cache_name in cache_names:
                log += f"**运行** `tournament_ranking.py` **cache_name:** {cache_name}\n"
                tournament_ranking_cmd = [
                    "python3", "src/tournament_ranking.py",
                    "--engine", engine,
                    "--experiment_plan_cache_dir", project_proposal_cache_dir + '/',
                    "--cache_name", cache_name,
                    "--ranking_score_dir", ranking_score_dir,
                    "--max_round", "5"
                ]
                log += f"**运行命令:** `{ ' '.join(tournament_ranking_cmd) }`\n"
                yield log

                for line in execute_command(tournament_ranking_cmd, env=env):
                    log += line
                    yield log
            
            log += "**点子排名完成。**\n\n"
            yield log
    else:
        log += "### 步骤 5: 点子排名（Project Proposal Ranking）已跳过。\n\n"
        yield log

    # 步骤 6: 点子过滤
    if run_proposal_filtering:
        # 检查依赖
        if not run_proposal_ranking and not os.path.exists(os.path.join(ranking_score_dir, "factuality_prompting_method", "round_5.json")):
            log += "**⚠️ 跳过步骤 6，因为步骤 5 未运行且缺少必要的排名分数文件。**\n\n"
            yield log
        else:
            current_step += 1
            progress_percentage = (current_step / total_steps) * 100
            progress(progress_percentage)
            log += "### 步骤 6: 点子过滤（Project Proposal Filtering）\n"
            yield log

            cache_dir = project_proposal_cache_dir + '/'
            passed_cache_dir = os.path.join(topic_cache_dir, "project_proposals_passed")
            cache_names = ["factuality_prompting_method"]  # 可以根据需要动态生成

            for cache_name in cache_names:
                log += f"**运行** `filter_ideas.py` **cache_name:** {cache_name}\n"
                filter_ideas_cmd = [
                    "python3", "src/filter_ideas.py",
                    "--engine", engine,
                    "--cache_dir", cache_dir,
                    "--cache_name", cache_name,
                    "--passed_cache_dir", passed_cache_dir,
                    "--score_file", f"{ranking_score_dir}/{cache_name}/round_5.json"
                ]
                log += f"**运行命令:** `{ ' '.join(filter_ideas_cmd) }`\n"
                yield log

                for line in execute_command(filter_ideas_cmd, env=env):
                    log += line
                    yield log
            
            log += "**点子过滤完成。**\n\n"
            yield log
    else:
        log += "### 步骤 6: 点子过滤（Project Proposal Filtering）已跳过。\n\n"
        yield log

    # 更新进度条到100%
    progress(100)
    # log += "### 工作流程完成。\n"
    # log += "**注意**：点子排名和过滤步骤已跳过。如需执行这些步骤，请手动添加相关脚本和界面组件。"
    yield log

with gr.Blocks() as demo:
    gr.Markdown("# 科研点子王")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 全局设置")

            api_key = gr.Textbox(
                label="API Key (可选)",
                type="password",
                placeholder="输入您的 API Key（如果需要）"
            )
            engine = gr.Textbox(
                label="Engine",
                value="qwen-turbo"
            )
            base_cache_dir = gr.Textbox(
                label="基础缓存目录",
                value="../cache_results_test/",
                placeholder="例如: ../cache_results_test/"
            )
            topic = gr.Textbox(
                label="主题描述",
                value="novel prompting methods to improve large language models’ performance on multilingual tasks or low-resource languages and vernacular languages",
                lines=2
            )

            gr.Markdown("## 文献综述设置")
            max_paper_bank_size = gr.Number(
                label="最大论文库大小",
                value=50
            )
            print_all = gr.Checkbox(
                label="打印所有日志",
                value=True
            )
            run_lit_review = gr.Checkbox(
                label="运行文献综述",
                value=True
            )

            gr.Markdown("## 生成有依据的创意设置")
            ideas_n = gr.Number(
                label="创意批次数量",
                value=5
            )
            methods = gr.Textbox(
                label="方法",
                value="prompting",
                placeholder="例如: prompting"
            )
            rag_values = gr.Textbox(
                label="RAG 值",
                value="True,False",
                placeholder="例如: True,False"
            )
            seeds = gr.Number(
                label="种子数量",
                value=2
            )
            run_idea_gen = gr.Checkbox(
                label="运行生成有依据的创意",
                value=True
            )

            gr.Markdown("## 创意去重设置")
            similarity_threshold = gr.Number(
                label="相似性阈值",
                value=0.8
            )
            run_dedup = gr.Checkbox(
                label="运行创意去重",
                value=True
            )

            gr.Markdown("## 点子生成设置")
            seed_pp = gr.Number(
                label="点子生成的种子",
                value=2024
            )
            run_proposal_gen = gr.Checkbox(
                label="运行点子生成",
                value=True
            )

            gr.Markdown("## 点子排名设置")
            run_proposal_ranking = gr.Checkbox(
                label="运行点子排名",
                value=False
            )

            gr.Markdown("## 点子过滤设置")
            run_proposal_filtering = gr.Checkbox(
                label="运行点子过滤",
                value=False
            )

            run_workflow_btn = gr.Button("运行整个工作流程")

        with gr.Column(scale=2):
            gr.Markdown("## 执行日志")
            # 使用 Markdown 组件以支持丰富的文本格式
            output = gr.Markdown(
                value=""
            )

    run_workflow_btn.click(
        run_workflow,
        inputs=[
            api_key,
            engine,
            base_cache_dir,
            topic,
            max_paper_bank_size,
            print_all,
            run_lit_review,
            run_idea_gen,
            run_dedup,
            run_proposal_gen,
            run_proposal_ranking,
            run_proposal_filtering,
            ideas_n,
            methods,
            rag_values,
            seeds,
            similarity_threshold,
            seed_pp
        ],
        outputs=output
    )

    gr.Markdown("**注意**：请确保所有 Python 脚本和缓存目录路径正确，且服务器环境已正确配置。")

demo.launch()

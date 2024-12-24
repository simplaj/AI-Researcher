import json
import gradio as gr
import subprocess
import os

def execute_command(command, env=None):
    """
    执行 shell 命令并同步逐行返回输出（完全单线程，无多进程）。
    """
    try:
        # 同步执行命令并捕获输出
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, check=False)
        
        # 按行返回结果
        for line in result.stdout.splitlines():
            yield line
        
        # 检查返回码
        if result.returncode != 0:
            yield f"**<span style='color:red;'>命令失败，退出代码 {result.returncode}</span>**\n"
    except Exception as e:
        yield f"**<span style='color:red;'>执行命令时发生错误: {e}</span>**\n"


def display_multiple_json(directory_path):
    """
    从指定目录中读取所有 JSON 文件并将内容显示为 Markdown。
    """
    all_json_content = ""
    if not os.path.exists(directory_path):
        return "**<span style='color:red;'>No passed project proposal. Try to detaily describe your `Topics` or increase `batchs per time` and `Run times`</span>**" 
    
    # 遍历目录中的所有 JSON 文件
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".json"):
            file_path = os.path.join(directory_path, file_name)
            
            # 尝试读取 JSON 文件内容
            with open(file_path, "r", encoding="utf-8") as file:
                json_content = json.load(file)
                
                # 添加文件名作为标题
                all_json_content += f"#### {file_name}\n"
                
                # 将 JSON 转为格式化字符串
                json_string = json.dumps(json_content, indent=4, ensure_ascii=False)
                
                # 添加到 Markdown 中，使用代码块
                all_json_content += f"```json\n{json_string}\n```\n\n"
                    
    # 如果没有任何 JSON 文件
    if not all_json_content:
        all_json_content = "未找到任何 JSON 文件。"
    
    return all_json_content


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
    result_json = ""
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
        progress_percentage = (current_step / total_steps)
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
        yield [log, result_json]
        for line in execute_command(lit_review_cmd, env=env):
            log += line
            yield [log, result_json]
        log += "**文献综述完成(Done!)。**\n\n"
        # 加载 JSON 数据
        with open(paper_cache, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        # 将 JSON 数据转换为字符串并嵌入代码块中
        result_json = f"```json\n{json.dumps(json_data, indent=4)}\n```"
    else:
        log += "### 步骤 1: 文献综述已跳过。（Literature Review Skipped）\n\n"
        yield [log, result_json]

    # 步骤 2: 生成有依据的科研点子
    if run_idea_gen:
        # 检查依赖
        if not run_lit_review and not os.path.exists(paper_cache):
            log += "**⚠️ 跳过步骤 2，因为步骤 1 未运行且缺少必要的文献综述缓存文件。(Skipped, lack of literature review files produced by step 1.)**\n\n"
            yield [log, result_json]
        else:
            current_step += 1
            progress_percentage = (current_step / total_steps)
            progress(progress_percentage)
            log += "### 步骤 2: 生成有依据的科研点子（Grounded Idea Generation）\n"
            yield [log, result_json]
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
                        yield [log, result_json]
                        for line in execute_command(grounded_idea_cmd, env=env):
                            log += line
                            yield [log, result_json]
            log += "**生成有依据的科研点子完成(Done!)。**\n\n"
            # 加载 JSON 数据
            with open(idea_cache, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # 将 JSON 数据转换为字符串并嵌入代码块中
            result_json = f"```json\n{json.dumps(json_data, indent=4)}\n```"
            yield [log, result_json]
    else:
        log += "### 步骤 2: 生成有依据的科研点子已跳过。（Grounded Idea Generation Skipped）\n\n"
        yield [log, result_json]

    # 步骤 3: 科研点子去重
    if run_dedup:
        # 检查依赖
        if not run_idea_gen and not os.path.exists(idea_cache):
            log += "**⚠️ 跳过步骤 3，因为步骤 2 未运行且缺少必要的科研点子缓存文件。(Skipped, lack of idea files produced by step 2.)**\n\n"
            yield [log, result_json]
        else:
            current_step += 1
            progress_percentage = (current_step / total_steps)
            progress(progress_percentage)
            log += "### 步骤 3: 科研点子去重（Idea Deduplication）\n"
            yield [log, result_json]

            # 分析语义相似性
            log += f"**运行** `analyze_ideas_semantic_similarity.py` **主题:** {topic}\n"
            analyze_cmd = [
                "python3", "src/analyze_ideas_semantic_similarity.py",
                "--cache_dir", os.path.join(topic_cache_dir, "seed_ideas"),
                "--cache_name", sanitized_topic,
                "--save_similarity_matrix"
            ]
            log += f"**运行命令:** `{ ' '.join(analyze_cmd) }`\n"
            yield [log, result_json]

            for line in execute_command(analyze_cmd, env=env):
                log += line
                yield [log, result_json]

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
            yield [log, result_json]

            for line in execute_command(dedup_cmd, env=env):
                log += line
                yield [log, result_json]

            log += "**科研点子去重完成(Done!)。**\n\n"
            # 加载 JSON 数据
            with open(os.path.join(dedup_cache_dir, f'{sanitized_topic}.json'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # 将 JSON 数据转换为字符串并嵌入代码块中
            result_json = f"```json\n{json.dumps(json_data, indent=4)}\n```"
            yield [log, result_json]
    else:
        log += "### 步骤 3: 科研点子去重已跳过。（Idea Deduplication Skipped）\n\n"
        yield [log, result_json]

    # 步骤 4: 科研计划生成
    if run_proposal_gen:
        # 检查依赖
        if not run_dedup and not os.path.exists(dedup_cache_dir):
            log += "**⚠️ 跳过步骤 4，因为步骤 3 未运行且缺少必要的去重缓存目录。(Skipped, lack of deduplicate files produced by step 3.)**\n\n"
            yield [log, result_json]
        else:
            current_step += 1
            progress_percentage = (current_step / total_steps)
            progress(progress_percentage)
            log += "### 步骤 4: 科研计划生成（Project Proposal Generation）\n"
            yield [log, result_json]

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
            yield [log, result_json]

            for line in execute_command(experiment_cmd, env=env):
                log += line
                yield [log, result_json]

            log += "**科研计划生成完成(Done!)。**\n\n"
            result_json = display_multiple_json(os.path.join(project_proposal_cache_dir, sanitized_topic))
            yield [log, result_json]
    else:
        log += "### 步骤 4: 科研计划生成已跳过。（Project Proposal Generation Skipped）\n\n"
        yield [log, result_json]

    # 步骤 5: 科研计划排名
    if run_proposal_ranking:
        # 检查依赖
        if not run_proposal_gen and not os.path.exists(experiment_plan_cache_dir):
            log += "**⚠️ 跳过步骤 5，因为步骤 4 未运行且缺少必要的科研点子缓存目录。(Skipped, lack of project proposal files produced by step 4.)**\n\n"
            yield [log, result_json]
        else:
            current_step += 1
            progress_percentage = (current_step / total_steps)
            progress(progress_percentage)
            log += "### 步骤 5: 科研计划排名（Project Proposal Ranking）\n"
            yield [log, result_json]

            ranking_score_dir = os.path.join(topic_cache_dir, "ranking")

            log += f"**运行** `tournament_ranking.py` **cache_name:** {sanitized_topic}\n"
            tournament_ranking_cmd = [
                "python3", "src/tournament_ranking.py",
                "--engine", engine,
                "--experiment_plan_cache_dir", project_proposal_cache_dir + '/',
                "--cache_name", sanitized_topic,
                "--ranking_score_dir", ranking_score_dir,
                "--max_round", "5"
            ]
            log += f"**运行命令:** `{ ' '.join(tournament_ranking_cmd) }`\n"
            yield [log, result_json]

            for line in execute_command(tournament_ranking_cmd, env=env):
                log += line
                yield [log, result_json]
        
            log += "**科研计划排名完成(Done!)。**\n\n"
            # 加载 JSON 数据
            with open(os.path.join(ranking_score_dir, f'{sanitized_topic}', 'top_ideas.json'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # 将 JSON 数据转换为字符串并嵌入代码块中
            result_json = f"```json\n{json.dumps(json_data, indent=4)}\n```"
            yield [log, result_json]
    else:
        log += "### 步骤 5: 科研计划排名已跳过。（Project Proposal Ranking Skipped）\n\n"
        yield [log, result_json]

    # 步骤 6: 科研计划过滤
    if run_proposal_filtering:
        # 检查依赖
        if not run_proposal_ranking and not os.path.exists(os.path.join(ranking_score_dir, sanitized_topic, "round_5.json")):
            log += "**⚠️ 跳过步骤 6，因为步骤 5 未运行且缺少必要的排名分数文件。(Skipped, lack of ranking files produced by step 5.)**\n\n"
            yield [log, result_json]
        else:
            current_step += 1
            progress_percentage = (current_step / total_steps)
            progress(progress_percentage)
            log += "### 步骤 6: 科研计划过滤（Project Proposal Filtering）\n"
            yield [log, result_json]

            cache_dir = project_proposal_cache_dir + '/'
            passed_cache_dir = os.path.join(topic_cache_dir, "project_proposals_passed")

            log += f"**运行** `filter_ideas.py` **cache_name:** {sanitized_topic}\n"
            filter_ideas_cmd = [
                "python3", "src/filter_ideas.py",
                "--engine", engine,
                "--cache_dir", cache_dir,
                "--cache_name", sanitized_topic,
                "--passed_cache_dir", passed_cache_dir + '/',
                "--score_file", f"{ranking_score_dir}/{sanitized_topic}/round_5.json"
            ]
            log += f"**运行命令:** `{ ' '.join(filter_ideas_cmd) }`\n"
            yield [log, result_json]

            for line in execute_command(filter_ideas_cmd, env=env):
                log += line
                yield [log, result_json]
            
            log += "**科研计划过滤完成(Done!)。**\n\n"
            result_json = display_multiple_json(os.path.join(passed_cache_dir, sanitized_topic))
            yield [log, result_json]
    else:
        log += "### 步骤 6: 科研计划过滤已跳过。（Project Proposal Filtering Skipped）\n\n"
        yield [log, result_json]

    # 更新进度条到100%
    progress(100)
    # log += "### 工作流程完成(Done!)。\n"
    # log += "**注意**：科研计划排名和过滤步骤已跳过。如需执行这些步骤，请手动添加相关脚本和界面组件。"
    yield [log, result_json]

def get_download_links(cache_dir, topic):
    sanitized_topic = "_".join(topic.strip().split(" "))
    files = {
        "文献综述": os.path.join(cache_dir, "lit_review", f"{sanitized_topic}.json"),
        "科研点子": os.path.join(cache_dir, "seed_ideas", f"{sanitized_topic}.json"),
        "去重点子": os.path.join(cache_dir, "ideas_dedup", f"{sanitized_topic}.json"),
        "科研计划排名": os.path.join(cache_dir, "ranking", sanitized_topic, "top_ideas.json"),
    }
    links = {}
    for key, path in files.items():
        if os.path.exists(path):
            links[key] = path
    return links

def download_links_display(links):
    result = []
    for name, path in links.items():
        result.append(display_file_content(path))
    return result[0], result[1], result[2], result[3]


def display_file_content(file_path):
    """
    读取并格式化 JSON 文件内容。
    """
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 格式化为段落形式
        formatted_content = f"```json\n{json.dumps(data, indent=4)}\n```"
        return formatted_content
    else:
        return f"**<span style='color:red;'>文件 {file_path} 不存在(File {file_path} dose not exists)。</span>**"
    

def fresh(cache_dir, topic):
    link = get_download_links(cache_dir, topic)
    return download_links_display(link)

with gr.Blocks(title="科研小点子（AI Researcher Spark）") as demo:
    with gr.Row():
        # 左侧放置较小宽度，用于显示 Logo
        with gr.Column(scale=1, min_width=80):
            gr.Image(
                value="sparklogo.png",
                show_label=False,
                show_download_button=False,
                show_fullscreen_button=False,
                show_share_button=False
            )
        # 右侧放置主要内容
        with gr.Column(scale=9):
            pass
    gr.Markdown('''# 科研小点子 (AI Researcher Spark)\n''')
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 全局设置(Global setting)")

            api_key = gr.Textbox(
                label="API Key (Optional)",
                type="password",
                placeholder="Input API Key（如果需要）",
                visible=False
            )
            engine = gr.Textbox(
                label="Engine",
                value="qwen-turbo"
            )
            base_cache_dir = gr.Textbox(
                label="基础缓存目录(Cache dir)",
                value="../cache_results_test/",
                placeholder="e.g.: ../cache_results_test/"
            )
            topic = gr.Textbox(
                label="主题描述(Topics)",
                value="novel prompting methods to improve large language models’ performance on multilingual tasks or low-resource languages and vernacular languages",
                lines=2
            )

            gr.Markdown("## 文献综述设置(Literature review setting)")
            max_paper_bank_size = gr.Number(
                label="最大论文库大小(Max paper bank size)",
                value=50
            )
            print_all = gr.Checkbox(
                label="打印所有日志(Print all logs)",
                value=True
            )
            run_lit_review = gr.Checkbox(
                label="运行文献综述(Need or not)",
                value=True
            )

            gr.Markdown("## 生成有依据的科研点子设置(Grounded idea generation setting)")
            ideas_n = gr.Number(
                label="科研点子批次数量(batchs per time)",
                value=5
            )
            methods = gr.Textbox(
                label="方法(method)",
                value="prompting",
                placeholder="e.g.: prompting"
            )
            rag_values = gr.Textbox(
                label="RAG",
                value="True",
                placeholder="e.g.: True,False"
            )
            seeds = gr.Number(
                label="种子数量(Run times)",
                value=2
            )
            run_idea_gen = gr.Checkbox(
                label="运行生成有依据的科研点子(Need or not)",
                value=True
            )

            gr.Markdown("## 科研点子去重设置(Idea deduplication)")
            similarity_threshold = gr.Number(
                label="相似性阈值(Similarity threshold)",
                value=0.8
            )
            run_dedup = gr.Checkbox(
                label="运行科研点子去重(Need or not)",
                value=True
            )

            gr.Markdown("## 科研计划生成设置(Project proposal generation setting)")
            seed_pp = gr.Number(
                label="科研计划生成的种子(Seed)",
                value=2024
            )
            run_proposal_gen = gr.Checkbox(
                label="运行科研计划生成(Need or not)",
                value=True
            )

            gr.Markdown("## 科研计划排名设置(Project proposal ranking setting)")
            run_proposal_ranking = gr.Checkbox(
                label="运行科研计划排名(Need or not)",
                value=False
            )

            gr.Markdown("## 科研计划过滤设置(Project proposal filtering setting)")
            run_proposal_filtering = gr.Checkbox(
                label="运行科研计划过滤(Need or not)",
                value=False
            )

            run_workflow_btn = gr.Button("运行整个工作流程(RUN)")

        with gr.Column(scale=2):
            gr.Markdown("## 执行日志(Logs)")
            # 使用 Markdown 组件以支持丰富的文本格式
            output = gr.Markdown(
                value="",
                height=300
            )
            gr.Markdown("## 结果(Results)")
            res = gr.Markdown(
                value="",
                height=900
            )
            down_btn = gr.Button("展示中间结果(Show Results)")
            gr.Markdown("## 文献综述(Literature Review)")
            lit = gr.Markdown(label='文献综述', height=300)
            gr.Markdown("## 科研点子(Ideas)")
            ideas = gr.Markdown(label='科研点子', height=300)
            gr.Markdown("## 去重点子(Ideas Deduplicate)")
            idea_dedup = gr.Markdown(label='去重点子', height=300)
            gr.Markdown("## 科研计划排名(Project Proposal Ranking)")
            ranking = gr.Markdown(label='科研计划排名', height=300)
            
    down_btn.click(
        fresh,
        inputs=[base_cache_dir, topic],
        outputs=[lit, ideas, idea_dedup, ranking]
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
        outputs=[output, res]
    )

    gr.Markdown("**注意**：请确保所有 Python 脚本和缓存目录路径正确，且服务器环境已正确配置。")

demo.launch(server_name='127.0.0.1', server_port=7860, allowed_paths=['/home/tzh/Project/AI-Researcher-Spark'])

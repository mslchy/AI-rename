import json
import os
import re
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from openai import OpenAI

# =========================
# 配置类与全局缓存
# =========================


@dataclass
class MinerUConfig:
    """
    MinerU API 配置
    文档参考: https://mineru.net/
    """

    api_token: str
    model_version: str = "vlm"  # 可选: vlm / pipeline
    enable_formula: bool = True
    enable_table: bool = True
    language: str = "ch"  # 默认中文


@dataclass
class ModelConfig:
    """
    大模型推理配置
    支持 OpenAI 格式 (含 Ollama/LM Studio) 和 Google Gemini

    - provider: openai / ollama / lm_studio / gemini
    - api_key: API 密钥（本地模型可为占位符，但字段仍需提供）
    - model: 模型名称 (如 gpt-4o-mini, llama3, gemini-1.5-flash)
    - api_base: OpenAI 兼容接口地址，本地模型必填 (如 http://localhost:11434/v1)
    """

    provider: str
    api_key: str
    model: str
    api_base: Optional[str] = None


# OpenAI 兼容客户端缓存，避免重复创建
_openai_client_cache: Dict[Tuple[str, str, str], OpenAI] = {}


# =========================
# 工具函数
# =========================


def sanitize_filename(name: str) -> str:
    """
    清洗文件名，移除系统非法字符，保留大多数字符，限制长度。
    兼顾可读性与跨平台安全性。
    """
    # 替换 Windows/Linux 文件名非法字符
    cleaned = re.sub(r'[\\/*?:"<>|]', "_", name)
    # 移除控制字符
    cleaned = re.sub(r"[\x00-\x1f]", "", cleaned)
    # 去掉首尾空格和点
    cleaned = cleaned.strip(" .")
    # 限制长度
    return cleaned[:100] or "untitled"


def ensure_unique_path(path: str) -> str:
    """
    若目标路径已存在，则自动追加 _1, _2 后缀，直到唯一。
    """
    if not os.path.exists(path):
        return path

    stem, ext = os.path.splitext(path)
    counter = 1
    new_path = f"{stem}_{counter}{ext}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{stem}_{counter}{ext}"
    return new_path


def clean_json_text(text: str) -> str:
    """
    清洗大模型返回的文本，尽可能提取纯 JSON 字符串：
    1. 优先提取 ```json ... ``` 或 ``` ... ``` 代码块
    2. 其次尝试从首个 '{' 到最后一个 '}' 的子串
    """
    # 1. 提取 ```json ... ``` 或 ``` ... ``` 内部内容
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)

    # 2. 截取最外层 {} 块
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]

    return text.strip()


def default_metadata() -> Dict:
    """兜底元数据模板，避免整个流程因为单个文件失败而中断。"""
    return {
        "doc_type": "unknown",
        "title": "unknown",
        "org_or_author": "unknown",
        "year": "0000",
        "topic_keywords": [],
        "parse_ok": False,
    }


def default_rename_rule(metadata: Dict, original_basename: str) -> str:
    """
    默认命名规则：
    年份_标题_类型
    - year 缺失时用 '0000'
    - title 缺失时回退到原始文件名（不含扩展名）
    - doc_type 缺失时用 'doc'
    """
    year = metadata.get("year") or "0000"
    title = metadata.get("title") or original_basename
    doc_type = metadata.get("doc_type") or "doc"
    return f"{year}_{title}_{doc_type}"


def rename_rule_org_year_title(metadata: Dict, original_basename: str) -> str:
    """
    备用命名规则 2：
    机构_年份_标题
    - org_or_author 缺失时用 'unknown_org'
    - year 缺失时 '0000'
    - title 缺失时回退原始文件名
    """
    org = metadata.get("org_or_author") or "unknown_org"
    year = metadata.get("year") or "0000"
    title = metadata.get("title") or original_basename
    return f"{org}_{year}_{title}"


def rename_rule_keyword_year_title(metadata: Dict, original_basename: str) -> str:
    """
    备用命名规则 3：
    第一关键词_年份_标题
    - topic_keywords 为空时用 'nokey'
    """
    keywords = metadata.get("topic_keywords") or []
    first_kw = ""
    if isinstance(keywords, list) and keywords:
        first_kw = str(keywords[0])
    first_kw = first_kw or "nokey"
    year = metadata.get("year") or "0000"
    title = metadata.get("title") or original_basename
    return f"{first_kw}_{year}_{title}"


def get_rename_rule_by_choice(choice: str):
    """
    根据用户输入选择命名规则函数。
    默认返回 default_rename_rule。
    """
    if choice == "2":
        return rename_rule_org_year_title
    if choice == "3":
        return rename_rule_keyword_year_title
    return default_rename_rule


# =========================
# MinerU：申请上传、上传文件、轮询结果
# =========================


def request_batch_upload_links(file_paths: Iterable[str], config: MinerUConfig) -> Dict:
    """
    向 MinerU 申请批量上传链接。
    注意：单次最多 200 个文件。
    返回数据中应包含 batch_id 和 file_urls。
    """
    timestamp = int(time.time())
    payload_files = [
        {"name": os.path.basename(path), "data_id": f"file_{idx}_{timestamp}"}
        for idx, path in enumerate(file_paths)
    ]

    data = {
        "files": payload_files,
        "model_version": config.model_version,
        "enable_formula": config.enable_formula,
        "enable_table": config.enable_table,
        "language": config.language,
    }

    print(f"📡 正在申请 {len(payload_files)} 个文件的上传链接...")
    try:
        response = requests.post(
            "https://mineru.net/api/v4/file-urls/batch",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.api_token}",
            },
            json=data,
            timeout=60,
        )
        response.raise_for_status()
        result = response.json()
    except Exception as exc:
        raise RuntimeError(f"申请上传链接失败: {exc}")

    if result.get("code") != 0:
        raise RuntimeError(f"获取上传链接失败: {result.get('msg') or result}")
    return result["data"]


def upload_files_to_urls(file_paths: Iterable[str], file_urls: Iterable[str]) -> None:
    """
    将本地文件上传到 MinerU 提供的预签名 URL。
    单文件失败不阻塞其他文件，并记录错误信息。
    """

    def _upload(one_file: str, one_url: str) -> Tuple[str, bool, str]:
        try:
            with open(one_file, "rb") as f:
                resp = requests.put(one_url, data=f, timeout=300)
            if resp.status_code == 200:
                return one_file, True, ""
            return one_file, False, f"HTTP {resp.status_code}: {resp.text}"
        except Exception as exc:
            return one_file, False, str(exc)

    file_list = list(file_paths)
    url_list = list(file_urls)
    if len(file_list) != len(url_list):
        raise ValueError("文件数量与上传链接数量不匹配")

    max_workers = min(8, len(file_list)) or 1
    print("🚀 开始并发上传文件...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_upload, fp, url) for fp, url in zip(file_list, url_list)]
        for fut in as_completed(futures):
            file_path, ok, err = fut.result()
            name = os.path.basename(file_path)
            if ok:
                print(f"✅ 上传成功: {name}")
            else:
                print(f"❌ 上传失败: {name} -> {err}")

    print("🏁 所有文件上传流程结束。")


def poll_batch_results(
    batch_id: str,
    config: MinerUConfig,
    interval: int = 10,
    max_attempts: int = 120,
) -> List[Dict]:
    """
    轮询批量任务状态，直到全部文档解析完成或超时。

    容错策略：
    - MinerU 返回 code != 0 时仅打印告警，等待下一轮
    - 网络异常也只打印告警并重试
    - 达到重试上限后抛出 TimeoutError
    """
    url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    headers = {"Authorization": f"Bearer {config.api_token}"}

    print(f"⏳ 开始轮询任务 Batch ID: {batch_id}")
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=30)
            resp.raise_for_status()
            result = resp.json()

            if result.get("code") != 0:
                msg = result.get("msg") or str(result)
                print(f"⚠️ 轮询接口返回错误 (尝试 {attempt}/{max_attempts}): {msg}")
                last_error = msg
                time.sleep(interval)
                continue

            extract_list = result.get("data", {}).get("extract_result", [])
            if not extract_list:
                print(f"⚠️ 未获取到解析结果列表 (尝试 {attempt}/{max_attempts})")
                time.sleep(interval)
                continue

            states = [item.get("state") for item in extract_list]
            done_count = states.count("done")
            failed_count = states.count("failed")
            unfinished = [s for s in states if s not in ("done", "failed")]

            if not unfinished:
                print(f"\n🎉 解析完成! 成功: {done_count}, 失败: {failed_count}")
                return extract_list

            print(
                f"\r[{attempt}/{max_attempts}] 处理中... 完成: {done_count}, "
                f"失败: {failed_count}, 剩余: {len(unfinished)}",
                end="",
                flush=True,
            )
            time.sleep(interval)

        except Exception as exc:
            last_error = str(exc)
            print(f"\n⚠️ 轮询网络异常 (尝试 {attempt}/{max_attempts}): {exc}")
            time.sleep(interval)

    raise TimeoutError(f"任务轮询超时，最后错误信息: {last_error}")


# =========================
# ZIP 下载与 Markdown 提取
# =========================


def download_zip(url: str, target_path: str) -> None:
    """下载 MinerU 生成的 ZIP 文件到指定路径。"""
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(target_path, "wb") as f:
        for chunk in resp.itercontent(chunk_size=8192):
            if chunk:
                f.write(chunk)


def extract_markdown_from_zip(zip_path: str, output_dir: str, extract_all: bool = False) -> str:
    """
    从 ZIP 包中提取第一份 Markdown 文件并返回其路径。

    - extract_all=False: 只将第一份 Markdown 平铺到 output_dir 根部。
    - extract_all=True: 将压缩包完整解压到独立子目录，并在其中查找 Markdown。
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        md_files = [name for name in zf.namelist() if name.lower().endswith(".md")]
        if not md_files:
            raise FileNotFoundError("压缩包中未找到 Markdown 文件")

        first_md = md_files[0]

        if not extract_all:
            # 只抽出 Markdown 文件到根目录，并用 ensure_unique_path 避免冲突
            base_name = os.path.basename(first_md)
            base_output = os.path.join(output_dir, base_name)
            output_path = ensure_unique_path(base_output)

            zf.extract(first_md, output_dir)
            extracted_path = os.path.join(output_dir, first_md)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if extracted_path != output_path:
                shutil.move(extracted_path, output_path)
            return output_path
        else:
            # 完整解压到独立子目录，再在其中查找第一份 Markdown
            subdir_name = os.path.splitext(os.path.basename(zip_path))[0]
            subdir_path = os.path.join(output_dir, subdir_name)
            os.makedirs(subdir_path, exist_ok=True)
            zf.extractall(subdir_path)

            # 在子目录树中搜索第一份 .md
            chosen_md_path = None
            for root, _, files in os.walk(subdir_path):
                for f in files:
                    if f.lower().endswith(".md"):
                        chosen_md_path = os.path.join(root, f)
                        break
                if chosen_md_path:
                    break

            if not chosen_md_path:
                raise FileNotFoundError("完整解压后未找到 Markdown 文件")

            return chosen_md_path


def _download_and_extract_single_markdown(
    item: Dict,
    workspace: str,
    extract_all: bool = False,
) -> Tuple[str, Optional[str]]:
    """
    下载并解压单个文件的 ZIP，返回 (file_name, markdown_path 或 None)。
    在上层以并发方式调用。
    """
    file_name = item.get("file_name", "unknown.pdf")
    state = item.get("state")
    if state != "done":
        print(f"⏭️ 跳过未成功文件: {file_name} (状态: {state}, 错误: {item.get('err_msg', '')})")
        return file_name, None

    zip_url = item.get("full_zip_url")
    if not zip_url:
        print(f"⚠️ 文件 {file_name} 无 zip 链接，跳过")
        return file_name, None

    try:
        local_zip = os.path.join(workspace, f"{os.path.splitext(file_name)[0]}.zip")
        download_zip(zip_url, local_zip)
        md_path = extract_markdown_from_zip(local_zip, workspace, extract_all=extract_all)
        print(f"📄 Markdown 提取完成: {file_name} -> {md_path}")
        return file_name, md_path
    except Exception as exc:
        print(f"❌ 下载或解压 {file_name} 失败: {exc}")
        return file_name, None


def fetch_markdown_results(
    extract_results: List[Dict],
    workspace: str,
    extract_all: bool = False,
) -> Dict[str, str]:
    """
    为每个解析完成的文件下载 ZIP，提取 Markdown 路径。
    采用并发方式提升整体速度。
    返回映射: {原始文件名: markdown_path}
    """
    os.makedirs(workspace, exist_ok=True)
    markdown_map: Dict[str, str] = {}

    if not extract_results:
        return markdown_map

    max_workers = min(8, len(extract_results)) or 1
    print("\n🚀 开始并发下载 ZIP 并提取 Markdown...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_download_and_extract_single_markdown, item, workspace, extract_all)
            for item in extract_results
        ]
        for fut in as_completed(futures):
            file_name, md_path = fut.result()
            if md_path:
                markdown_map[file_name] = md_path

    print(f"✅ Markdown 提取完成，共 {len(markdown_map)} 个文件。")
    return markdown_map


# =========================
# 大模型调用与 JSON 解析
# =========================


def build_prompt(markdown_text: str) -> str:
    """构造提示词，引导模型输出规范 JSON。"""
    return (
        "你是一个专业的文档元数据提取助手。请阅读以下 Markdown 内容，"
        "生成一个严格的 JSON 对象，字段说明如下：\n"
        "1. doc_type: 文档类型（例如：论文、报告、通知、合同、课件、书籍等）\n"
        "2. title: 文档标题\n"
        "3. org_or_author: 发布机构或作者（优先机构，若无则作者）\n"
        "4. year: 发布年份（4位数字字符串，如 '2023'，找不到则返回 '0000'）\n"
        "5. topic_keywords: 3-5 个主题关键词的字符串数组\n"
        "请只返回 JSON，不要包含额外的解释、自然语言或 Markdown 代码块标记。\n\n"
        "文档内容如下：\n"
        f"{markdown_text}"
    )


def _get_cached_openai_client(config: ModelConfig) -> OpenAI:
    """
    获取并缓存 OpenAI 兼容客户端，减少重复连接开销。
    缓存 key = (provider, api_base 或 '', api_key)
    """
    key = (config.provider.lower(), config.api_base or "", config.api_key)
    if key not in _openai_client_cache:
        _openai_client_cache[key] = OpenAI(api_key=config.api_key, base_url=config.api_base)
    return _openai_client_cache[key]


def _call_openai_like_model(prompt: str, config: ModelConfig) -> str:
    """
    通过 OpenAI 兼容接口（OpenAI/ollama/LM Studio 等）获取回复。
    """
    client = _get_cached_openai_client(config)
    response = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,  # 低温度保证格式更稳定
        max_tokens=1000,
    )
    return response.choices[0].message.content


def _call_gemini_model(prompt: str, config: ModelConfig) -> str:
    """调用 Gemini HTTP 接口（官方 REST），返回文本内容。"""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model}:generateContent"
    params = {"key": config.api_key}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(url, params=params, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini 返回为空: {data}")
    return candidates[0]["content"]["parts"][0]["text"]


def generate_metadata(markdown_text: str, model_cfg: ModelConfig) -> Dict:
    """
    根据 Markdown 内容调用大模型，生成结构化元数据。

    策略：
    - 只截取前 10k 字符，避免上下文溢出
    - 使用 clean_json_text 提取 JSON
    - JSON 解析失败时返回默认元数据，而不是中断流程
    - 解析成功时补充 parse_ok=True
    """
    truncated = markdown_text[:10000]
    if len(markdown_text) > 10000:
        truncated += "\n...(content truncated for metadata extraction)..."

    prompt = build_prompt(truncated)

    try:
        provider = model_cfg.provider.lower()
        if provider in {"openai", "ollama", "lm_studio"}:
            raw_text = _call_openai_like_model(prompt, model_cfg)
        elif provider == "gemini":
            raw_text = _call_gemini_model(prompt, model_cfg)
        else:
            raise ValueError(f"不支持的模型提供方: {model_cfg.provider}")

        json_str = clean_json_text(raw_text)
        metadata = json.loads(json_str)
        if not isinstance(metadata, dict):
            raise ValueError("模型返回的 JSON 非对象类型")

        # 标记解析成功
        metadata.setdefault("parse_ok", True)
        return metadata

    except Exception as exc:
        print(f"⚠️ 元数据解析失败，使用默认模板兜底: {exc}")
        return default_metadata()


# =========================
# 重命名逻辑
# =========================


def rename_files_with_metadata(
    directory: str,
    metadata_map: Dict[str, Dict],
    mode: str = "copy",
    rename_rule=default_rename_rule,
) -> None:
    """
    根据元数据批量重命名文件。

    - directory: 原始 PDF 目录
    - metadata_map: {原始文件名: 元数据 dict}
    - mode: "overwrite" 在原目录改名；"copy" 复制到子目录 rename_output
    - rename_rule: (metadata, original_basename) -> 新文件名（不含扩展名）
    """
    if mode == "overwrite":
        target_dir = directory
    else:
        target_dir = os.path.join(directory, "rename_output")

    os.makedirs(target_dir, exist_ok=True)

    for original_name, metadata in metadata_map.items():
        src_path = os.path.join(directory, original_name)
        if not os.path.exists(src_path):
            print(f"⚠️ 源文件不存在，跳过: {src_path}")
            continue

        original_basename = os.path.splitext(original_name)[0]
        new_base = sanitize_filename(rename_rule(metadata, original_basename))
        new_path = os.path.join(target_dir, new_base + ".pdf")
        new_path = ensure_unique_path(new_path)

        try:
            if mode == "overwrite":
                # 若目标路径与源路径实际相同，则无需动作
                if os.path.abspath(src_path) == os.path.abspath(new_path):
                    print(f"ℹ️ 文件名未变化，跳过: {original_name}")
                    continue
                os.rename(src_path, new_path)
                print(f"✅ 重命名成功: {original_name} -> {os.path.basename(new_path)}")
            else:
                shutil.copy2(src_path, new_path)
                print(f"✅ 复制并重命名: {original_name} -> {os.path.basename(new_path)}")
        except OSError as exc:
            print(f"⚠️ 重命名/复制文件失败: {src_path} -> {new_path}, 错误: {exc}")


# =========================
# 主流程：上传 -> 轮询 -> 下载 -> LLM -> 重命名
# =========================


def run_pipeline(
    pdf_dir: str,
    mineru_cfg: MinerUConfig,
    model_cfg: ModelConfig,
    rename_mode: str = "copy",
    rename_rule=default_rename_rule,
    extract_all_zip: bool = False,
) -> None:
    """
    完整处理流程：
    1) 扫描本地 PDF
    2) 向 MinerU 申请上传链接并上传
    3) 轮询解析状态
    4) 下载 ZIP 并提取 Markdown
    5) 调大模型生成元数据 JSON
    6) 基于元数据重命名/复制 PDF

    参数：
    - rename_mode: "overwrite" 或 "copy"，默认安全的 "copy"
    - rename_rule: 命名规则函数
    - extract_all_zip: 是否完整解压 MinerU ZIP 到独立子目录（True 时保留全部 MinerU 输出）
    """
    if not os.path.isdir(pdf_dir):
        print(f"❌ 目录不存在: {pdf_dir}")
        return

    pdf_files = [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        print("❌ 指定目录下未找到 PDF 文件")
        return

    # MinerU 单次限制 200 个文件，这里做简单截断并提示
    if len(pdf_files) > 200:
        print("⚠️ 检测到 PDF 文件超过 200 个，MinerU 单次批处理限制为 200。")
        print("💡 当前将仅处理前 200 个文件，如需处理全部，请手动分批运行。")
        pdf_files = pdf_files[:200]

    # 1) 申请上传链接
    upload_info = request_batch_upload_links(pdf_files, mineru_cfg)
    batch_id = upload_info["batch_id"]
    file_urls = upload_info["file_urls"]

    # 2) 上传文件
    upload_files_to_urls(pdf_files, file_urls)

    # 3) 轮询解析状态
    extract_results = poll_batch_results(batch_id, mineru_cfg)

    # 4) 下载 ZIP 并提取 Markdown
    workspace = os.path.join(pdf_dir, "mineru_workspace")
    markdown_map = fetch_markdown_results(extract_results, workspace, extract_all=extract_all_zip)

    # 5) 调用大模型生成元数据
    metadata_map: Dict[str, Dict] = {}

    def _process_metadata(file_name: str, md_path: str) -> Tuple[str, Optional[Dict]]:
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                md_text = f.read()
            metadata = generate_metadata(md_text, model_cfg)

            json_path = os.path.join(
                workspace,
                f"{os.path.splitext(file_name)[0]}_metadata.json",
            )
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(metadata, jf, ensure_ascii=False, indent=2)
            print(f"📝 元数据写入: {json_path}")
            return file_name, metadata
        except Exception as exc:
            print(f"❌ 处理 {file_name} 的元数据失败: {exc}")
            return file_name, None

    if not markdown_map:
        print("⚠️ 未获取到任何 Markdown 文件，流程结束。")
        return

    print("\n🚀 开始并发调用大模型生成元数据...")
    with ThreadPoolExecutor(max_workers=min(6, len(markdown_map))) as executor:
        futures = [
            executor.submit(_process_metadata, fn, mp)
            for fn, mp in markdown_map.items()
        ]
        for fut in as_completed(futures):
            file_name, metadata = fut.result()
            if metadata:
                metadata_map[file_name] = metadata

    if not metadata_map:
        print("⚠️ 未成功生成任何元数据，跳过重命名。")
        return

    # 6) 重命名 / 复制 PDF
    print("\n🔄 开始基于元数据进行重命名/复制...")
    rename_files_with_metadata(pdf_dir, metadata_map, mode=rename_mode, rename_rule=rename_rule)
    print("\n✨ 所有任务处理完成！")


# =========================
# CLI 交互入口
# =========================


def main():
    print("=== 文档智能重命名工具 (MinerU + LLM) ===\n")

    pdf_dir = input("📂 请输入 PDF 目录路径: ").strip()
    if not os.path.isdir(pdf_dir):
        print("❌ 路径不存在或不是目录")
        return

    mineru_token = input("🔑 请输入 MinerU API Token: ").strip()
    if not mineru_token:
        print("❌ MinerU Token 不能为空")
        return

    print("\n🤖 选择模型提供方:")
    print("1. OpenAI (含兼容接口，如 DeepSeek、Moonshot 等)")
    print("2. Ollama (本地)")
    print("3. LM Studio (本地)")
    print("4. Gemini (Google)")
    choice = input("请输入序号 (默认 1): ").strip()

    provider_map = {
        "1": "openai",
        "2": "ollama",
        "3": "lm_studio",
        "4": "gemini",
    }
    provider = provider_map.get(choice, "openai")

    api_key = input("🔑 请输入模型 API Key (本地模型可填占位符): ").strip()
    if not api_key:
        api_key = "placeholder"

    model_name = input("🧠 请输入模型名称 (如 gpt-4o-mini, llama3, gemini-1.5-flash): ").strip()
    if not model_name:
        print("❌ 模型名称不能为空")
        return

    api_base: Optional[str] = None
    if provider in {"ollama", "lm_studio"}:
        default_url = "http://localhost:11434/v1" if provider == "ollama" else "http://localhost:1234/v1"
        inp = input(f"🌐 请输入 Base URL (默认 {default_url}): ").strip()
        api_base = inp or default_url
    elif provider == "openai":
        inp = input("🌐 如需自定义 OpenAI Base URL，请输入 (留空使用官方): ").strip()
        api_base = inp or None
    else:
        # Gemini 不需要 base_url
        api_base = None

    mode_input = input("\n📝 重命名模式: [1] 覆盖原文件 [2] 复制到新目录 (默认 2): ").strip()
    rename_mode = "overwrite" if mode_input == "1" else "copy"

    print("\n📛 选择命名规则:")
    print("1. 年份_标题_类型 (默认)")
    print("2. 机构_年份_标题")
    print("3. 第一关键词_年份_标题")
    rule_choice = input("请输入序号 (默认 1): ").strip() or "1"
    rename_rule = get_rename_rule_by_choice(rule_choice)

    extract_choice = input("\n📦 是否完整解压 MinerU ZIP 到独立子目录? (y/N): ").strip().lower()
    extract_all_zip = extract_choice in {"y", "yes"}

    mineru_cfg = MinerUConfig(api_token=mineru_token)
    model_cfg = ModelConfig(
        provider=provider,
        api_key=api_key,
        model=model_name,
        api_base=api_base,
    )

    run_pipeline(
        pdf_dir,
        mineru_cfg,
        model_cfg,
        rename_mode=rename_mode,
        rename_rule=rename_rule,
        extract_all_zip=extract_all_zip,
    )


if __name__ == "__main__":
    main()

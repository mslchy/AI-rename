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


# -------------------------
# MinerU 相关配置与数据结构
# -------------------------
@dataclass
class MinerUConfig:
    """用于管理 MinerU API 所需的信息。"""

    api_token: str
    model_version: str = "vlm"
    enable_formula: bool = True
    enable_table: bool = True
    language: str = "ch"


@dataclass
class ModelConfig:
    """
    管理用户选择的推理模型信息。

    - provider: openai / ollama / lm_studio / gemini
    - api_key: API 密钥（本地模型可为空字符串，但字段仍需提供）
    - api_base: openai 兼容接口的 base url（本地模型必填），例如 http://localhost:11434/v1
    - model: 模型名称，例如 gpt-4o-mini、llama3、gemini-1.5-flash
    """

    provider: str
    api_key: str
    model: str
    api_base: Optional[str] = None


# -------------------------
# MinerU: 申请上传、上传文件、轮询结果
# -------------------------
def request_batch_upload_links(file_paths: Iterable[str], config: MinerUConfig) -> Dict:
    """向 MinerU 申请批量上传链接，返回包含 batch_id 与上传 URL 的数据。"""

    payload_files = [{"name": os.path.basename(path), "data_id": f"data_{idx}"} for idx, path in enumerate(file_paths)]
    data = {
        "files": payload_files,
        "model_version": config.model_version,
        "enable_formula": config.enable_formula,
        "enable_table": config.enable_table,
        "language": config.language,
    }

    response = requests.post(
        "https://mineru.net/api/v4/file-urls/batch",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.api_token}"},
        json=data,
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("code") != 0:
        raise RuntimeError(f"获取上传链接失败: {result}")
    return result["data"]


def upload_files_to_urls(file_paths: Iterable[str], file_urls: Iterable[str]) -> None:
    """将本地文件上传到 MinerU 预签名 URL，失败不阻塞其他文件。"""

    def _upload(one_file: str, one_url: str) -> Tuple[str, bool, str]:
        try:
            with open(one_file, "rb") as f:
                resp = requests.put(one_url, data=f, timeout=300)
            if resp.status_code != 200:
                return one_file, False, resp.text
            return one_file, True, ""
        except Exception as exc:  # 捕获单文件错误，避免打断其他上传
            return one_file, False, str(exc)

    file_list = list(file_paths)
    url_list = list(file_urls)
    max_workers = min(8, len(file_list)) or 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_upload, fp, url) for fp, url in zip(file_list, url_list)]
        for fut in as_completed(futures):
            file_path, ok, err = fut.result()
            if ok:
                print(f"Upload success -> {file_path}")
            else:
                print(f"Upload failed -> {file_path}: {err}")


def poll_batch_results(batch_id: str, config: MinerUConfig, interval: int = 10, max_attempts: int = 60) -> List[Dict]:
    """轮询批量任务状态，直到全部文档解析完成或超时。"""

    url = f"https://mineru.net/api/v4/extract-results/batch/{batch_id}"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.api_token}"}

    for attempt in range(max_attempts):
        response = requests.get(url, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        if result.get("code") != 0:
            raise RuntimeError(f"查询任务失败: {result}")

        extract_list = result.get("data", {}).get("extract_result", [])
        unfinished = [item for item in extract_list if item.get("state") not in {"done", "failed"}]
        if not unfinished:
            return extract_list

        states = ", ".join(f"{item['file_name']} -> {item['state']}" for item in unfinished)
        print(f"Polling {attempt + 1}/{max_attempts}, unfinished: {states}")
        time.sleep(interval)

    raise TimeoutError("轮询超时，仍有文件未完成解析")


# -------------------------
# 解析结果下载与 Markdown 提取
# -------------------------
def download_zip(url: str, target_path: str) -> None:
    """下载 MinerU 生成的 ZIP 文件到指定路径。"""

    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(target_path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


def extract_markdown_from_zip(zip_path: str, output_dir: str) -> str:
    """从 ZIP 包中提取第一份 Markdown 文件并返回其路径。"""

    with zipfile.ZipFile(zip_path, "r") as zf:
        markdown_files = [name for name in zf.namelist() if name.lower().endswith(".md")]
        if not markdown_files:
            raise FileNotFoundError("压缩包中未找到 Markdown 文件")
        first_md = markdown_files[0]
        # 如存在同名文件，追加计数后缀避免冲突
        base_output = os.path.join(output_dir, os.path.basename(first_md))
        output_path = base_output
        counter = 1
        while os.path.exists(output_path):
            stem, ext = os.path.splitext(base_output)
            output_path = f"{stem}_{counter}{ext}"
            counter += 1
        zf.extract(first_md, output_dir)
        # 如果 ZIP 内有子目录，统一移动到输出目录根部
        extracted_path = os.path.join(output_dir, first_md)
        if extracted_path != output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.move(extracted_path, output_path)
        return output_path


def fetch_markdown_results(extract_results: List[Dict], workspace: str) -> Dict[str, str]:
    """为每个解析完成的文件下载 ZIP，提取 Markdown 路径。"""

    os.makedirs(workspace, exist_ok=True)
    markdown_map: Dict[str, str] = {}

    for item in extract_results:
        file_name = item.get("file_name", "unknown.pdf")
        state = item.get("state")
        if state != "done":
            print(f"跳过未完成的文件 {file_name}, state={state}, error={item.get('err_msg', '')}")
            continue

        zip_url = item.get("full_zip_url")
        if not zip_url:
            print(f"文件 {file_name} 无 zip 链接，跳过")
            continue

        try:
            local_zip = os.path.join(workspace, f"{os.path.splitext(file_name)[0]}.zip")
            download_zip(zip_url, local_zip)
            markdown_path = extract_markdown_from_zip(local_zip, workspace)
            markdown_map[file_name] = markdown_path
            print(f"Markdown extracted for {file_name} -> {markdown_path}")
        except Exception as exc:
            print(f"下载或解压 {file_name} 失败: {exc}")

    return markdown_map


# -------------------------
# 大模型调用：生成结构化 JSON
# -------------------------
def build_prompt(markdown_text: str) -> str:
    """构造统一提示，指导模型输出规范 JSON。"""

    return (
        "你是一名文档元数据提取助手。请阅读以下 Markdown 内容，生成 JSON 对象，字段含义：\n"
        "- doc_type: 文档类型（报告、论文、通知等）\n"
        "- title: 文档标题或核心名称\n"
        "- org_or_author: 机构或作者\n"
        "- year: 4 位年份\n"
        "- topic_keywords: 2-5 个主题关键词数组\n"
        "严格返回 JSON，不要包含其他文本。Markdown 内容如下：\n" + markdown_text
    )


def parse_json_response(text: str) -> Dict:
    """将模型返回的字符串解析为 JSON 对象，并在失败时给出可读错误。"""

    # 去掉可能存在的 Markdown 代码块包裹
    text = re.sub(r"```[a-zA-Z]*\n|```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 尝试截取第一个 JSON 块
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = text[start : end + 1]
            return json.loads(cleaned)
        raise ValueError(f"模型返回内容不是合法 JSON: {text}")


_openai_client_cache: Dict[Tuple[str, Optional[str]], OpenAI] = {}


def _get_cached_openai_client(config: ModelConfig) -> OpenAI:
    """缓存并复用 OpenAI 兼容客户端，减少重复连接开销。"""

    key = (config.api_base or "", config.api_key)
    if key not in _openai_client_cache:
        _openai_client_cache[key] = OpenAI(api_key=config.api_key, base_url=config.api_base)
    return _openai_client_cache[key]


def call_openai_like_model(prompt: str, config: ModelConfig) -> str:
    """通过 OpenAI 兼容接口（OpenAI/ollama/LM Studio 等）获取回复。"""

    client = _get_cached_openai_client(config)
    response = client.chat.completions.create(
        model=config.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content


def call_gemini_model(prompt: str, config: ModelConfig) -> str:
    """调用 Gemini HTTP 接口（使用官方 REST），返回文本内容。"""

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{config.model}:generateContent"
    params = {"key": config.api_key}
    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, params=params, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise RuntimeError(f"Gemini 返回为空: {data}")
    return candidates[0]["content"]["parts"][0]["text"]


def generate_metadata(markdown_text: str, model_cfg: ModelConfig) -> Dict:
    """根据 Markdown 内容调用用户指定模型，生成结构化元数据。"""

    # 为避免长文档导致模型上下文溢出，仅截取前 10k 字符
    prompt = build_prompt(markdown_text[:10000])
    if model_cfg.provider.lower() in {"openai", "ollama", "lm_studio"}:
        raw_text = call_openai_like_model(prompt, model_cfg)
    elif model_cfg.provider.lower() == "gemini":
        raw_text = call_gemini_model(prompt, model_cfg)
    else:
        raise ValueError("不支持的模型提供方，请选择 openai/ollama/lm_studio/gemini")

    metadata = parse_json_response(raw_text)
    return metadata


# -------------------------
# 基于元数据的重命名/复制
# -------------------------
def sanitize_filename(name: str) -> str:
    """清洗文件名中的非法字符，并限制长度。"""

    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5_\-]+", "_", name).strip("._")
    return cleaned[:80] or "untitled"


def ensure_unique_path(path: str) -> str:
    """若目标文件已存在，追加计数后缀避免覆盖。"""

    if not os.path.exists(path):
        return path

    stem, ext = os.path.splitext(path)
    counter = 1
    new_path = f"{stem}_{counter}{ext}"
    while os.path.exists(new_path):
        counter += 1
        new_path = f"{stem}_{counter}{ext}"
    return new_path


def default_rename_rule(metadata: Dict, original_basename: str) -> str:
    """默认命名规则：年份_标题_类型，缺失字段自动回退。"""

    year = metadata.get("year") or "yyyy"
    title = metadata.get("title") or original_basename
    doc_type = metadata.get("doc_type") or "doc"
    return f"{year}_{title}_{doc_type}"


def rename_files_with_metadata(
    directory: str,
    metadata_map: Dict[str, Dict],
    rename_rule=default_rename_rule,
    mode: str = "overwrite",
) -> None:
    """
    根据元数据批量重命名文件。

    - mode="overwrite": 在原目录直接改名。
    - mode="copy": 生成 rename 目录，复制并使用新名字。
    """

    target_dir = directory if mode == "overwrite" else os.path.join(directory, "rename")
    os.makedirs(target_dir, exist_ok=True)

    for original_name, metadata in metadata_map.items():
        base = os.path.splitext(original_name)[0]
        new_base = sanitize_filename(rename_rule(metadata, base))
        new_path = ensure_unique_path(os.path.join(target_dir, new_base + ".pdf"))

        src_path = os.path.join(directory, original_name)
        if not os.path.exists(src_path):
            print(f"源文件不存在，跳过: {src_path}")
            continue

        if mode == "overwrite":
            os.rename(src_path, new_path)
        else:
            shutil.copy2(src_path, new_path)
        print(f"重命名完成: {original_name} -> {new_path}")


# -------------------------
# 主流程：上传 -> 轮询 -> 解析 -> 命名
# -------------------------
def run_pipeline(
    pdf_dir: str,
    mineru_cfg: MinerUConfig,
    model_cfg: ModelConfig,
    rename_mode: str = "overwrite",
) -> None:
    """完整处理流程：批量上传 PDF、获取 Markdown、调用模型生成 JSON，并执行重命名。"""

    pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    if not pdf_files:
        raise FileNotFoundError("指定目录下未找到 PDF 文件")

    # 1) 向 MinerU 申请上传链接并上传
    upload_info = request_batch_upload_links(pdf_files, mineru_cfg)
    batch_id = upload_info["batch_id"]
    upload_files_to_urls(pdf_files, upload_info["file_urls"])

    # 2) 轮询解析状态并获取结果列表
    extract_results = poll_batch_results(batch_id, mineru_cfg)

    # 3) 下载并提取 Markdown
    workspace = os.path.join(pdf_dir, "mineru_outputs")
    markdown_map = fetch_markdown_results(extract_results, workspace)

    # 4) 将 Markdown 发送到指定大模型，得到 JSON 元数据
    metadata_map: Dict[str, Dict] = {}

    def _process_metadata(file_name: str, md_path: str) -> Tuple[str, Optional[Dict]]:
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                md_text = f.read()
            metadata = generate_metadata(md_text, model_cfg)

            json_path = os.path.join(workspace, f"{os.path.splitext(file_name)[0]}_metadata.json")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(metadata, jf, ensure_ascii=False, indent=2)
            print(f"元数据写入: {json_path}")
            return file_name, metadata
        except Exception as exc:
            print(f"处理 {file_name} 的元数据失败: {exc}")
            return file_name, None

    with ThreadPoolExecutor(max_workers=min(6, len(markdown_map) or 1)) as executor:
        futures = [executor.submit(_process_metadata, fn, mp) for fn, mp in markdown_map.items()]
        for fut in as_completed(futures):
            file_name, metadata = fut.result()
            if metadata:
                metadata_map[file_name] = metadata

    # 5) 基于元数据重命名或复制
    rename_files_with_metadata(pdf_dir, metadata_map, mode=rename_mode)


def main():
    """简单的交互入口，示范如何配置 MinerU 与模型参数。"""

    pdf_dir = input("请输入包含 PDF 的目录路径：").strip()
    token = input("请输入 MinerU API Token：").strip()

    # 用户选择模型提供方
    provider = input("选择模型提供方(openai/ollama/lm_studio/gemini)：").strip().lower()
    model_name = input("请输入模型名称（例如 gpt-4o-mini 或 llama3）：").strip()
    api_key = input("请输入模型 API Key（本地模型可填占位符）：").strip()
    api_base = None
    if provider in {"ollama", "lm_studio"}:
        api_base = input("请输入本地模型的 OpenAI 兼容接口地址，例如 http://localhost:11434/v1：").strip()
    elif provider == "openai":
        # 默认使用官方接口，可根据需要自定义 base_url
        custom_base = input("如需自定义 OpenAI base_url，请输入（留空则使用默认）：").strip()
        api_base = custom_base or None
    else:
        # gemini 不需要 base_url，但必须提供正确模型名和 API Key
        api_base = None

    rename_mode = input("重命名方式：overwrite(覆盖) / copy(另存到 rename 目录)：").strip() or "overwrite"

    mineru_cfg = MinerUConfig(api_token=token)
    model_cfg = ModelConfig(provider=provider, api_key=api_key, api_base=api_base, model=model_name)

    run_pipeline(pdf_dir, mineru_cfg, model_cfg, rename_mode=rename_mode)


if __name__ == "__main__":
    main()

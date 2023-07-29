from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
import pydantic_core
from PyPDF2 import PdfReader
import os
import sys
import json
from pydantic.dataclasses import dataclass
import argparse

anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


@dataclass
class StructuredOutput:
    file_name: str
    topic: str
    processes: list[str]
    features: list[str]
    properties: list[str]


def eprintln(arg):
    "Print to stderr"
    print(arg, file=sys.stderr)


def eprint(arg):
    "Print to stderr, suitable for streaing content"
    print(arg, flush=True, end="", file=sys.stderr)


def parse_from_end_token(claude_output: str, end_token: str) -> str:
    """Expects a string that has some structured output and then {end_token}"""
    end_index = claude_output.index(end_token)
    if end_index < 0:
        raise IndexError(f"Did not find {end_token} in the output")
    return claude_output[0:end_index]


def parse_structured_output(claude_output: str) -> str:
    """Expects a string that is a jsonobject, and then </output>"""
    return parse_from_end_token(claude_output, "</output>")


def get_pdf_content(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pdf_content = ""
    for page in reader.pages:
        text = page.extract_text()
        pdf_content = pdf_content + "\n" + text
    return pdf_content


def build_get_topic_prompt(pdf_content: str) -> str:
    return (
        f"{HUMAN_PROMPT} "
        f"The following is an academic paper in the materials science field related "
        f"to manufacturing and designing materials."
        "in order to affect certain properties of the material."
        "The document be provided within <document></document> tags. "
        "Here is the document:"
        f"<document>{pdf_content}</document>"
        "Please tell me the topic -- general field of materials science that the paper "
        "is related to. In order to relay this information to me, please include your "
        "response in <topic></topic> tags. Please limit your topic to no more then "
        "5 words. Example responses are as follows, in <example></example> tags."
        "<example><topic>Stainless Steel</topic></example>"
        "<example><topic>Batterries</topic></example>"
        "<example><topic>Lithium Ion Batteries</topic></example>"
        "<example><topic>Concrete</topic></example>"
        "<example><topic>Polymers</topic></example>"
        f"{AI_PROMPT} <topic>"
    )


def parse_topic_output(claude_output: str) -> str:
    """Expects a string that ends with </topic>"""
    return parse_from_end_token(claude_output, "</topic>")


def get_document_topic(document_path: str) -> str:
    pdf_content = get_pdf_content(document_path)
    my_prompt = build_get_topic_prompt(pdf_content)
    stream = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=1000,
        prompt=my_prompt,
        stream=True,
    )
    raw_output = ""
    for completion in stream:
        s = completion.completion
        eprint(s)
        raw_output = raw_output + s
    eprintln("")
    return parse_topic_output(raw_output)


def build_structured_output_prompt(topic: str, pdf_content: str) -> str:
    return (
        f"{HUMAN_PROMPT} "
        f"The following is an academic paper in the materials science field related "
        f"to manufacturing and designing {topic} "
        "in order to affect certain properties of the material."
        "The document be provided within <document></document> tags. "
        "Here is the document:"
        f"<document>{pdf_content}</document>"
        "Please get some structured data from that document."
        "In particular, you will be taking the following steps, that result in an "
        "output json object, included within <output></output> tags. "
        f"1. Based on the document, "
        "you will list all of the manufacturing and processing steps for {topic}. "
        "  Please include each individual step within a json array of strings in the "
        '"processes" field '
        "of the output."
        "2. Based on the document, list at least 5 microstructural features which "
        f"determine the performance of {topic}"
        "  Please include each individual feature within a json array of strings in "
        'the "features" field '
        "of the output object."
        f"3. List at least five properties essential for a high performance {topic}. "
        "  Please include each individual property within a json array of strings in "
        'the "properties" field "of the output object.'
        "Please be sure to include the overall JSON output within <output></output> "
        "tags. Example output is included in the following <example></example> tags. "
        "<example>"
        "<output>"
        "{"
        '  "processes": ["Process 0", "Process 1", "Process 2"]'
        '  "features": ["Feature 0", "Feature 1", "Feature 2"]'
        '  "properties": ["Property 0", "Property 1", "Property 2"]'
        "}"
        "</output>"
        "</example>"
        "Please ONLY output the JSON object within the <output></output> tags and "
        "nothing else."
        f"<document>{pdf_content}</document>"
        f"{AI_PROMPT} <output>"
    )


def process_document(topic: str | None, document_path: str) -> StructuredOutput:
    pdf_content = get_pdf_content(document_path)
    if topic is None:
        topic = get_document_topic(document_path)
    my_prompt = build_structured_output_prompt(topic, pdf_content)
    stream = anthropic.completions.create(
        model="claude-2",
        max_tokens_to_sample=1000,
        prompt=my_prompt,
        stream=True,
    )
    raw_output = ""
    for completion in stream:
        s = completion.completion
        eprint(s)
        raw_output = raw_output + s
    eprintln("")
    raw_json_str = parse_structured_output(raw_output)
    json_output = json.loads(raw_json_str)
    json_output["file_name"] = document_path
    json_output["topic"] = topic
    typed_output = StructuredOutput(**json_output)
    return typed_output


def run_on_test_corpus() -> None:
    CORPUS_DIR = "./test_corpus"
    files_to_process = os.listdir(CORPUS_DIR)
    results: list[StructuredOutput] = []

    for file in files_to_process:
        if file == ".DS_Store":
            continue
        eprintln(f"Processing document: {file}")
        output = process_document(None, f"{CORPUS_DIR}/{file}")
        results.append(output)

    json_results = []
    for r in results:
        json_results.append(pydantic_core.to_jsonable_python(r))
    eprintln(json.dumps(json_results, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument(
        "--document-path",
        required=False,
        type=str,
        nargs="?",
        help="a target file",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.document_path:
        print(args.document_path)
        topic = get_document_topic(args.document_path)
        eprintln(topic)
        process_document(topic, args.document_path)
    else:
        run_on_test_corpus()

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from langchain.chat_models.anthropic import ChatAnthropic
from material_application_nodes import claude_chat
import pydantic_core
from PyPDF2 import PdfReader
import os
import sys
import json
from pydantic.dataclasses import dataclass
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

anthropic = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
chat = ChatAnthropic(
    client=anthropic,
    model="claude-2",
    max_tokens_to_sample=300,
    temperature=0,
    streaming=True,
    verbose=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
)


@dataclass
class StructuredOutput:
    file_name: str
    processes: list[str]
    features: list[str]
    properties: list[str]


def eprintln(arg):
    "Print to stderr"
    print(arg, file=sys.stderr)


def eprint(arg):
    "Print to stderr, suitable for streaing content"
    print(arg, flush=True, end="", file=sys.stderr)


def parse_output(claude_output: str) -> str:
    """Expects a string that is a jsonobject, and then </output>"""
    end_index = claude_output.index("</output>")
    if end_index < 0:
        raise IndexError("Did not find </output> in the output")
    return claude_output[0:end_index]


def get_pdf_content(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    pdf_content = ""
    for page in reader.pages:
        text = page.extract_text()
        pdf_content = pdf_content + "\n" + text
    return pdf_content


def build_prompt(topic: str, pdf_content: str) -> str:
    return (
        f"The following is an academic paper in the materials science field related "
        "to manufacturing and designing {topic} "
        "in order to affect certain properties of the material."
        "The document be provided within <document></document> tags. "
        "Here is the document:"
        f"<document>{pdf_content}</document>"
        "Please get some structured data from that document."
        "In particular, you will be taking the following steps, that result in an "
        "output json object, included within <output></output> tags. "
        f"1. You will list all of the manufacturing and processing steps for {topic}. "
        "  Please include each individual step within a json array of strings in the "
        '"processes" field '
        "of the output."
        "2. List at least 5 microstructural features which determine the performance of"
        f"of {topic}"
        "  Please include each individual feature within a json array of strings in "
        'the "features" field '
        "of the output object."
        "3. List at least five properties essential for a high performance steel. "
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
    )


def process_document(document_path: str) -> StructuredOutput:
    pdf_content = get_pdf_content(document_path)
    my_prompt = build_prompt("steel", pdf_content)
    result = claude_chat(content=my_prompt, ai_message="<output>")
    raw_json_str = parse_output(result)
    json_output = json.loads(raw_json_str)
    json_output["file_name"] = document_path
    typed_output = StructuredOutput(**json_output)
    return typed_output


CORPUS_DIR = "./test_corpus"
files_to_process = os.listdir(CORPUS_DIR)
for file in files_to_process[:2]:
    results: list[StructuredOutput] = []
    eprintln(f"Processing document: {file}")
    output = process_document(f"{CORPUS_DIR}/{file}")
    eprintln("Result is: ")
    results.append(output)
    json_results = []
    for r in results:
        json_results.append(pydantic_core.to_jsonable_python(r))
    eprintln(json.dumps(json_results, indent=2))

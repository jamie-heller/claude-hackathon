from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from PyPDF2 import PdfReader
import os
import sys
import json
from pydantic.dataclasses import dataclass


example_pdf = "/Users/jamieheller/code/workspace/CitrineInformatics/claude-hackathon/corpus/#2-Corrosion Science-1999--Electrochemical investigation of the influence of nitrogen alloying on pitting corrosion of austenitic stainless steels.pdf"
reader = PdfReader(example_pdf)
number_of_pages = len(reader.pages)
page = reader.pages[0]
pdf_content = ""
for page in reader.pages:
    text = page.extract_text()
    pdf_content = pdf_content + "\n" + text


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


my_api_key = os.environ.get("ANTHROPIC_API_KEY")
anthropic = Anthropic(api_key=my_api_key)
my_prompt = (
    f"{HUMAN_PROMPT} "
    "Please help me summarize, in a structured way,  "
    "the following academic paper related to manufacturing steel "
    "and its resultant properties."
    "The document be provided within <document></document> tags. "
    "In particular, you will be taking the following steps, that result in an output "
    "json object, included within <output></output> tags. "
    "1. You will list all of the manufacturing and processing steps for steel. "
    '  Please include each individual step within a json array of strings in the the "processes" field '
    "of the output."
    "2. List at least 5 microstructural features which determine the performance of steel"
    '  Please include each individual feature within a json array of strings in the "features" field '
    "of the output object."
    "3. List at least five properties essential for a high performance steel. "
    '  Please include each individual property within a json array of strings in the "properties" field '
    "of the output object."
    "Please be sure to include the overall JSON output within <output></output> tags. "
    "Example output is included in the following <example></example> tags. "
    "<example>"
    "<output>"
    "{"
    '  "processes": ["Process 0", "Process 1", "Process 2"]'
    '  "features": ["Feature 0", "Feature 1", "Feature 2"]'
    '  "properties": ["Property 0", "Property 1", "Property 2"]'
    "}"
    "</output>"
    "</example>"
    "Please ONLY output the JSON object within the <output></output> tags and nothing else"
    f"<document>{pdf_content}</document>"
    f"{AI_PROMPT} <output>"
)
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

# eprintln(raw_output)
raw_json_str = parse_output(raw_output)
print(raw_json_str)
output = json.loads(raw_json_str)

# eprintln(output)


@dataclass
class Output:
    processes: list[str]
    features: list[str]
    properties: list[str]


typed_output = Output(**output)


# eprintln(typed_output)

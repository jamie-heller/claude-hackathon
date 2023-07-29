from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from PyPDF2 import PdfReader
import os

example_pdf = "/Users/jamieheller/code/workspace/CitrineInformatics/claude-hackathon/corpus/#2-Corrosion Science-1999--Electrochemical investigation of the influence of nitrogen alloying on pitting corrosion of austenitic stainless steels.pdf"
reader = PdfReader(example_pdf)
number_of_pages = len(reader.pages)
page = reader.pages[0]
pdf_content = ""
for page in reader.pages:
    text = page.extract_text()
    pdf_content = pdf_content + "\n" + text

my_api_key = os.environ.get("ANTHROPIC_API_KEY")
anthropic = Anthropic(api_key=my_api_key)
my_prompt = (
    f"{HUMAN_PROMPT} "
    "Please help me summarize, in a structured way,  "
    "the following academic paper related to manufacturing steel "
    "and its resultant properties."
    "The document be provided within <document></document> tags. "
    "In particular, you will be taking the following steps"
    "1. You will list all of the manufacturing and processing steps for steel. "
    "  Please include your summary within <processes></processes> tags. "
    "  Please include each individual process within <process></process> tags. "
    "2. List at least 5 microstructural features which determine the performance of steel"
    "  Please include your summary within <features></features> tags. "
    "  Please include each individual process within <feature></feature> tags. "
    "3. List at least five properties essential for a high performance steel. "
    "Do not list just one property."
    "  Please include your summary within <properties></properties> tags. "
    "  Please include each individual process within <property></property> tags. "
    "Please be sure to include the overall output within <output></output> tags. "
    "Example output is included in the following <example></example> tags. "
    "<example><output>"
    "<processes>"
    "<process>First Process</process>"
    "<process>Second Process</process>"
    "<process>Third Process</process>"
    "</processes>"
    "<features>"
    "<feature>First Feature</feature>"
    "<feature>Second Feature</feature>"
    "<feature>Third Feature</feature>"
    "</features>"
    "<properties>"
    "<property>First Feature</property>"
    "</properties>"
    "</output></example>"
    f"<document>{pdf_content}</document>"
    f"{AI_PROMPT} <summary>"
)
stream = anthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=1000,
    prompt=my_prompt,
    stream=True,
)
for completion in stream:
    print(completion.completion, end="", flush=True)

print("")

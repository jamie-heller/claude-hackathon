from util import claude_chat
import pydantic_core
from PyPDF2 import PdfReader
import os
import sys
import json
from pydantic.dataclasses import dataclass
import argparse
import networkx as nx
import matplotlib.pyplot as plt


@dataclass
class StructuredOutput:
    file_name: str
    topic: str
    processes: list[str]
    structures: list[str]
    properties: list[str]
    process_structure_influences: dict[str, str]


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
    )


def parse_topic_output(claude_output: str) -> str:
    """Expects a string that ends with </topic>"""
    return parse_from_end_token(claude_output, "</topic>")


def parse_bool_output(claude_output: str) -> str:
    return parse_from_end_token(claude_output, "</output>")


def build_x_affects_y_prompt(topic: str, pdf_content: str, x: str, y: str) -> str:
    return (
        f"The following is an academic paper in the materials science field related "
        f"to manufacturing and designing materials, specifically {topic}."
        "in order to affect certain properties of the material."
        "The document be provided within <document></document> tags. "
        "Here is the document:"
        f"<document>{pdf_content}</document>"
        f'Please help me determine if "{x}" strongly affects or is correlated with '
        f'"{y}" according '
        f'the academic paper. Please only indicate that "{x}" affects or is correlated '
        f'with "{y}" '
        "if you can find evidence supporting that fact. If you cannot"
        "find evidence supporting that fact or do not know, indicate that"
        f'"{x}" does NOT affect "{y}" by default.'
        "Your output should be returned withing <output></output> tags."
        'The output should be simply the word "true" or "false"'
        "Here are 2 samples of good output, each in <example></example> tags"
        "Good Example 1: "
        "<example><output>true</output></example>"
        "Good Example 2: "
        "<example><output>false</output></example>"
        "Here is an example of BAD output."
        "Bad Example 1: "
        "<example><output>Some reasoning. false</output></example>"
        "Bad Example 2: "
        "<example><output>Some other reasoning. true</output></example>"
        "Please make sure to format your response like the good examples, where"
        "the content within the <output></output> tags is just true or false"
    )


def get_x_affects_y(topic: str, pdf_content: str, x: str, y: str) -> bool:
    my_prompt = build_x_affects_y_prompt(topic, pdf_content, x, y)
    result = claude_chat(content=my_prompt, ai_message="<output>")
    bool_str = parse_bool_output(result)
    eprintln("***")
    eprintln(f"Parsed response: {bool_str}")
    eprintln("***")
    if bool_str.strip().lower() == "true":
        return True
    return False


def add_process_structure_edges(
    pdf_content: str, structured_output: StructuredOutput, G: nx.DiGraph
) -> None:
    for p in structured_output.processes:
        for s in structured_output.structures:
            does_affect = get_x_affects_y(structured_output.topic, pdf_content, p, s)
            if does_affect:
                G.add_edge(p, s)
                structured_output.process_structure_influences[p] = s


def add_structure_property_edges(
    pdf_content: str, structured_output: StructuredOutput, G: nx.DiGraph
) -> None:
    for s in structured_output.structures:
        for p in structured_output.properties:
            does_affect = get_x_affects_y(structured_output.topic, pdf_content, s, p)
            if does_affect:
                G.add_edge(s, p)
                structured_output.process_structure_influences[s] = p


def get_document_topic(pdf_content: str) -> str:
    my_prompt = build_get_topic_prompt(pdf_content)
    result = claude_chat(content=my_prompt, ai_message="<topic>")
    return parse_topic_output(result)


def build_structured_output_prompt(topic: str, pdf_content: str) -> str:
    return (
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
        f"you will list all of the manufacturing and processing steps for {topic}. "
        "  Please include each individual step within a json array of strings in the "
        '"processes" field '
        "of the output."
        "2. Based on the document, list at least 5 microstructural features which "
        f"determine the performance of {topic}"
        "  Please include each individual structure within a json array of strings in "
        'the "structures" field '
        "of the output object."
        f"3. List at least five properties essential for a high performance {topic}. "
        "  Please include each individual property within a json array of strings in "
        'the "properties" field "of the output object.'
        "Please be sure to include the overall JSON output within <output></output> "
        "tags. Example output is included in the following <example></example> tags. "
        "<example>"
        "<output>"
        "{"
        '  "processes": ["Process 0", "Process 1", "Process 2", "Process 3"], '
        '  "structures": ["Structure 0", "Structure 1"], '
        '  "properties": ["Property 0", "Property 1", "Property 2", "Property 3", '
        '"Property 4"]'
        "}"
        "Here is another example of good output"
        "<example>"
        "<output>"
        "{"
        '  "processes": ["PA", "PB", "PC", "PD", "PE"],'
        '  "structures": ["S0", "S1", "S2"], '
        '  "properties": ["Prop0", "Prop2"] '
        "}"
        "</output>"
        "</example>"
        "Please ONLY output the JSON object within the <output></output> tags and "
        "nothing else."
        f"<document>{pdf_content}</document>"
    )


def process_document(topic: str | None, document_path: str) -> StructuredOutput:
    pdf_content = get_pdf_content(document_path)
    if topic is None:
        topic = get_document_topic(pdf_content)
    my_prompt = build_structured_output_prompt(topic, pdf_content)
    result = claude_chat(content=my_prompt, ai_message="<output>")
    raw_json_str = parse_structured_output(result)
    json_output = json.loads(raw_json_str)
    json_output["file_name"] = document_path
    json_output["topic"] = topic
    json_output["process_structure_influences"] = {}
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


def build_initial_graph(structured_output: StructuredOutput) -> nx.DiGraph:
    G = nx.DiGraph()
    for i in structured_output.processes:
        G.add_node(i, layer=0)
    for i in structured_output.structures:
        G.add_node(i, layer=1)
    for i in structured_output.properties:
        G.add_node(i, layer=2)
    return G


def draw_graph(filename: str, G: nx.DiGraph) -> None:
    # Create a layout for the nodes in the graph
    pos = nx.multipartite_layout(G, subset_key="layer")

    # Draw nodes
    plt.figure(dpi=200, figsize=(8, 5))

    subset_color = ["lightskyblue", "lightblue", "lightsteelblue"]
    color = [subset_color[data["layer"]] for v, data in G.nodes(data=True)]
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=color, margins=0.2)
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="lightgray")

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=4, font_color="black")

    # Show the plot
    plt.suptitle(
        topic.capitalize() + ": \nProcess    -    Structure    -    Property",
        fontsize=16,
    )
    plt.axis("off")
    print(f"Saving {filename}")
    plt.savefig(filename)


if __name__ == "__main__":
    args = parse_args()
    if args.document_path:
        print(args.document_path)
        pdf_content = get_pdf_content(args.document_path)
        topic = get_document_topic(pdf_content)
        eprintln(topic)
        output = process_document(topic, args.document_path)
        G = build_initial_graph(output)
        add_process_structure_edges(pdf_content, output, G)
        add_structure_property_edges(pdf_content, output, G)
        eprintln(json.dumps(pydantic_core.to_jsonable_python(output), indent=2))
        filename = f"{output.topic}.png"
        draw_graph(filename, G)
    else:
        run_on_test_corpus()

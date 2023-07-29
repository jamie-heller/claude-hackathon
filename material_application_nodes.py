from anthropic import Anthropic
from langchain.chat_models.anthropic import ChatAnthropic
# from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.schema import HumanMessage
from dotenv import load_dotenv

# import matplotlib.pyplot as plt
# import networkx as nx
import os
import ast

load_dotenv(".env")

anthropic = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

chat = ChatAnthropic(client=anthropic, model="claude-2", max_tokens_to_sample=300, temperature=0)

def claude_chat(content: str) -> str:
    """Run the Claude chat feature with given content

    Parameters
    ----------
    content : str
        The text you want to send to Claude

    Returns
    -------
    str
        Claude-2's response to your content
    """    
    messages = [HumanMessage(content=content)]
    result = chat(messages=messages).content
    return result



def condense_text(items: list) -> list:
    """condense test from each item into a couple words

    Parameters
    ----------
    items : list
        list of text

    Returns
    -------
    list
        list of condensed text
    """    
    for i in range(len(items)):
        if len(items[i]) > 30:
            # set the temeprature to 0 for reproducibility
            content = items[i] + "\n Summarize the previous text in 3 words. Do not use more than 3 words. Do not return an empty response."
            result = claude_chat(content=content)
            if len(result) < 1:
                content = items[i] + "\n Summarize the previous text in 2 words. Do not use more than 2 words. Do not return an empty response."
                result = claude_chat(content=content)
            items[i] = result.strip()
    return items


def get_nodes(topic: str) -> dict:
    """Return a python dictionary with processes, structure, and properties based on a particular
    material application.

    Parameters
    ----------
    topic : str
        material application you want process-structure-properties from

    Returns
    -------
    dict
        mapping with processes, structures, and properties
    """    
    results = {}

    # processing nodes returned as a python list
    content = "List all the manufacturing and processing steps for" + topic + " in the order in which they are performed."
    messages = [HumanMessage(content=content)]
    
    result = chat(messages=messages).content
    results["processing_original"] = result

    content = result.split('\n', 1)[-1] + " \nPrint this list as a python list of comma separated values, e.g.: `[foo, bar, baz]`."
    messages = [HumanMessage(content=content)]
    result = chat(messages=messages).content
    processing = ast.literal_eval(result)
    processing = condense_text(processing)
    results["processing"] = processing

    # structure nodes returned as a python list
    content = "List at least five microstructural features which determine the performance of "+topic+"."
    result = claude_chat(content=content)
    results["structures_original"] = result
    content = result.split('\n', 1)[-1] + " \nPrint this list as a python list of comma separated values, e.g.: `[foo, bar, baz]`."
    result = claude_chat(content=content)
    structures = ast.literal_eval(result)
    structures = condense_text(structures)
    results["structures"] = structures

    # property nodes returned as a python list
    content = "List at least five properties essential for a high performance "+topic+". Do not list just one property."""
    result = claude_chat(content=content)
    results["properties_original"] = result
    content = result.split('\n', 1)[-1] + " \nPrint this list as a python list of comma separated values, e.g.: `[foo, bar, baz]`."
    result = claude_chat(content=content)
    properties = ast.literal_eval(result)
    properties = condense_text(properties)
    results["properties"] = properties
    
    return results

# results = get_nodes(topic="stainless steel")


# def make_graph(topic: str) -> None:
#     """make a graph of the processes, structure, and properties for a given materials application

#     Parameters
#     ----------
#     topic : str
#         a materials application
#     """    
#     results = get_nodes(topic=topic)

#     G = nx.DiGraph()

#     # detrmine nodes
#     results = get_nodes(topic)

#     # Add nodes
#     for i in results["processing"]:
#         G.add_node(i, layer=0)
#     for i in results["structures"]:
#         G.add_node(i, layer=1)
#     for i in results["properties"]:
#         G.add_node(i, layer=2)

#     for i in results["processing"]:
#         for j in results["structures"]:
#             content = "Does " + i + " strongly affect " + j + " in " + topic + "?"
#             result = claude_chat(content=content)
#             if "yes" in result.lower():
#                 G.add_edge(i,j)

#     for i in results["structures"]:
#         for j in results["properties"]:
#             content = "Does " + i + " strongly affect " + j + " in " + topic + "?"
#             result = claude_chat(content=content)
#             if "yes" in result.lower():
#                 G.add_edge(i,j)

#     # Create a layout for the nodes in the graph
#     pos = nx.multipartite_layout(G, subset_key="layer")

#     # Draw nodes
#     plt.figure(dpi=200, figsize=(8,5))

#     subset_color = ["lightskyblue", "lightblue", "lightsteelblue"]
#     color = [subset_color[data["layer"]] for v, data in G.nodes(data=True)]
#     nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=color, margins=0.2)

#     # Draw edges
#     nx.draw_networkx_edges(G, pos, edge_color="lightgray")

#     # Draw labels
#     nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

#     # Show the plot
#     plt.suptitle(topic.capitalize()+': \nProcess    -    Structure    -    Property', fontsize=16)
#     plt.axis("off")
#     plt.show()

# make_graph("stainless steel")
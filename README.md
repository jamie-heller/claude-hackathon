# Iron Claude

https://github.com/jamie-heller/claude-hackathon/assets/121886584/2693e1b8-3395-4782-b2cb-97f5e33a7bcc

### (Part of the claude 2 hackathon)

## Elevator Pitch
We leverage Claude 2 to extract data from Materials Science research papers in a format that is suitable for shaping graph-based ML models on platforms like Citrine Informatics.

## Inspiration
At Citrine Informatics, we build machine learning models for materials scientists and chemists. This enables them to make predictions on the next experiments to make, saving them resources, energy, time, and money. But it all starts with requiring structured data. The materials industry is far behind other industries (e.g. Biotech, Fintech, etc) when it comes to having good data management and structured data.

## What it does
The Iron Calude ingests unstructured materials/chemicals-specific data (e.g. research papers), determines the physical application, and extracts the relevant process-structure-properties as nodes. Claude builds a graph linking each of these nodes together as they impact one another. This graph can be used by Citrine Platform to inform it how to build the right machine learning model architecture for running experimental predictions.

## How we built it
This project was built with python using the anthropic, langchain, and networkx libraries. We have utility functions that parse unstructured data, form the anthropic client, format the prompt, and run Claude-2's model. Our main script can be ran from the command line with an optional `document_path` argument to an unstructured file. Or it can be ran on a corpus of documents.



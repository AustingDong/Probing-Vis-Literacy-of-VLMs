from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate

def format_input(question, options):
    response_schemas = [
        ResponseSchema(name="answer", description="The option you selected (e.g. A), referring to the answer."),
        ResponseSchema(name="reason", description="A brief reason about why you selected that answer.")
    ]

    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()

    option_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    
    prompt_template = PromptTemplate.from_template(
        """You are an expert chart reasoning assistant. Based on the following chart question, select the correct option and explain why.

Question:
{question}

Options:
{option_text}

{format_instructions}
"""
    )

    prompt = prompt_template.format(
        question=question,
        option_text=option_text,
        format_instructions=format_instructions
    )

    return prompt, parser



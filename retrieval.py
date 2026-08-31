from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from transformers import BitsAndBytesConfig
import torch
import os
from dotenv import load_dotenv

from config import (
    DB_NAME,
    COLLECTION_NAME,
    CANDIDATE_K,
    FINAL_K,
    RERANKER_BATCH_SIZE,
    MODEL,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    EMBEDDING_DEVICE,
    RERANKER_DEVICE,
)

from tools import calculator, tools

load_dotenv()

#=========================================================================================
#db_name and collection_name to be assigned through config.py
chroma = chromadb.PersistentClient(path=DB_NAME)
vectorstore = chroma.get_collection(name=COLLECTION_NAME)

#==========================================================================================
#Embedding Quantization and Loading model
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL,
    model_kwargs={"device_map": EMBEDDING_DEVICE,
                  "quantization_config":quantization_config,
                  "torch_dtype":torch.float16},
    tokenizer_kwargs={"padding_side": "left"},
)
#===========================================================================================
#Loading Reranker Model
reranker_model = CrossEncoder(
    RERANKER_MODEL,
    prompts={"classification": "Determine whether the annual-report passage contains factual, numerical, comparative, or qualitative evidence needed to answer the question."},
    default_prompt_name="classification",
    device=RERANKER_DEVICE,
    model_kwargs={"torch_dtype": torch.float16}
)

def reranker(question, documents):
  contexts = documents["documents"][0]
  pairs = [(question, context) for context in contexts]
  scores = reranker_model.predict(
      pairs,
      batch_size=RERANKER_BATCH_SIZE,
  )
  metadatas = documents["metadatas"][0]
  triplets = zip(scores, contexts, metadatas)
  scored_contexts_sorted = sorted(triplets, key=lambda x: x[0], reverse=True)
  reranked_contexts = [context for score, context, metadata in scored_contexts_sorted][:FINAL_K]
  reranked_metadatas = [metadata for score, context, metadata in scored_contexts_sorted][:FINAL_K]
  return reranked_contexts, reranked_metadatas

#================================================================================================
COMPARISON_WORDS = [
    "compare",
    "compared",
    "versus",
    "vs",
    "higher",
    "lower",
    "highest",
    "lowest",
    "better",
    "worse",
    "which company",
    "difference",
    "among"
]

def filter_by_company(question, contexts, metadatas):

    question_lower = question.lower()

    companies = [
        company
        for company in ["infosys", "tcs", "itc"]
        if company in question_lower
    ]

    is_comparison = any(
        word in question_lower
        for word in COMPARISON_WORDS
    )

    # Filter only when exactly one company is mentioned
    # and the question is not comparative
    if len(companies) != 1 or is_comparison:
        return contexts, metadatas

    company = companies[0]

    filtered = [
        (context, metadata)
        for context, metadata in zip(
            contexts,
            metadatas
        )
        if company in metadata.get(
            "source",
            ""
        ).lower()
    ]

    # Safety fallback
    if not filtered:
        return contexts, metadatas

    filtered_contexts = [
        context
        for context, metadata in filtered
    ]

    filtered_metadatas = [
        metadata
        for context, metadata in filtered
    ]

    return (
        filtered_contexts,
        filtered_metadatas
    )
#===============================================================================================
def get_company(metadata):

    source = metadata.get(
        "source",
        ""
    ).lower()

    if "infosys" in source:
        return "Infosys"

    if "tcs" in source:
        return "TCS"

    if "itc" in source:
        return "ITC"

    return "Unknown"
#================================================================================================
SYSTEM_PROMPT = """You are a knowledgeable, friendly assistant answering questions about the annual reports of Infosys, ITC and TCS.

Each context passage includes a Source field identifying the annual report it came from.

IMPORTANT SOURCE RULES:
- Treat the Source field as indicating which company the facts in that passage belong to.
- Never attribute a figure from one company's source to another company.
- If the question asks about only one company, prioritize passages from that company's report and ignore figures from other companies unless they are explicitly needed.
- If the question compares companies, keep each company's figures separate and verify the source before comparing them.
- Do not infer that two similar figures belong to the same company.

When arithmetic is required, use the calculator tool rather than performing the calculation yourself.
Use only figures supported by the provided context.

If the answer is not supported by the context, say you don't know.

### CONTEXT

{context}
"""

llm = ChatOpenAI(model=MODEL)


llm_with_tools = llm.bind_tools([calculator])

def question_answer(question):
    query_embeddings = embedding_model.encode(question, prompt_name="query")
    docs = vectorstore.query(query_embeddings=query_embeddings, n_results=CANDIDATE_K)
    
    reranked_contexts, reranked_metadatas = reranker(
        question,
        docs
    )
    
    reranked_contexts, reranked_metadatas = filter_by_company(
        question,
        reranked_contexts,
        reranked_metadatas
    )
    
    reranked_contexts_string = "\n\n".join(
        f"""### Passage {i}
    
    Company: {get_company(metadata)}
    Source: {metadata.get("source", "Unknown")}
    
    {context}"""
        for i, (context, metadata) in enumerate(
            zip(
                reranked_contexts,
                reranked_metadatas
            ),
            start=1
        )
    )
    
    system_message = SYSTEM_PROMPT.format(context=reranked_contexts_string)
    messages = [SystemMessage(content=system_message),HumanMessage(content=question)]
    response = llm_with_tools.invoke(messages)
    
    MAX_TOOL_ROUNDS = 3
    tool_round = 0
    
    while response.tool_calls:
    
        tool_round += 1
    
        if tool_round > MAX_TOOL_ROUNDS:
            break
        messages.append(response)

        for tool_call in response.tool_calls:
            selected_tool = tools[tool_call["name"]]
            arguments = tool_call["args"]
            
            try:
                result = selected_tool.invoke(arguments)
                tool_content = str(result)
        
            except Exception as e:
                tool_content = (
                    f"Calculator error: {e}. "
                    "Retry using only plain arithmetic with numbers, "
                    "parentheses, and +, -, *, /, **, %. "
                    "Do not use functions such as round(), min(), max(), "
                    "or comparison operators."
                )
            
            tool_message = ToolMessage(
                content=tool_content,
                tool_call_id=tool_call["id"]
            )
        
            messages.append(tool_message)

        response = llm_with_tools.invoke(messages)

    answer = response.content
    return answer, reranked_contexts, reranked_metadatas

# question_answer("What share of Infosys’ top 200 clients were involved in AI journeys, and how many AI projects were underway?")

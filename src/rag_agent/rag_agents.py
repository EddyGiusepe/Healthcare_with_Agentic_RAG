import os
import sys
from qdrant_client import QdrantClient
from fastembed.text import TextEmbedding
from pydantic import BaseModel, Field
from typing import Type, Any
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from dotenv import load_dotenv, find_dotenv
# Adding the root directory to the PYTHONPATH:
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from utils.constants import *

load_dotenv(find_dotenv())

# Initialize qdrant client
qdrant_client = QdrantClient(url=os.environ.get('QDRANT_URL'), api_key=os.environ.get('QDRANT_API_KEY'))

# initialize the text embedding
embedding_model = TextEmbedding(model_name='snowflake/snowflake-arctic-embed-m')


class SearchInput(BaseModel):
    """Input schema for search tool."""
    query: str = Field(..., description="The search query")


class SearchMedicalHistoryTool(BaseTool):
    name: str = "search_medical_records"
    description: str = "Search through medical records using vector similarity"
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> Any:
        # Use OpenAI embeddings to match data_loader.py
        query_vector = next(embedding_model.query_embed(query=query))

        search_results = qdrant_client.search(
            collection_name='medical_records',
            query_vector=query_vector,
            limit=10,
            score_threshold=0.7
        )

        return [
            {
                "score": hit.score,
                "text": hit.payload.get('text', 'N/A'),
            }
            for hit in search_results
        ]


def trigger_crew(query: str) -> str:
    # initialize the tools
    search_tool = SearchMedicalHistoryTool()

    # Create agents:
    researcher = Agent(
        role='Assistente de Pesquisa',
        goal='Encontrar e analisar informações relevantes',
        backstory="""Você é um especialista em encontrar e analisar informações.
                  Você sabe quando procurar registros de histórico médico, e quando 
                  realizar uma análise detalhada.""",
        tools=[search_tool],
        verbose=False
    )

    synthesizer = Agent(
        role='Sintetizador de Informações',
        goal='Criar respostas completas e claras',
        backstory="""Você é um especialista em tomar informações brutas e analisar
                  e criar respostas claras e presente-as como insights acessíveis.""",
        verbose=False
    )

    # Create tasks with expected_output:
    research_task = Task(
        description=f"""
                    1. Processar esta query: '{query}'
                    2. Se precisa de informações de histórico médico, use a ferramenta de busca.
                    3. Para análise detalhada, use a ferramenta de busca.
                    4. Explique sua seleção de ferramenta e processo.
                    5. Sempre responda em português (pt-br) brasileiro.
                    6. Se a query não é sobre histórico médico, responda que não temos informações sobre o assunto.
                    """,
        expected_output="""
                       Um dicionário contendo:
                       - As ferramentas usadas
                       - Os resultados brutos de cada ferramenta
                       - Qualquer análise realizada
                       """,
        agent=researcher
    )

    synthesis_task = Task(
        description="""
                    1. Pegue os resultados da pesquisa e crie uma resposta clara.
                    2. Explique o processo usado e por que foi apropriado.
                    3. Certifique-se de que a resposta diretamente responda à query original.
                    4. Sempre responda em português (pt-br) brasileiro.
                    5. Se a query não é sobre histórico médico, responda que não temos informações sobre o assunto.
                    """,
        expected_output="""
                       Uma resposta clara e estruturada que inclui:
                       - Resposta direta à query
                       - Suporte à evidência da pesquisa
                       - Apresente-a na forma de bullets
                       """,
        agent=synthesizer
    )

    # Create and run crew
    crew = Crew(
        agents=[researcher, synthesizer],
        tasks=[research_task, synthesis_task],
        verbose=False
    )

    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    while True:
        query = input(f"\n{CYAN}Enter your query{RESET} (type 'bye' or 'quit' to exit): ").strip()

        if query.lower() in ['bye', 'quit']:
            print("Goodbye!")
            break

        if not query:
            print(f"{GREEN}Please enter a valid query.{RESET}")
            continue

        try:
            result = trigger_crew(query)
            print(f"\nResult: {result}")
        except Exception as e:
            print(f"{RED}Error processing query: {str(e)}{RESET}")
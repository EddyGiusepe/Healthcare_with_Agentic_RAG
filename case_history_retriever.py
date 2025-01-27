#! /usr/bin/env python3
"""
Senior Data Scientist.: Dr. Eddy Giusepe Chirinos Isidro

Execução
========
streamlit run case_history_retriever.py
"""
import streamlit as st
import traceback
from src.rag_agent.rag_agents import trigger_crew
import openlit

openlit.init(otlp_endpoint="http://127.0.0.1:4318")


def main():
    st.set_page_config(
        page_title="Query Interface",
        page_icon="🤖",
        layout="wide"
    )

    st.title("Recuperador de Histórico Médico")
    st.markdown("Digite sua consulta abaixo para começar.")

    # Initialize session state:
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Query input using a form
    with st.form(key='query_form'):
        query = st.text_input(
            "Digite sua consulta:",
            key="query_input",
            placeholder="Digite sua consulta aqui . . ."
        )
        submit_button = st.form_submit_button("Enviar")

    # Clear history button outside the form
    if st.button("Limpar Histórico"):
        st.session_state.chat_history = []

    # Process the query when form is submitted
    if submit_button and query:
        try:
            result = trigger_crew(query)

            # Add to chat history
            st.session_state.chat_history.append({
                "query": query,
                "result": result,
                "error": None
            })

        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            st.session_state.chat_history.append({
                "query": query,
                "result": None,
                "error": error_msg
            })

    # Display chat history in reverse chronological order
    st.markdown("### Histórico de Chat")
    for item in reversed(st.session_state.chat_history):
        with st.expander(f"Query: {item['query']}", expanded=True):
            if item['result']:
                st.success(item['result'])
            if item['error']:
                st.error(item['error'])

    # Add some helpful information at the bottom
    st.markdown("---")
    st.markdown("""
    **Dicas:**
    - Digite sua consulta no campo de texto acima
    - Pressione Enter ou clique em Enviar para processar sua consulta
    - Clique em Limpar Histórico para começar de novo
    - Cada consulta e seu resultado serão salvos no histórico de chat
    """)


if __name__ == "__main__":
    main()
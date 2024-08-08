
@app.route('/answer', methods=['POST'])
def answer():
    history = session.get('history', [])
    CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory='db',
        anonymized_telemetry=False
    )
    db = Chroma(persist_directory='db', embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})

    template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )
    llm = ChatOpenAI(model_name=model_path, temperature=0.1)
    chain_type_kwargs = {
            "verbose": True,
            "prompt": prompt,
            "memory": ConversationBufferMemory(
                memory_key="history",
                input_key="question"
            ),
        }
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        verbose=True,
        chain_type_kwargs=chain_type_kwargs
    )
    query = request.form['query']
    result = qa.run({'query': query, 'history': history}) 
    history.append({'Human: ' + query})
    history.append({'AI': result})
    session['history'] = history 
    print(history)
    return jsonify({'answer': result})
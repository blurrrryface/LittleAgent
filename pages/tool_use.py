import streamlit as st

from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain.chains import LLMMathChain
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_core.runnables import RunnableConfig

from app.tools.models.model_tokens import models_tokens
from app.tools.models.select_llms import get_llms
from streamlit_agent.clear_results import with_clear_container



st.set_page_config(
    page_title="MRKL", page_icon="🦜", layout="wide", initial_sidebar_state="collapsed"
)
all_provider = set(models_tokens.keys())
with st.sidebar:
    st.sidebar.success("选择一个模型.")
    provider = st.sidebar.selectbox(
        "选择一个模型提供商",
        all_provider
    )
    model_name = st.sidebar.selectbox(
        "选择一个模型",
        models_tokens[provider]
    )
    system_prompt = st.sidebar.text_area(
        "系统提示",
        "You are a helpful assistant."
    )


# Tools setup
llm_config = {
            "provider": provider,
            "model": model_name,
            "streaming": True,
        }

llm = get_llms(llm_config)

# search = DuckDuckGoSearchAPIWrapper()
llm_math_chain = LLMMathChain.from_llm(llm)

# Make the DB connection read-only to reduce risk of injection attacks
# See: https://python.langchain.com/docs/security
# creator = lambda: sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
# db = SQLDatabase(create_engine("sqlite:///", creator=creator))
# db_chain = SQLDatabaseChain.from_llm(llm, db)

tools = [
    # Tool(
    #     name="Search",
    #     func=search.run,
    #     description="useful for when you need to answer questions about current events. You should ask targeted questions",
    # ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math",
    ),
    # Tool(
    #     name="FooBar DB",
    #     func=db_chain.run,
    #     description="useful for when you need to answer questions about FooBar. Input should be in the form of a question containing full context",
    # ),
]

# Initialize agent
react_agent = create_react_agent(llm, tools, hub.pull("hwchase17/react"))
mrkl = AgentExecutor(agent=react_agent, tools=tools)

with st.form(key="form"):
    user_input = st.text_input("请提一个问题：")
    submit_clicked = st.form_submit_button("Submit Question")

output_container = st.empty()
if with_clear_container(submit_clicked):
    output_container = output_container.container()
    output_container.chat_message("user").write(user_input)

    answer_container = output_container.chat_message("assistant", avatar="🦜")
    st_callback = StreamlitCallbackHandler(answer_container)
    cfg = RunnableConfig()
    cfg["callbacks"] = [st_callback]

    answer = mrkl.invoke({"input": user_input}, cfg)

    answer_container.write(answer["output"])
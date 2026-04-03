import os
import sqlalchemy
import streamlit as st
import pandas as pd
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

def get_db_url():
    try:
        return st.secrets.get("SUPABASE_DB_URL") or os.environ.get("SUPABASE_DB_URL")
    except Exception:
        return os.environ.get("SUPABASE_DB_URL")

@st.cache_resource
def init_db():
    db_url = get_db_url()
    if not db_url:
        return None, None
    engine = sqlalchemy.create_engine(db_url)
    db = SQLDatabase(engine, include_tables=["hdb_resale"])
    return engine, db

@st.cache_data
def load_and_process_data(_engine):
    data = pd.read_sql("SELECT * FROM hdb_resale", _engine)
    data['psf'] = data['resale_price'] / (data['floor_area_sqm'] * 10.7639)
    data['floor_area_sqf'] = data['floor_area_sqm'] * 10.7639
    data['sold_year_month'] = pd.to_datetime(data['sold_year_month'])
    data['flat_type'] = data['flat_type'].str.replace('MULTI-GENERATION', 'MULTI GENERATION')

    correction_map = {
        '2-ROOM': '2-room',
        'APARTMENT': 'Apartment',
        'Improved-Maisonette': 'Executive Maisonette',
        'IMPROVED-MAISONETTE': 'Executive Maisonette',
        'IMPROVED': 'Improved',
        'MAISONETTE': 'Maisonette',
        'Model A-Maisonette': 'Maisonette',
        'MODEL A-MAISONETTE': 'Maisonette',
        'MODEL A': 'Model A',
        'MULTI GENERATION': 'Multi Generation',
        'Premium Apartment Loft': 'Premium Apartment',
        'PREMIUM APARTMENT': 'Premium Apartment',
        'Premium Maisonette': 'Executive Maisonette',
        'SIMPLIFIED': 'Simplified',
        'STANDARD': 'Standard',
        'TERRACE': 'Terrace',
        'NEW GENERATION': 'New Generation'
    }
    data = data.replace({'flat_model': correction_map})

    data['sold_year'] = data['sold_year_month'].dt.strftime('%Y').astype(int)
    data['sold_remaining_lease'] = 99 - (data['sold_year'] - data['lease_commence_date'])
    data['remaining_lease_in_2024'] = 99 - (2024 - data['lease_commence_date'])
    return data

@st.cache_data
def filter_data(data, search_flat_type, search_block, search_town):
    filtered_data = data.copy()
    if search_flat_type:
        filtered_data = filtered_data[filtered_data['flat_type'].str.contains(search_flat_type, case=False, na=False)]
    if search_block:
        filtered_data = filtered_data[filtered_data['block'].str.contains(search_block, case=False, na=False)]
    if search_town:
        filtered_data = filtered_data[filtered_data['town'].str.contains(search_town, case=False, na=False)]
    return filtered_data

@st.cache_data
def sort_data(data, sort_by_price, sort_by_date):
    if sort_by_price == "Highest":
        data = data.sort_values(by='resale_price', ascending=False)
    else:
        data = data.sort_values(by='resale_price', ascending=True)
    if sort_by_date == "Latest":
        data = data.sort_values(by='sold_year_month', ascending=False)
    else:
        data = data.sort_values(by='sold_year_month', ascending=True)
    return data

def calculate_monthly_installment(loan_amount, interest_rate, tenure_years):
    monthly_rate = interest_rate / 100 / 12
    num_payments = tenure_years * 12
    monthly_installment = (loan_amount * monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
    return monthly_installment

def get_sql_chain(db):
    template = """
    You are a data analyst at a property company. You are interacting with a user who is asking you questions about the company's property transactions database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.

    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}

    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.

    Question: {question}
    SQL Query:
    """

    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    classify_prompt = ChatPromptTemplate.from_template("""
    Determine if the following query is a general question about the assistant's capabilities
    or a specific question about property data. Respond with only 'GENERAL' or 'SPECIFIC'.

    Query: {query}

    Classification:""")

    classification = llm.invoke(classify_prompt.format_messages(query=user_query)).content.strip()

    if classification == "GENERAL":
        general_response_prompt = ChatPromptTemplate.from_template("""
        You are a property data assistant. Please respond to the following general question
        about your capabilities or the service you provide. Be helpful and informative, but
        also mention that you're specialized in answering questions about property data.

        Question: {question}

        Response:""")
        return llm.invoke(general_response_prompt.format_messages(question=user_query)).content
    else:
        try:
            sql_chain = get_sql_chain(db)

            template = """
            You are a data analyst at a property company. You are interacting with a user who is asking you questions about the company's property transactions database.
            Based on the table schema below, question, sql query, and sql response, write a natural language response.
            <SCHEMA>{schema}</SCHEMA>

            Conversation History: {chat_history}
            SQL Query: <SQL>{query}</SQL>
            User question: {question}
            SQL Response: {response}"""

            prompt = ChatPromptTemplate.from_template(template)

            chain = (
                RunnablePassthrough.assign(query=sql_chain).assign(
                    schema=lambda _: db.get_table_info(),
                    response=lambda vars: db.run(vars["query"]),
                )
                | prompt
                | llm
                | StrOutputParser()
            )

            return chain.invoke({
                "question": user_query,
                "chat_history": chat_history,
            })
        except Exception as e:
            return f"I apologize, but I encountered an error while trying to process your query about property data. Could you please rephrase your question or ask about specific aspects of the property transactions? Error details: {str(e)}"

def property_transact(engine):
    data = load_and_process_data(engine)

    st.sidebar.header("Search")
    search_flat_type = st.sidebar.text_input("Search by Flat Type")
    search_block = st.sidebar.text_input("Search by Block")
    search_town = st.sidebar.text_input("Search by Town")

    filtered_data = filter_data(data, search_flat_type, search_block, search_town)

    st.sidebar.header("Sorting Options")
    sort_by_price = st.sidebar.radio("Sort by Resale Price", ["Highest", "Lowest"])
    sort_by_date = st.sidebar.radio("Sort by Transaction Date", ["Latest", "Oldest"])

    sorted_data = sort_data(filtered_data, sort_by_price, sort_by_date)

    items_per_page = 20
    total_items = len(sorted_data)
    total_pages = (total_items // items_per_page) + (1 if total_items % items_per_page > 0 else 0)
    page_number = st.sidebar.number_input('Page Number', min_value=1, max_value=total_pages, step=1)
    start_idx = (page_number - 1) * items_per_page
    end_idx = start_idx + items_per_page
    paginated_data = sorted_data.iloc[start_idx:end_idx]

    st.title("Filtered Data Table")
    st.subheader(f"No. of Transactions: {total_items}")

    for idx, row in paginated_data.iterrows():
        st.markdown(f"""
        <div style="background-color:#444; padding:10px; border-radius:5px; margin:10px 0; position:relative;">
            <h4 style="color:white;">{row['block']} {row['street_name']}</h4>
            <p style="color:white; font-size:14px;">{row['town']}</p>
                {row['flat_type']} <span style="font-style:italic;">({row['flat_model']})</span><br>
                Storey: {row['storey_range']}<br>
                Area: {row['floor_area_sqm']} sqm <span style="font-style:italic;">({row['floor_area_sqf']:.2f} sqf)</span><br>
                Built: {row['lease_commence_date']}<br>
                Remaining Lease: {row['sold_remaining_lease']} years<br>
            </p>
            <div style="position:absolute; bottom:10px; right:10px; text-align:right;">
                <p style="color:white;font-size:24px;"><b>Price: ${row['resale_price']}</b></p>
                <p style="color:white;"><b>PSF: ${row['psf']:.2f}</b></p>
                <p style="color:white;"><b>Transaction Date: {row['sold_year_month'].strftime('%Y-%m')}</b></p>
            </div>
        </div>
        """, unsafe_allow_html=True)

def mortgage_calculator():
    st.title("Mortgage Calculator")

    loan_amount = st.number_input("Loan Amount ($)", min_value=0.0, value=100000.0, step=1000.0)
    interest_rate = st.number_input("Annual Interest Rate (%)", min_value=0.0, max_value=20.0, value=3.0, step=0.1)
    tenure_years = st.number_input("Loan Tenure (Years)", min_value=1, max_value=35, value=30, step=1)

    if st.button("Calculate"):
        monthly_installment = calculate_monthly_installment(loan_amount, interest_rate, tenure_years)
        st.success(f"Your estimated monthly installment is: ${monthly_installment:.2f}")

        total_payment = monthly_installment * tenure_years * 12
        total_interest = total_payment - loan_amount

        st.write(f"Total amount paid over {tenure_years} years: ${total_payment:.2f}")
        st.write(f"Total interest paid: ${total_interest:.2f}")

def chat_with_data(db):
    st.title("Chat with Property Data by SGbros")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello! I'm a property data assistant. Ask me anything about the property transactions."),
        ]

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)

    user_query = st.chat_input("Type a message...")
    if user_query is not None and user_query.strip() != "":
        st.session_state.chat_history.append(HumanMessage(content=user_query))

        with st.chat_message("Human"):
            st.markdown(user_query)

        with st.chat_message("AI"):
            response = get_response(user_query, db, st.session_state.chat_history)
            st.markdown(response)

        st.session_state.chat_history.append(AIMessage(content=response))

def main():
    try:
        if "GROQ_API_KEY" in st.secrets:
            os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    engine, db = init_db()
    if engine is None:
        st.error("SUPABASE_DB_URL is not set. Please configure it in Streamlit secrets.")
        st.stop()

    st.sidebar.title("Navigation")
    tabs = ["Property Dashboard", "Mortgage Calculator", "Chat with Data"]
    choice = st.sidebar.radio("Select a tab:", tabs)

    if choice == "Property Dashboard":
        property_transact(engine)
    elif choice == "Mortgage Calculator":
        mortgage_calculator()
    elif choice == "Chat with Data":
        chat_with_data(db)

if __name__ == "__main__":
    main()

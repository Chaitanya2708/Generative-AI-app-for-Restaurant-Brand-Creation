import streamlit as st 
import LangChain_Helper

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Set page config
st.set_page_config(page_title="Restaurant Name Generator 🍽️", page_icon="🍴", layout="centered")

# Custom CSS to improve layout
st.markdown("""
    <style>
    .big-font {
        font-size: 26px !important;
        font-weight: bold;
        color: #ff4b4b;
    }
    .section-title {
        font-size: 20px;
        margin-top: 25px;
        color: #6c63ff;
    }
    .menu-item {
        font-size: 16px;
        padding: 4px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font"> Restaurant Name & Menu Generator</p>', unsafe_allow_html=True)


# Sidebar input
st.sidebar.header(" Select or Enter Cuisine")
default_cuisine = st.sidebar.selectbox("Pick a Cuisine", ("Indian", "Italian", "Mexican", "Arabic", "American"))
custom_cuisine = st.sidebar.text_input("Or type your own cuisine")

# Determine which cuisine to use
selected_cuisine = custom_cuisine.strip() if custom_cuisine else default_cuisine

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    groq_api_key="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",
)

def generate_restaurant_name_and_items(cuisine):
    # Prompt to get restaurant name
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restaurant for {cuisine} cuisine. Suggest one fancy name for it."
    )

    # Chain to get restaurant name
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")

    # Prompt to get menu items
    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest some signature menu items for a restaurant named {restaurant_name}. Return it as a comma-separated list."
    )

    # Chain to get food items
    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")

    # Sequential chain to combine both steps
    restaurant_chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items'],
        verbose=False
    )

    # Run the chain
    response = restaurant_chain({'cuisine': cuisine})
    return response

# Generate and display
if selected_cuisine:
    with st.spinner("Cooking up a name and menu... 🍳"):
        response = generate_restaurant_name_and_items(selected_cuisine)

    st.markdown(f"<div class='section-title'>Restaurant Name:</div>", unsafe_allow_html=True)
    st.success(response['restaurant_name'].strip())

    st.markdown(f"<div class='section-title'>Signature Menu Items:</div>", unsafe_allow_html=True)
    menu_items = [item.strip() for item in response['menu_items'].strip().split(",")]

    cols = st.columns(2)
    for i, item in enumerate(menu_items):
        with cols[i % 2]:
            st.markdown(f"- {item}", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("*Try different cuisines to explore new ideas!*")


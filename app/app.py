import os
import sys

# إضافة المسار الجذري للمشروع لمسار البحث
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
from dotenv import load_dotenv

# استيرادات مباشرة من باقي المجلدات
from src.vectore_store import VectorStoreBuilder
from src.recommender import AnimeRecommender
from config.config import GROQ_API_KEY, MODEL_NAME
from utils.logger import get_logger
from utils.custom_exception import CustomException

st.set_page_config(page_title="Anime Recommender", layout="wide")
load_dotenv()

logger = get_logger(__name__)

@st.cache_resource
def init_pipeline(persist_dir="chroma_db"):
    try:
        logger.info("Initializing Anime Recommendation Pipeline...")

        # تحميل قاعدة البيانات الشعاعية (vector store)
        vector_builder = VectorStoreBuilder(csv_path="", persist_dir=persist_dir)
        retriever = vector_builder.load_vector_store().as_retriever()

        recommender = AnimeRecommender(retriever, GROQ_API_KEY, MODEL_NAME)

        logger.info("Pipeline initialized successfully.")
        return recommender

    except Exception as e:
        logger.error(f"Pipeline initialization failed: {str(e)}")
        raise CustomException("Error during pipeline initialization", e)

# بدء Streamlit UI
recommender = init_pipeline()

st.title("🎌 Anime Recommender System")

query = st.text_input("🎯 Enter your anime preferences (e.g., action anime with strong character development):")

if query:
    with st.spinner("🔍 Fetching recommendations for you..."):
        try:
            response = recommender.get_recommendation(query)
            st.markdown("### 📺 Recommendations:")
            st.write(response)
        except Exception as e:
            st.error(f"❌ Failed to get recommendation: {e}")

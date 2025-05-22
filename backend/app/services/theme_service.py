import asyncio
import uuid
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI # New
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings
from app.models.query import IndividualDocumentAnswer, Theme # Pydantic models
import logging
import json
import re

logger = logging.getLogger(__name__)

class ThemeIdentificationService:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.llm = ChatGoogleGenerativeAI(
                model=settings.GEMINI_GENERATIVE_MODEL_NAME,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=0.2,
                convert_system_message_to_human=True    
            )
            cls._instance.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english', 
                min_df=1, # Consider at least 1 document
                max_df=0.9, # Ignore terms that are too frequent
                ngram_range=(1,2) # Uni- and bi-grams
            )
            # AgglomerativeClustering: n_clusters=None with distance_threshold allows dynamic cluster count
            cls._instance.cluster_model = AgglomerativeClustering(
                n_clusters=None, 
                distance_threshold=1.2, # Adjust this threshold based on desired granularity (lower = more clusters)
                linkage='ward',
                metric='cosine' # Cosine distance is often good for text
            )
        return cls._instance

    async def _llm_generate_theme_label_and_summary(
        self, 
        user_query: str, 
        texts_in_cluster: List[str], 
    ) -> Tuple[str, str]:
        if not texts_in_cluster:
            return "Uncategorized Theme", "Not enough information to determine a theme."

        # Limit context to avoid excessive token usage/cost
        # Concatenate a sample of texts, focusing on diversity if possible (not done here)
        context_sample_limit = 5
        context_for_llm = "\n\n---\n\n".join(texts_in_cluster[:context_sample_limit])
        
        # Ensure the prompt guides the LLM to produce JSON
        prompt_template_str = (
            "You are an expert in thematic analysis. Based on the following user query and a collection of related text excerpts from documents, "
            "provide a concise and descriptive theme label (3-5 words maximum) and a brief summary (1-2 sentences) that captures the common topic discussed in these excerpts, specifically in relation to the user's query.\n"
            "User Query: \"{user_query}\"\n\n"
            "Relevant Excerpts:\n\"\"\"\n{context_for_llm}\n\"\"\"\n\n"
            "Respond ONLY in JSON format with two keys: 'theme_label' and 'theme_summary'. Do not add any text before or after the JSON object.\n"
            "Example JSON response: {{\"theme_label\": \"Data Security Measures\", \"theme_summary\": \"The documents discuss various measures and protocols for ensuring data security and preventing unauthorized access.\"}}"
        )
        prompt = ChatPromptTemplate.from_template(prompt_template_str)
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            llm_response_str = await chain.ainvoke({
                "user_query": user_query,
                "context_for_llm": context_for_llm
            })
            
            # Robust JSON parsing
            # Sometimes LLMs add ```json ... ``` or other explanations
            json_match = re.search(r'\{.*\}', llm_response_str, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                response_json = json.loads(json_str)
                theme_label = response_json.get("theme_label", "Unnamed Theme (LLM parsing)")
                theme_summary = response_json.get("theme_summary", "Summary unavailable (LLM parsing).")
                return theme_label, theme_summary
            else:
                logger.warning(f"LLM did not return valid JSON for theme generation. Response: {llm_response_str}")
                return "Theme (LLM Format Error)", "Could not parse theme details from LLM."

        except json.JSONDecodeError as jde:
            logger.error(f"Failed to parse JSON from LLM for theme: {llm_response_str}. Error: {jde}", exc_info=True)
            return "Theme (JSON Error)", "Error parsing theme details."
        except Exception as e:
            logger.error(f"Error in LLM theme generation: {e}", exc_info=True)
            return "Theme (Generation Error)", "Could not generate theme summary due to an error."


    async def identify_and_summarize_themes(
        self, 
        user_query: str, 
        document_answers: List[IndividualDocumentAnswer]
    ) -> Tuple[List[Theme], Optional[str]]:
        
        valid_answers_for_theming = [
            ans for ans in document_answers 
            if ans.extracted_answer and 
               not ans.error_message and 
               "no relevant information" not in ans.extracted_answer.lower() and
               "does not provide an answer" not in ans.extracted_answer.lower() and
               "unable to generate response" not in ans.extracted_answer.lower()
        ]

        if len(valid_answers_for_theming) < 2: # Not enough distinct content to cluster meaningfully
            logger.info("Not enough valid document answers to perform robust theme clustering.")
            # Fallback: if there's one valid answer, make it a single theme
            if len(valid_answers_for_theming) == 1:
                single_ans = valid_answers_for_theming[0]
                label, summary = await self._llm_generate_theme_label_and_summary(user_query, [single_ans.extracted_answer])
                single_theme = Theme(
                    theme_label=label,
                    theme_summary=summary,
                    supporting_documents=[{"document_id": single_ans.document_id, "filename": single_ans.filename}]
                )
                overview = f"The primary finding related to '{user_query}' is: {summary}"
                return [single_theme], overview
            return [], "No significant themes identified due to lack of diverse relevant information in documents."

        texts_for_clustering = [ans.extracted_answer for ans in valid_answers_for_theming]
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts_for_clustering)
            if tfidf_matrix.shape[0] <= 1 or tfidf_matrix.shape[1] == 0: # Not enough features or samples
                 logger.warning("TF-IDF matrix too sparse for clustering. Grouping all as one theme.")
                 # Assign all to cluster 0 if clustering cannot proceed
                 cluster_labels_for_valid_answers = np.zeros(len(valid_answers_for_theming), dtype=int)
            else:
                num_samples = tfidf_matrix.shape[0]
                # Ensure distance_threshold is appropriate or enough samples exist.
                # AgglomerativeClustering can sometimes result in one big cluster if threshold is too high.
                if num_samples < 2: # Cannot cluster less than 2 samples
                    cluster_labels_for_valid_answers = np.zeros(num_samples, dtype=int)
                else:
                    self.cluster_model.n_clusters = None # Ensure dynamic clustering based on distance_threshold
                    cluster_labels_for_valid_answers = self.cluster_model.fit_predict(tfidf_matrix.toarray())
        
        except Exception as e:
            logger.error(f"Clustering failed: {e}. Attempting to group all as one theme.", exc_info=True)
            cluster_labels_for_valid_answers = np.zeros(len(valid_answers_for_theming), dtype=int)

        identified_themes_pydantic: List[Theme] = []
        
        # Process each cluster
        unique_cluster_ids = sorted(list(set(cluster_labels_for_valid_answers)))
        
        # Limit number of themes to avoid overwhelming output / cost
        if len(unique_cluster_ids) > settings.MAX_THEMES_TO_IDENTIFY:
            logger.info(f"More than {settings.MAX_THEMES_TO_IDENTIFY} clusters found ({len(unique_cluster_ids)}). Limiting to top themes based on size (heuristic).")
            # A simple heuristic: take largest clusters. More sophisticated would be by intra-cluster similarity or relevance.
            cluster_sizes = {cid: list(cluster_labels_for_valid_answers).count(cid) for cid in unique_cluster_ids}
            sorted_clusters_by_size = sorted(cluster_sizes.items(), key=lambda item: item[1], reverse=True)
            unique_cluster_ids = [cid for cid, size in sorted_clusters_by_size[:settings.MAX_THEMES_TO_IDENTIFY]]


        llm_tasks = []
        cluster_id_to_doc_info_map = {} # To map LLM results back to docs

        for i, cluster_id in enumerate(unique_cluster_ids):
            indices_in_this_cluster = [
                idx for idx, label in enumerate(cluster_labels_for_valid_answers) if label == cluster_id
            ]
            
            texts_in_this_cluster = [valid_answers_for_theming[j].extracted_answer for j in indices_in_this_cluster]
            doc_info_for_this_cluster = [
                {"document_id": valid_answers_for_theming[j].document_id, "filename": valid_answers_for_theming[j].filename}
                for j in indices_in_this_cluster
            ]
            
            if texts_in_this_cluster:
                # Store doc info to be used after LLM call
                # Using a simple unique key for the task if cluster_id might not be unique for some reason (though it should be)
                task_key = f"cluster_{cluster_id}_task_{i}"
                cluster_id_to_doc_info_map[task_key] = doc_info_for_this_cluster
                llm_tasks.append(
                    self._llm_generate_theme_label_and_summary(user_query, texts_in_this_cluster)
                )
        
        # Run LLM calls concurrently
        if llm_tasks:
            llm_results_tuples = await asyncio.gather(*llm_tasks, return_exceptions=True)
        else:
            llm_results_tuples = []

        task_keys_list = list(cluster_id_to_doc_info_map.keys()) # Ensure order matches llm_tasks
        for i, result_tuple in enumerate(llm_results_tuples):
            if isinstance(result_tuple, Exception):
                logger.error(f"LLM task for theme generation failed: {result_tuple}", exc_info=True)
                continue # Skip this theme

            theme_label, theme_summary = result_tuple
            
            # Get the corresponding document info for this theme
            current_task_key = task_keys_list[i]
            supporting_docs_for_theme = cluster_id_to_doc_info_map.get(current_task_key, [])

            identified_themes_pydantic.append(Theme(
                theme_label=theme_label,
                theme_summary=theme_summary,
                supporting_documents=supporting_docs_for_theme
            ))
        
        # Synthesize an overall response using LLM
        synthesized_overview: Optional[str] = None
        if identified_themes_pydantic:
            theme_summaries_for_overview_prompt = "\n".join(
                [f"- Theme '{th.theme_label}': {th.theme_summary}" for th in identified_themes_pydantic]
            )
            overview_prompt_str = (
                "You are a research summarizer. Based on the following themes identified from multiple documents in response to the user's query, "
                "provide a brief (2-3 sentences) synthesized overview. This overview should act as a high-level summary of the collective findings. Do not list the themes again.\n\n"
                "User Query: \"{user_query}\"\n\n"
                "Identified Themes and their Summaries:\n{theme_summaries_for_overview_prompt}\n\n"
                "Synthesized Overview of Collective Findings (2-3 sentences maximum):"
            )
            overview_prompt = ChatPromptTemplate.from_template(overview_prompt_str)
            overview_chain = overview_prompt | self.llm | StrOutputParser()
            try:
                synthesized_overview = await overview_chain.ainvoke({
                    "user_query": user_query,
                    "theme_summaries_for_overview_prompt": theme_summaries_for_overview_prompt
                })
            except Exception as e:
                logger.error(f"Failed to generate synthesized overview: {e}", exc_info=True)
                synthesized_overview = "Could not synthesize an overall view due to an error."
        else:
             synthesized_overview = "No distinct themes were identified to synthesize an overview."


        return identified_themes_pydantic, synthesized_overview

# Singleton accessor
_theme_service_instance = None

def get_theme_service():
    global _theme_service_instance
    if _theme_service_instance is None:
        _theme_service_instance = ThemeIdentificationService()
    return _theme_service_instance
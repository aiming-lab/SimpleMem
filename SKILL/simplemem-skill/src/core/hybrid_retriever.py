"""
Hybrid Retriever - Stage 3: Adaptive Query-Aware Retrieval with Pruning (Section 3.3)

Paper Reference: Section 3.3 - Adaptive Query-Aware Retrieval with Pruning
Implements:
- Hybrid scoring function S(q, m_k) aggregating semantic, lexical, and symbolic signals
- Query Complexity estimation C_q for adaptive retrieval depth
- Dynamic retrieval depth k_dyn = k_base · (1 + δ · C_q)
- Complexity-Aware Pruning to minimize token usage while maximizing accuracy
"""
from typing import List, Optional, Dict, Any
from models.memory_entry import MemoryEntry
from utils.llm_client import LLMClient
from database.vector_store import VectorStore
import config
import re
from datetime import datetime, timedelta
import dateparser
import concurrent.futures


class HybridRetriever:
    """
    Hybrid Retriever - Stage 3: Adaptive Query-Aware Retrieval with Pruning

    Paper Reference: Section 3.3 - Adaptive Query-Aware Retrieval with Pruning

    Core Components:
    1. Query-aware retrieval across three structured layers:
       - Semantic Layer: Dense vector similarity
       - Lexical Layer: Sparse keyword matching (BM25)
       - Symbolic Layer: Metadata filtering
    2. Hybrid Scoring Function S(q, m_k): aggregates multi-layer signals
    3. Complexity-Aware Pruning: dynamic depth based on C_q
    4. Planning-based multi-query decomposition for comprehensive retrieval
    """
    def __init__(
        self,
        llm_client: LLMClient,
        vector_store: VectorStore,
        semantic_top_k: int = None,
        keyword_top_k: int = None,
        structured_top_k: int = None,
        enable_planning: bool = True,
        enable_reflection: bool = True,
        max_reflection_rounds: int = 2,
        enable_parallel_retrieval: bool = True,
        max_retrieval_workers: int = 3
    ):
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.semantic_top_k = semantic_top_k or config.SEMANTIC_TOP_K
        self.keyword_top_k = keyword_top_k or config.KEYWORD_TOP_K
        self.structured_top_k = structured_top_k or config.STRUCTURED_TOP_K
        
        # Use config values as default if not explicitly provided
        self.enable_planning = enable_planning if enable_planning is not None else getattr(config, 'ENABLE_PLANNING', True)
        self.enable_reflection = enable_reflection if enable_reflection is not None else getattr(config, 'ENABLE_REFLECTION', True)
        self.max_reflection_rounds = max_reflection_rounds if max_reflection_rounds is not None else getattr(config, 'MAX_REFLECTION_ROUNDS', 2)
        self.enable_parallel_retrieval = enable_parallel_retrieval if enable_parallel_retrieval is not None else getattr(config, 'ENABLE_PARALLEL_RETRIEVAL', True)
        self.max_retrieval_workers = max_retrieval_workers if max_retrieval_workers is not None else getattr(config, 'MAX_RETRIEVAL_WORKERS', 3)

    def retrieve(self, query: str, enable_reflection: Optional[bool] = None) -> List[MemoryEntry]:
        """
        Execute retrieval with planning and optional reflection

        Args:
        - query: Search query
        - enable_reflection: Override the global reflection setting for this query
                           (useful for adversarial questions that shouldn't use reflection)

        Returns: List of relevant MemoryEntry
        """
        if self.enable_planning:
            return self._retrieve_with_planning(query, enable_reflection)
        else:
            # Fallback to simple semantic search
            return self._semantic_search(query)
    
    def _retrieve_with_planning(self, query: str, enable_reflection: Optional[bool] = None) -> List[MemoryEntry]:
        """
        Execute retrieval with intelligent planning process
        
        Args:
        - query: Search query  
        - enable_reflection: Override reflection setting for this query
        """
        print(f"\n[Planning] Analyzing information requirements for: {query}")
        
        # Step 1: Intelligent analysis of what information is needed
        information_plan = self._analyze_information_requirements(query)
        print(f"[Planning] Identified {len(information_plan['required_info'])} information requirements")
        
        # Step 2: Generate minimal necessary queries based on the plan
        search_queries = self._generate_targeted_queries(query, information_plan)
        print(f"[Planning] Generated {len(search_queries)} targeted queries")
        
        # Step 3: Execute searches for all queries (parallel or sequential)
        if self.enable_parallel_retrieval and len(search_queries) > 1:
            all_results = self._execute_parallel_searches(search_queries)
        else:
            all_results = []
            for i, search_query in enumerate(search_queries, 1):
                print(f"[Search {i}] {search_query}")
                results = self._semantic_search(search_query)
                all_results.extend(results)
        
        # Step 4: Merge and deduplicate results
        merged_results = self._merge_and_deduplicate_entries(all_results)
        print(f"[Planning] Found {len(merged_results)} unique results")
        
        # Step 5: Optional reflection-based additional retrieval
        # Use override parameter if provided, otherwise use global setting
        should_use_reflection = enable_reflection if enable_reflection is not None else self.enable_reflection
        
        if should_use_reflection:
            merged_results = self._retrieve_with_intelligent_reflection(query, merged_results, information_plan)
        
        return merged_results
    
    def _retrieve_with_reflection(self, query: str, initial_results: List[MemoryEntry]) -> List[MemoryEntry]:
        """
        Execute reflection-based additional retrieval
        """
        current_results = initial_results
        
        for round_num in range(self.max_reflection_rounds):
            print(f"\n[Reflection Round {round_num + 1}] Checking if results are sufficient...")
            
            # Quick answer attempt with current results
            if not current_results:
                answer_status = "no_results"
            else:
                answer_status = self._check_answer_adequacy(query, current_results)
            
            if answer_status == "sufficient":
                print(f"[Reflection Round {round_num + 1}] Information is sufficient")
                break
            elif answer_status == "insufficient":
                print(f"[Reflection Round {round_num + 1}] Information is insufficient, generating additional queries...")
                
                # Generate additional targeted queries based on what's missing
                additional_queries = self._generate_additional_queries(query, current_results)
                print(f"[Reflection Round {round_num + 1}] Generated {len(additional_queries)} additional queries")
                
                # Execute additional searches (parallel or sequential)
                if self.enable_parallel_retrieval and len(additional_queries) > 1:
                    print(f"[Reflection Round {round_num + 1}] Executing {len(additional_queries)} additional queries in parallel")
                    additional_results = self._execute_parallel_additional_searches(additional_queries, round_num + 1)
                else:
                    additional_results = []
                    for i, add_query in enumerate(additional_queries, 1):
                        print(f"[Additional Search {i}] {add_query}")
                        results = self._semantic_search(add_query)
                        additional_results.extend(results)
                
                # Merge with existing results
                all_results = current_results + additional_results
                current_results = self._merge_and_deduplicate_entries(all_results)
                print(f"[Reflection Round {round_num + 1}] Total results: {len(current_results)}")
                
            else:  # "no_results"
                print(f"[Reflection Round {round_num + 1}] No results found, cannot continue reflection")
                break
        
        return current_results

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to analyze query intent and extract structured information
        """
        prompt = f"""
Analyze the following query and extract key information:

Query: {query}

Please extract:
1. keywords: List of keywords (names, places, topic words, etc.)
2. persons: Person names mentioned
3. time_expression: Time expression (if any)
4. location: Location (if any)
5. entities: Entities (companies, products, etc.)

Return in JSON format:
```json
{{
  "keywords": ["keyword1", "keyword2", ...],
  "persons": ["name1", "name2", ...],
  "time_expression": "time expression or null",
  "location": "location or null",
  "entities": ["entity1", ...]
}}
```

Return ONLY JSON, no other content.
"""

        messages = [
            {"role": "system", "content": "You are a query analysis assistant. You must output valid JSON format."},
            {"role": "user", "content": prompt}
        ]

        # Retry up to 3 times
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Use JSON format if configured
                response_format = None
                if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                    response_format = {"type": "json_object"}

                response = self.llm_client.chat_completion(
                    messages,
                    temperature=0.1,
                    response_format=response_format
                )
                analysis = self.llm_client.extract_json(response)
                return analysis
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Query analysis attempt {attempt + 1}/{max_retries} failed: {e}. Retrying...")
                else:
                    print(f"Query analysis failed after {max_retries} attempts: {e}")
                    # Return default values
                    return {
                        "keywords": [query],
                        "persons": [],
                        "time_expression": None,
                        "location": None,
                        "entities": []
                    }

    def _semantic_search(self, query: str) -> List[MemoryEntry]:
        """
        Semantic Layer Retrieval

        Paper Reference: Section 3.3 - Part of hybrid scoring function S(q, m_k)
        Retrieves based on dense vector similarity: λ₁ · cos(e_q, v_k)
        """
        return self.vector_store.semantic_search(query, top_k=self.semantic_top_k)

    def _keyword_search(
        self,
        query: str,
        query_analysis: Dict[str, Any]
    ) -> List[MemoryEntry]:
        """
        Lexical Layer Retrieval

        Paper Reference: Section 3.3 - Part of hybrid scoring function S(q, m_k)
        Retrieves based on sparse keyword matching: λ₂ · BM25(q_lex, S_k)
        """
        keywords = query_analysis.get("keywords", [])
        if not keywords:
            # If no keywords extracted, use query itself
            keywords = [query]

        return self.vector_store.keyword_search(keywords, top_k=self.keyword_top_k)

    def _structured_search(self, query_analysis: Dict[str, Any]) -> List[MemoryEntry]:
        """
        Symbolic Layer Retrieval

        Paper Reference: Section 3.3 - Part of hybrid scoring function S(q, m_k)
        Hard filter based on symbolic constraints: γ · 𝕀(R_k ⊨ C_meta)
        """
        persons = query_analysis.get("persons", [])
        location = query_analysis.get("location")
        entities = query_analysis.get("entities", [])
        time_expression = query_analysis.get("time_expression")

        # Parse time range
        timestamp_range = None
        if time_expression:
            timestamp_range = self._parse_time_range(time_expression)

        # Return empty if no structured conditions
        if not any([persons, location, entities, timestamp_range]):
            return []

        # Execute structured search
        return self.vector_store.structured_search(
            persons=persons if persons else None,
            location=location,
            entities=entities if entities else None,
            timestamp_range=timestamp_range,
            top_k=self.structured_top_k
        )

    def _parse_time_range(self, time_expression: str) -> Optional[tuple]:
        """
        Parse time expression to time range

        Examples:
        - "last week" -> (last Monday 00:00, last Sunday 23:59)
        - "November 15" -> (2025-11-15 00:00, 2025-11-15 23:59)
        """
        try:
            # Use dateparser to parse
            parsed_date = dateparser.parse(
                time_expression,
                settings={'PREFER_DATES_FROM': 'past'}
            )

            if parsed_date:
                # Generate time range (for the day)
                start_time = parsed_date.replace(hour=0, minute=0, second=0)
                end_time = parsed_date.replace(hour=23, minute=59, second=59)

                # Expand range for weekly expressions
                if "week" in time_expression.lower() or "周" in time_expression:
                    start_time = start_time - timedelta(days=7)
                    end_time = end_time + timedelta(days=7)

                return (
                    start_time.isoformat(),
                    end_time.isoformat()
                )
        except Exception as e:
            print(f"Time parsing failed: {e}")

        return None

    def _merge_and_deduplicate(
        self,
        results: Dict[str, List[MemoryEntry]]
    ) -> List[MemoryEntry]:
        """
        Merge multi-path retrieval results and deduplicate
        """
        seen_ids = set()
        merged = []

        # Merge by priority (structured > semantic > keyword)
        for source in ['structured', 'semantic', 'keyword']:
            for entry in results.get(source, []):
                if entry.entry_id not in seen_ids:
                    seen_ids.add(entry.entry_id)
                    merged.append(entry)

        return merged
    
    def _generate_search_queries(self, query: str) -> List[str]:
        """
        Generate multiple search queries for comprehensive retrieval
        """
        prompt = f"""
You are helping with information retrieval. Given a user question, generate multiple search queries that would help find comprehensive information to answer the question.

Original Question: {query}

Please generate 3-5 different search queries that cover various aspects and angles of this question. Each query should be focused and specific.

Guidelines:
1. Include the original question as one query
2. Break down complex questions into component parts
3. Consider synonyms and alternative phrasings
4. Think about related concepts that might be relevant
5. Consider temporal, spatial, or contextual variations

Return your response in JSON format:
```json
{{
  "queries": [
    "search query 1",
    "search query 2", 
    "search query 3",
    ...
  ]
}}
```

Return ONLY the JSON, no other text.
"""
        
        messages = [
            {"role": "system", "content": "You are a search query generation assistant. You must output valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Use JSON format if configured
            response_format = None
            if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                response_format = {"type": "json_object"}
                
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.3,
                response_format=response_format
            )
            
            result = self.llm_client.extract_json(response)
            queries = result.get("queries", [query])
            
            # Ensure original query is included
            if query not in queries:
                queries.insert(0, query)
                
            return queries
            
        except Exception as e:
            print(f"Failed to generate search queries: {e}")
            # Fallback to original query
            return [query]
    
    def _merge_and_deduplicate_entries(self, entries: List[MemoryEntry]) -> List[MemoryEntry]:
        """
        Merge and deduplicate memory entries by entry_id
        """
        seen_ids = set()
        merged = []
        
        for entry in entries:
            if entry.entry_id not in seen_ids:
                seen_ids.add(entry.entry_id)
                merged.append(entry)
        
        return merged
    
    def _check_answer_adequacy(self, query: str, contexts: List[MemoryEntry]) -> str:
        """
        Check if current contexts are sufficient to answer the query
        Returns: "sufficient", "insufficient", or "no_results"
        """
        if not contexts:
            return "no_results"
        
        # Format contexts
        context_str = self._format_contexts_for_check(contexts)
        
        prompt = f"""
You are evaluating whether the provided context contains sufficient information to answer a user question.

Question: {query}

Context:
{context_str}

Please evaluate whether the context contains enough information to provide a meaningful, accurate answer to the question.

Consider these criteria:
1. Does the context directly address the question being asked?
2. Are there key details necessary to answer the question?
3. Is the information specific enough to avoid vague responses?

Return your evaluation in JSON format:
```json
{{
  "assessment": "sufficient" OR "insufficient",
  "reasoning": "Brief explanation of why the context is or isn't sufficient",
  "missing_info": ["list", "of", "missing", "information"] (only if insufficient)
}}
```

Return ONLY the JSON, no other text.
"""
        
        messages = [
            {"role": "system", "content": "You are an information adequacy evaluator. You must output valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Use JSON format if configured
            response_format = None
            if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                response_format = {"type": "json_object"}
                
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.1,
                response_format=response_format
            )
            
            result = self.llm_client.extract_json(response)
            return result.get("assessment", "insufficient")
            
        except Exception as e:
            print(f"Failed to check answer adequacy: {e}")
            # Default to insufficient to be safe
            return "insufficient"
    
    def _generate_additional_queries(self, original_query: str, current_contexts: List[MemoryEntry]) -> List[str]:
        """
        Generate additional targeted queries based on what's missing
        """
        context_str = self._format_contexts_for_check(current_contexts)
        
        prompt = f"""
Based on the original question and current available information, generate additional specific search queries that would help find the missing information needed to answer the question completely.

Original Question: {original_query}

Current Available Information:
{context_str}

Analyze what specific information is still missing and generate 2-4 targeted search queries that would help find this missing information.

The queries should be:
1. Specific and focused on the missing information
2. Different from the original question
3. Likely to find complementary information

Return your response in JSON format:
```json
{{
  "missing_analysis": "Brief analysis of what's missing",
  "additional_queries": [
    "specific search query 1",
    "specific search query 2",
    ...
  ]
}}
```

Return ONLY the JSON, no other text.
"""
        
        messages = [
            {"role": "system", "content": "You are a search strategy assistant. You must output valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Use JSON format if configured
            response_format = None
            if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                response_format = {"type": "json_object"}
                
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.3,
                response_format=response_format
            )
            
            result = self.llm_client.extract_json(response)
            return result.get("additional_queries", [])
            
        except Exception as e:
            print(f"Failed to generate additional queries: {e}")
            return []
    
    def _format_contexts_for_check(self, contexts: List[MemoryEntry]) -> str:
        """
        Format contexts for adequacy checking (more concise than full format)
        """
        formatted = []
        for i, entry in enumerate(contexts, 1):
            parts = [f"[Info {i}] {entry.lossless_restatement}"]
            if entry.timestamp:
                parts.append(f"Time: {entry.timestamp}")
            formatted.append(" | ".join(parts))
        
        return "\n".join(formatted)
    
    def _execute_parallel_searches(self, search_queries: List[str]) -> List[MemoryEntry]:
        """
        Execute multiple search queries in parallel using ThreadPoolExecutor
        """
        print(f"[Parallel Search] Executing {len(search_queries)} queries in parallel with {self.max_retrieval_workers} workers")
        all_results = []
        
        try:
            # Use ThreadPoolExecutor for parallel retrieval
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_retrieval_workers) as executor:
                # Submit all search tasks
                future_to_query = {}
                for i, query in enumerate(search_queries, 1):
                    future = executor.submit(self._semantic_search_worker, query, i)
                    future_to_query[future] = (query, i)
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_query):
                    query, query_num = future_to_query[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                        print(f"[Parallel Search] Query {query_num} completed: {len(results)} results")
                    except Exception as e:
                        print(f"[Parallel Search] Query {query_num} failed: {e}")
                        
        except Exception as e:
            print(f"[Parallel Search] Parallel execution failed: {e}. Falling back to sequential search...")
            # Fallback to sequential processing
            for i, query in enumerate(search_queries, 1):
                try:
                    print(f"[Sequential Search {i}] {query}")
                    results = self._semantic_search(query)
                    all_results.extend(results)
                except Exception as search_e:
                    print(f"[Sequential Search {i}] Failed: {search_e}")
        
        return all_results
    
    def _semantic_search_worker(self, query: str, query_num: int) -> List[MemoryEntry]:
        """
        Worker function for parallel semantic search
        """
        print(f"[Search {query_num}] {query}")
        return self._semantic_search(query)
    
    def _execute_parallel_additional_searches(self, additional_queries: List[str], round_num: int) -> List[MemoryEntry]:
        """
        Execute additional reflection queries in parallel
        """
        all_results = []
        
        try:
            # Use ThreadPoolExecutor for parallel retrieval
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_retrieval_workers) as executor:
                # Submit all search tasks
                future_to_query = {}
                for i, query in enumerate(additional_queries, 1):
                    future = executor.submit(self._additional_search_worker, query, i, round_num)
                    future_to_query[future] = (query, i)
                
                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_query):
                    query, query_num = future_to_query[future]
                    try:
                        results = future.result()
                        all_results.extend(results)
                        print(f"[Reflection Round {round_num}] Additional query {query_num} completed: {len(results)} results")
                    except Exception as e:
                        print(f"[Reflection Round {round_num}] Additional query {query_num} failed: {e}")
                        
        except Exception as e:
            print(f"[Reflection Round {round_num}] Parallel execution failed: {e}. Falling back to sequential search...")
            # Fallback to sequential processing
            for i, query in enumerate(additional_queries, 1):
                try:
                    print(f"[Additional Search {i}] {query}")
                    results = self._semantic_search(query)
                    all_results.extend(results)
                except Exception as search_e:
                    print(f"[Additional Search {i}] Failed: {search_e}")
        
        return all_results
    
    def _additional_search_worker(self, query: str, query_num: int, round_num: int) -> List[MemoryEntry]:
        """
        Worker function for parallel additional search in reflection
        """
        print(f"[Additional Search {query_num}] {query}")
        return self._semantic_search(query)
    
    def _analyze_information_requirements(self, query: str) -> Dict[str, Any]:
        """
        Query Complexity Estimation C_q

        Paper Reference: Section 3.3 - Eq. (8)
        Analyzes query complexity to determine minimal information requirements
        and optimal retrieval depth k_dyn
        """
        prompt = f"""
Analyze the following question and determine what specific information is required to answer it comprehensively.

Question: {query}

Think step by step:
1. What type of question is this? (factual, temporal, relational, explanatory, etc.)
2. What key entities, events, or concepts need to be identified?
3. What relationships or connections need to be established?
4. What minimal set of information pieces would be sufficient to answer this question?

Return your analysis in JSON format:
```json
{{
  "question_type": "type of question",
  "key_entities": ["entity1", "entity2", ...],
  "required_info": [
    {{
      "info_type": "what kind of information",
      "description": "specific information needed",
      "priority": "high/medium/low"
    }}
  ],
  "relationships": ["relationship1", "relationship2", ...],
  "minimal_queries_needed": 2
}}
```

Focus on identifying the minimal essential information needed, not exhaustive details.

Return ONLY the JSON, no other text.
"""
        
        messages = [
            {"role": "system", "content": "You are an intelligent information requirement analyst. You must output valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Use JSON format if configured
            response_format = None
            if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                response_format = {"type": "json_object"}
                
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.2,
                response_format=response_format
            )
            
            result = self.llm_client.extract_json(response)
            return result
            
        except Exception as e:
            print(f"Failed to analyze information requirements: {e}")
            # Fallback to simple analysis
            return {
                "question_type": "general",
                "key_entities": [query],
                "required_info": [{"info_type": "general", "description": "relevant information", "priority": "high"}],
                "relationships": [],
                "minimal_queries_needed": 1
            }
    
    def _generate_targeted_queries(self, original_query: str, information_plan: Dict[str, Any]) -> List[str]:
        """
        Generate minimal targeted queries based on information requirements analysis
        """
        prompt = f"""
Based on the information requirements analysis, generate the minimal set of targeted search queries needed to gather the required information.

Original Question: {original_query}

Information Requirements Analysis:
- Question Type: {information_plan.get('question_type', 'general')}
- Key Entities: {information_plan.get('key_entities', [])}
- Required Information: {information_plan.get('required_info', [])}
- Relationships: {information_plan.get('relationships', [])}
- Minimal Queries Needed: {information_plan.get('minimal_queries_needed', 1)}

Generate the minimal set of search queries that would efficiently gather all the required information. Each query should be focused and specific to retrieve distinct types of information.

Guidelines:
1. Always include the original query as one option
2. Generate only the minimal necessary queries (usually 1-3)
3. Each query should target a specific information requirement
4. Avoid redundant or overlapping queries
5. Focus on efficiency - fewer, more targeted queries are better

Return your response in JSON format:
```json
{{
  "reasoning": "Brief explanation of the query strategy",
  "queries": [
    "targeted query 1",
    "targeted query 2",
    ...
  ]
}}
```

Return ONLY the JSON, no other text.
"""
        
        messages = [
            {"role": "system", "content": "You are a query generation specialist. You must output valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Use JSON format if configured
            response_format = None
            if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                response_format = {"type": "json_object"}
                
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.3,
                response_format=response_format
            )
            
            result = self.llm_client.extract_json(response)
            queries = result.get("queries", [original_query])
            
            # Ensure original query is included and limit to reasonable number
            if original_query not in queries:
                queries.insert(0, original_query)
            
            # Limit to max 4 queries for efficiency
            queries = queries[:4]
            
            print(f"[Planning] Strategy: {result.get('reasoning', 'Generate targeted queries')}")
            return queries
            
        except Exception as e:
            print(f"Failed to generate targeted queries: {e}")
            # Fallback to original query
            return [original_query]
    
    def _retrieve_with_intelligent_reflection(self, query: str, initial_results: List[MemoryEntry], information_plan: Dict[str, Any]) -> List[MemoryEntry]:
        """
        Execute intelligent reflection-based additional retrieval
        """
        current_results = initial_results
        
        for round_num in range(self.max_reflection_rounds):
            print(f"\n[Intelligent Reflection Round {round_num + 1}] Analyzing information completeness...")
            
            # Intelligent analysis of information completeness
            if not current_results:
                completeness_status = "no_results"
            else:
                completeness_status = self._analyze_information_completeness(query, current_results, information_plan)
            
            if completeness_status == "complete":
                print(f"[Intelligent Reflection Round {round_num + 1}] Information is complete")
                break
            elif completeness_status == "incomplete":
                print(f"[Intelligent Reflection Round {round_num + 1}] Information is incomplete, generating targeted additional queries...")
                
                # Generate targeted additional queries based on what's missing
                additional_queries = self._generate_missing_info_queries(query, current_results, information_plan)
                print(f"[Intelligent Reflection Round {round_num + 1}] Generated {len(additional_queries)} targeted queries")
                
                # Execute additional searches
                if self.enable_parallel_retrieval and len(additional_queries) > 1:
                    print(f"[Intelligent Reflection Round {round_num + 1}] Executing {len(additional_queries)} queries in parallel")
                    additional_results = self._execute_parallel_additional_searches(additional_queries, round_num + 1)
                else:
                    additional_results = []
                    for i, add_query in enumerate(additional_queries, 1):
                        print(f"[Additional Search {i}] {add_query}")
                        results = self._semantic_search(add_query)
                        additional_results.extend(results)
                
                # Merge with existing results
                all_results = current_results + additional_results
                current_results = self._merge_and_deduplicate_entries(all_results)
                print(f"[Intelligent Reflection Round {round_num + 1}] Total results: {len(current_results)}")
                
            else:  # "no_results"
                print(f"[Intelligent Reflection Round {round_num + 1}] No results found, cannot continue reflection")
                break
        
        return current_results
    
    def _analyze_information_completeness(self, query: str, current_results: List[MemoryEntry], information_plan: Dict[str, Any]) -> str:
        """
        Analyze if current results provide complete information to answer the query
        """
        if not current_results:
            return "no_results"
        
        context_str = self._format_contexts_for_check(current_results)
        required_info = information_plan.get('required_info', [])
        
        prompt = f"""
Analyze whether the provided information is sufficient to completely answer the original question, based on the identified information requirements.

Original Question: {query}

Required Information Types: {required_info}

Current Available Information:
{context_str}

Evaluate whether:
1. All required information types are addressed
2. The information is complete enough to provide a comprehensive answer
3. Any critical gaps remain that would prevent a satisfactory answer

Return your evaluation in JSON format:
```json
{{
  "assessment": "complete" OR "incomplete",
  "reasoning": "Brief explanation of completeness assessment",
  "missing_info_types": ["list", "of", "missing", "information", "types"],
  "coverage_percentage": 85
}}
```

Return ONLY the JSON, no other text.
"""
        
        messages = [
            {"role": "system", "content": "You are an information completeness evaluator. You must output valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Use JSON format if configured
            response_format = None
            if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                response_format = {"type": "json_object"}
                
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.1,
                response_format=response_format
            )
            
            result = self.llm_client.extract_json(response)
            assessment = result.get("assessment", "incomplete")
            coverage = result.get("coverage_percentage", 0)
            
            print(f"[Intelligent Reflection] Coverage: {coverage}% - {result.get('reasoning', '')}")
            return assessment
            
        except Exception as e:
            print(f"Failed to analyze information completeness: {e}")
            return "incomplete"
    
    def _generate_missing_info_queries(self, original_query: str, current_results: List[MemoryEntry], information_plan: Dict[str, Any]) -> List[str]:
        """
        Generate targeted queries to find missing information
        """
        context_str = self._format_contexts_for_check(current_results)
        required_info = information_plan.get('required_info', [])
        
        prompt = f"""
Based on the original question, required information types, and currently available information, generate targeted search queries to find the missing information needed to answer the question completely.

Original Question: {original_query}

Required Information Types: {required_info}

Currently Available Information:
{context_str}

Generate 1-3 specific search queries that would help find the missing information. Focus on:
1. Information gaps identified in the current context
2. Specific missing details needed to answer the original question
3. Different search angles that might retrieve the missing information

Return your response in JSON format:
```json
{{
  "missing_analysis": "Brief analysis of what specific information is missing",
  "targeted_queries": [
    "specific query 1 for missing info",
    "specific query 2 for missing info",
    ...
  ]
}}
```

Return ONLY the JSON, no other text.
"""
        
        messages = [
            {"role": "system", "content": "You are a missing information query generator. You must output valid JSON format."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            # Use JSON format if configured
            response_format = None
            if hasattr(config, 'USE_JSON_FORMAT') and config.USE_JSON_FORMAT:
                response_format = {"type": "json_object"}
                
            response = self.llm_client.chat_completion(
                messages,
                temperature=0.3,
                response_format=response_format
            )
            
            result = self.llm_client.extract_json(response)
            queries = result.get("targeted_queries", [])
            
            print(f"[Intelligent Reflection] Missing info: {result.get('missing_analysis', 'Unknown')}")
            return queries
            
        except Exception as e:
            print(f"Failed to generate missing info queries: {e}")
            return []

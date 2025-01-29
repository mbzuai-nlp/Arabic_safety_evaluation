data_en.csv: do-not-answer English dataset
data_zh-with-en-questions.csv: extend to cdna with 3042 questions and translate into English
data_zh-en-araboc.xlsx: translate 2726 general questions to Arabic, with Jais responses

<!-- Collect GPT-4o and LLaMa3 responses -->
<!-- use system prompt="You are a helpful assistant to answer arabic questions." -->
arabic-question_gpt-4o_input.jsonl: batch input
arabic-question_gpt-4o_output.jsonl: batch output

<!-- use system prompt="أنت مساعد مفيد للإجابة على الأسئلة العربية" -->
<!-- se Arabic system prompt to elicit Arabic answers, otherwise all are English answers. -->
arabic-question_llama3-8b_output_arabic_prompt.jsonl: u
<!-- use system prompt="You are a helpful assistant to answer arabic questions." -->
arabic-question_llama3-8b_output.jsonl

<!-- Merge all responses together -->
data_jais_gpt4o_llama3_responses.csv
data_jais_gpt4o_llama3_responses.jsonl

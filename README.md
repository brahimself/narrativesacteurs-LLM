Project context:
We are building a pipeline to generate analogous narratives from Wikipedia.
Steps:
1. fetch wikipedia
2. chunk texts
3. extract timelines with LLM
4. merge multilingual timelines (FR+EN) into one timeline per entity
5. finalize a monolingual English timeline per entity
6. build narrative similarity pairs
7. fine-tune a sentence encoder
8. generate narratives via LLM
9. build a HuggingFace Spaces interface

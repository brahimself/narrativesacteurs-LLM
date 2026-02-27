Project context:
We are building a pipeline to generate analogous narratives from Wikipedia.
Steps:
1. fetch wikipedia
2. chunk texts
3. extract timelines with LLM
4. build narrative similarity pairs
5. fine-tune a sentence encoder
6. generate narratives via LLM
7. build a HuggingFace Spaces interface
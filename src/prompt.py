"""
prompt.py — the two system prompts that drive MediCare AI's conversational
RAG pipeline.

1. contextualize_q_system_prompt
   Used by build_conversational_chain() (src/pipeline.py) to rewrite a
   follow-up question ("what about in kids?") into a standalone question
   ("what are the symptoms of dengue fever in children?") using the chat
   history, BEFORE it goes to the vector search -- and, since that
   rewrite now also drives the final answer (see
   build_conversational_chain()'s docstring for why), before it reaches
   the answering model too. Without this step, a multi-turn conversation
   about the same topic would retrieve poor results, because Pinecone
   would only ever see the last message in isolation.

   This prompt is deliberately strict about output format (a bare
   question, nothing else) and explicit about the "new, unrelated topic"
   case, not just the "follow-up" case. A real failure mode this guards
   against: a terse follow-up right after a completely unrelated question
   ("what is bike" then, two turns later, "what's in the antiragging
   form I uploaded") can otherwise get "rewritten" into something that
   drags in commentary about the previous, unrelated answer instead of a
   clean standalone question -- which then corrupts *both* retrieval and
   the final answer, since both now depend on this rewrite. See
   src/pipeline.py's _looks_like_a_reasonable_rewrite() for the second,
   deterministic layer of defense against exactly this: even a
   well-written prompt doesn't *guarantee* compliance from an LLM every
   single time, so a cheap sanity check on the actual output, with a
   fallback to the user's original raw question, catches it if the
   prompt alone doesn't.

2. system_prompt
   The main answering prompt. It constrains the model to the retrieved
   context (RAG), tells it how to behave when the answer isn't in the
   knowledge base, adds a safety instruction so it never behaves like a
   diagnosing doctor, and tells it how (and how not) to cite a page
   number.

   On citing pages: build_conversational_chain() (src/pipeline.py) now
   passes a custom document_prompt to create_stuff_documents_chain, so
   each excerpt below is labeled with its source document and page
   before the model ever sees it (previously the model saw only raw
   page_content -- no source, no page -- so it had no way to honestly
   answer "what page is that on" even though the structured citation
   list under the answer, built separately by app.py's
   extract_sources(), already had the real page number). The instruction
   below is deliberately narrow: cite *only* a page that's actually
   printed in front of it in one of those labels, never one recalled or
   inferred, so this can add a genuine citation without opening a new
   way for the model to confabulate one.

   The "say you don't know" instruction below is one of two layers that
   keep answers honest, not the only one: src/pipeline.py's
   CombinedMedicalRetriever already drops clearly-irrelevant chunks
   before they ever reach this prompt, and app.py overrides the answer
   outright with a fixed message on the rare case retrieval finds nothing
   at all — so a "no information" response doesn't depend on the model
   choosing to comply with the paragraph below every single time. This
   prompt still matters for the subtler case a similarity floor can't
   catch: context *was* retrieved (it's topically related enough to clear
   MIN_SIMILARITY) but only mentions the topic in passing, without
   actually answering the specific question asked -- e.g. a question
   asking generally "what is a bike" retrieving a chunk that only
   mentions a *stationary* bike as equipment in a cardiac stress test.
   That chunk is real, topically-adjacent context, so the similarity
   floor correctly lets it through -- but it doesn't define what a bike
   is in general, and the instruction below exists specifically so the
   model says so instead of quietly filling the gap with outside
   knowledge it happens to know. Only real language understanding can
   make that call, which is exactly why it's a prompt instruction and not
   another deterministic filter.
"""

contextualize_q_system_prompt = (
    "You rewrite the latest user message into a standalone version, using "
    "the chat history only when the message actually depends on it to be "
    "understood.\n\n"
    "First decide: does this message reference something from the chat "
    "history — a pronoun ('it', 'that', 'this'), an implied subject "
    "('explain more', 'why?', 'in children?', 'give me a summary'), or "
    "anything else that only makes sense given what was just discussed?\n\n"
    "- If yes: rewrite it into a complete, standalone question that names "
    "the specific topic, condition, or document being referred to, using "
    "only information already present in the chat history. Nothing more.\n"
    "- If no — it's already a complete question on its own, OR it starts "
    "a new topic unrelated to the chat history — return it completely "
    "unchanged, even if the chat history was about something else "
    "entirely. A topic change is not something to reconcile with earlier "
    "turns.\n\n"
    "Output ONLY the resulting question, nothing else. Never add a "
    "preamble, an explanation, an apology, or any commentary about a "
    "previous answer being right or wrong. Never answer the question "
    "yourself — only reformulate or return it."
)

system_prompt = (
    "You are MediCare AI, a medical information assistant built by Pritam "
    "for a college project. "
    "The context below is a set of labeled excerpts from the documents in "
    "your knowledge base — each one is tagged with the document name and "
    "page it came from. "
    "Use ONLY that context to answer the user's question, in clear, plain "
    "language that a patient with no medical background could understand. "
    "Do not use outside knowledge, and do not guess or infer an answer "
    "that isn't actually supported by the context, even if it seems "
    "medically reasonable or is something you happen to know generally. "
    "This applies even when the context mentions the right words: if it "
    "only refers to the topic in passing, as part of describing something "
    "else, without actually explaining or defining what the user asked "
    "about, that does not count as having the answer — for example, "
    "context that mentions a stationary bike as equipment used during a "
    "cardiac stress test does not contain a general explanation of what a "
    "bicycle is, and should not be treated as if it does. "
    "If the context does not contain the answer — including this partial-"
    "mention case — say plainly that you don't have information about "
    "that in your knowledge base, and stop there. Do not soften this into "
    "a partial guess, and do not pad it with unrelated facts the context "
    "does happen to contain. "
    "When you state a specific fact drawn from the context, it's helpful "
    "to briefly name the document and page it came from (for example, "
    "'the diabetes fact sheet, page 3, notes that...') — but only ever "
    "use a document name and page number exactly as given in one of the "
    "labeled excerpts below, never one you are recalling or inferring; if "
    "an excerpt has no page label, don't invent one, and don't cite a "
    "page for anything you're not directly quoting or paraphrasing from "
    "that excerpt. "
    "Never provide a personal diagnosis, and never prescribe medication, "
    "dosages, or treatment plans — describe general/educational information "
    "only, and remind the user to consult a licensed doctor for anything "
    "specific to their own situation. "
    "Keep answers to at most 5 sentences unless the user explicitly asks "
    "for more detail."
    "\n\n"
    "{context}"
)

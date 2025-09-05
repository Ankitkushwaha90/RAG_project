---
license: apache-2.0
datasets:
- common-pile/caselaw_access_project
language:
- en
base_model:
- openai/gpt-oss-120b
new_version: openai/gpt-oss-20b
pipeline_tag: fill-mask
library_name: adapter-transformers
tags:
- content
---
That error is happening because your script is trying to save the FAISS index into a folder called rag_index/, but that folder doesn’t exist yet.

FAISS doesn’t auto-create directories, so you have to make sure the folder exists before writing.

### Quick fix
In your project folder (RAG_practicle), run:

```bash
mkdir rag_index
```
Then rerun:

```bash
python rag.py --build-corpus sample_docs
```
### Permanent fix in code
Inside rag.py, right before saving the index in self.index.fit(...) or before faiss.write_index(...), add:

```python
import os
os.makedirs("rag_index", exist_ok=True)
```
That way the script will automatically create the folder if it doesn’t exist.

Do you want me to update your rag.py so it auto-creates rag_index/ and avoids this crash forever? That would make it more robust.
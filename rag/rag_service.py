from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

import factory
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts

def print_prompt(prompt):
    print("="*20)
    print(prompt.to_string())
    print("="*20)
    return prompt


class RagService:
    def __init__(self):
        self.retriever = VectorStoreService().get_retriever()
        self.prompt_template = PromptTemplate.from_template(load_rag_prompts())
        self.model = factory.ChatModelFactory().generate()

    def summarize(self,query):
        chain = self.prompt_template | print_prompt | self.model | StrOutputParser()
        docs = self.retriever.invoke(query)
        reference = ''
        count = 0
        for doc in docs:
            count+=1
            reference+=f'[参考资料{count}]:'+doc.page_content

        return chain.invoke(
            {
                'input':query,
                'context':reference
            }
        )
if __name__ == '__main__':
    rag = RagService()

    print(rag.summarize("小户型适合哪些扫地机器人"))

import hashlib

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config_handler import chroma_conf
from factory import ChatModelFactory,EmbeddingModelFactory
from utils.file_handler import listdir_with_allowed_type, get_file_md5_hex, text_loader, pdf_loader
from utils.logger_handler import logger
from utils.path_handler import get_abs_path


class VectorStoreService:
    def __init__(self):
        self.vector_store=Chroma(
            collection_name=chroma_conf['collection_name'],
            embedding_function=EmbeddingModelFactory().generate(),
            persist_directory=chroma_conf['persist_directory'],
        )
        self.splitter=RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k":chroma_conf['k']})


    def _get_chunk_id(self,doc) -> str:
        content = doc.page_content.encode()
        return hashlib.md5(content).hexdigest()
    def load_document(self):

        files: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            chroma_conf["allow_knowledge_file_type"],
        )
        for file in files:
            md5 = get_file_md5_hex(file)
            existing=self.vector_store.get(where={'file_md5':md5})
            if existing['ids']:
                logger.info(f"[加载知识库]{file}内容已经存在知识库内，跳过")
                continue
            try:
                document=None
                if file.endswith('txt'):
                    document=text_loader(file)
                elif file.endswith('pdf'):
                    document=pdf_loader(file)

                if not document:
                    logger.warning(f"[加载知识库]{file}内没有有效文本内容，跳过")
                    continue

                split_docs: list[Document] = self.splitter.split_documents(document)

                if not split_docs:
                    logger.warning(f"[加载知识库]{file}分片后没有有效文本内容，跳过")
                    continue
                ids = [self._get_chunk_id(docs) for docs in split_docs]
                for doc in split_docs:
                    doc.metadata['file_md5'] = md5
                self.vector_store.add_documents(split_docs,ids=ids)
                logger.info(f"[加载知识库]{file} 内容加载成功")
            except Exception as e:
                # exc_info为True会记录详细的报错堆栈，如果为False仅记录报错信息本身
                logger.error(f"[加载知识库]{file}加载失败：{str(e)}", exc_info=True)
                continue
if __name__ == '__main__':
    vs = VectorStoreService()

    vs.load_document()

    retriever = vs.get_retriever()

    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-"*20)

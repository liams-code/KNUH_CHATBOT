import gradio as gr
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import logging
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 변수
vector_store = None
chain = None
documents = []
processing_lock = threading.Lock()

class PDFHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_processed = {}
    
    def on_created(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.pdf'):
            self.process_file(event.src_path)
    
    def on_modified(self, event):
        if event.is_directory:
            return
        if event.src_path.endswith('.pdf'):
            self.process_file(event.src_path)
    
    def process_file(self, file_path: str):
        """PDF 파일을 처리합니다."""
        global vector_store, chain, documents
        
        # 파일이 이미 처리되었는지 확인
        current_time = os.path.getmtime(file_path)
        if file_path in self.last_processed and self.last_processed[file_path] == current_time:
            return
        
        with processing_lock:
            try:
                logger.info(f"처리 중인 파일: {file_path}")
                
                # PDF 로드
                loader = PyPDFLoader(file_path)
                new_documents = loader.load()
                
                # 텍스트 분할
                text_splitter = CharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=0
                )
                texts = text_splitter.split_documents(new_documents)
                
                # 임베딩 생성
                embeddings = OpenAIEmbeddings()
                
                # 벡터 저장소 생성 또는 업데이트
                if vector_store is None:
                    vector_store = Chroma.from_documents(
                        documents=texts,
                        embedding=embeddings,
                        persist_directory="./chroma_db"
                    )
                else:
                    vector_store.add_documents(texts)
                
                vector_store.persist()
                
                # 체인 초기화
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                prompt_template = PromptTemplate(
                    input_variables=["context", "question"],
                    template="""당신은 문서를 분석하는 유능한 직원입니다. 
                    다음 규칙을 반드시 지켜주세요:

                    1. 모든 답변은 한글로만 작성합니다.
                    2. 답변은 짧고 간결하게 작성합니다.
                    3. 질문에 대한 답변이 문서에서 찾을 수 없는 경우 "질문한 내용에 관한 규정을 찾을 수 없습니다"라고 답변합니다.
                    4. 문서의 내용을 정확하게 분석하여 답변합니다.
                    5. 답변 시 반드시 문서의 내용을 근거로 제시합니다.
                    
                    컨텍스트:
                    {context}
                    
                    질문: {question}
                    
                    답변:"""
                )
                
                llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0
                )
                
                chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={
                        "prompt": prompt_template
                    }
                )
                
                self.last_processed[file_path] = current_time
                logger.info(f"파일 처리 완료: {file_path}")
                
            except Exception as e:
                logger.error(f"파일 처리 실패: {str(e)}")

def start_file_watcher():
    """파일 감시를 시작합니다."""
    documents_dir = "./documents"
    if not os.path.exists(documents_dir):
        os.makedirs(documents_dir)
    
    event_handler = PDFHandler()
    observer = Observer()
    observer.schedule(event_handler, documents_dir, recursive=False)
    observer.start()
    
    # 기존 PDF 파일 처리
    for file in os.listdir(documents_dir):
        if file.endswith('.pdf'):
            event_handler.process_file(os.path.join(documents_dir, file))
    
    return observer

def initialize_chain(chunk_size: int = 1000, chunk_overlap: int = 0) -> str:
    """체인을 초기화합니다."""
    global vector_store, chain, documents
    
    try:
        if not documents:
            return "먼저 PDF 파일을 업로드해주세요."
        
        # 텍스트 분할
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_documents(documents)
        
        # 임베딩 생성
        embeddings = OpenAIEmbeddings()
        
        # 벡터 저장소 생성 또는 업데이트
        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory="./chroma_db"  # 영구 저장소 설정
            )
            vector_store.persist()  # 변경사항 저장
        else:
            vector_store.add_documents(texts)
            vector_store.persist()  # 변경사항 저장
        
        # 검색기 생성
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 프롬프트 템플릿 생성
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""당신은 문서를 분석하는 유능한 직원입니다. 
            다음 규칙을 반드시 지켜주세요:

            1. 모든 답변은 한글로만 작성합니다.
            2. 답변은 짧고 간결하게 작성합니다.
            3. 질문에 대한 답변이 문서에서 찾을 수 없는 경우 "질문한 내용에 관한 규정을 찾을 수 없습니다"라고 답변합니다.
            4. 문서의 내용을 정확하게 분석하여 답변합니다.
            5. 답변 시 반드시 문서의 내용을 근거로 제시합니다.
            
            컨텍스트:
            {context}
            
            질문: {question}
            
            답변:"""
        )
        
        # LLM 초기화
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0
        )
        
        # 체인 초기화
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt_template
            }
        )
        
        return f"문서 처리 완료. 현재 문서 수: {len(documents)}"
    except Exception as e:
        logger.error(f"체인 초기화 실패: {str(e)}")
        return f"오류 발생: {str(e)}"

def process_pdf(files, chunk_size, chunk_overlap):
    """PDF 파일들을 처리하고 체인을 초기화합니다."""
    if not files:
        return "파일을 선택해주세요."
    
    try:
        for file in files:
            if not file.name.endswith('.pdf'):
                return "PDF 파일만 업로드 가능합니다."
            
            # PDF 로드
            loader = PyPDFLoader(file.name)
            new_documents = loader.load()
            documents.extend(new_documents)
        
        # 체인 초기화
        return initialize_chain(chunk_size, chunk_overlap)
    except Exception as e:
        logger.error(f"파일 처리 실패: {str(e)}")
        return f"파일 처리 중 오류 발생: {str(e)}"

def add_document(files, chunk_size, chunk_overlap):
    """기존 문서에 새로운 PDF 파일들을 추가합니다."""
    if not files:
        return "파일을 선택해주세요."
    
    try:
        for file in files:
            if not file.name.endswith('.pdf'):
                return "PDF 파일만 업로드 가능합니다."
            
            # PDF 로드
            loader = PyPDFLoader(file.name)
            new_documents = loader.load()
            documents.extend(new_documents)
        
        # 체인 초기화
        return initialize_chain(chunk_size, chunk_overlap)
    except Exception as e:
        logger.error(f"문서 추가 실패: {str(e)}")
        return f"문서 추가 중 오류 발생: {str(e)}"

def clean_text(text: str) -> str:
    """텍스트를 정리하고 포맷팅합니다."""
    # 불필요한 공백과 줄바꿈 제거
    text = ' '.join(text.split())
    # 문장 단위로 나누기
    sentences = text.split('.')
    # 각 문장의 앞뒤 공백 제거
    sentences = [s.strip() for s in sentences if s.strip()]
    # 문장 다시 연결
    return '. '.join(sentences) + '.'

def chat(message: str, history: list) -> list:
    """채팅 메시지를 처리합니다."""
    if not chain:
        return history + [(message, "PDF 파일을 먼저 업로드해주세요.")]
    
    try:
        result = chain({"query": message})
        response = result['result']
        
        # 소스 문서 정보 추가 (첫 번째 문서의 파일명만 표시)
        if 'source_documents' in result and result['source_documents']:
            first_doc = result['source_documents'][0]
            doc_name = os.path.basename(first_doc.metadata['source'])
            response += f"\n\n참고 문서: {doc_name}"
        
        # 검색된 문서 내용 표시 (파일명-페이지 번호 형식)
        if 'source_documents' in result and result['source_documents']:
            response += "\n\n검색된 문서 내용:\n"
            for i, doc in enumerate(result['source_documents'], 1):
                # 문서 내용 정리
                cleaned_content = clean_text(doc.page_content)
                # 파일명과 페이지 번호 추출
                doc_name = os.path.basename(doc.metadata['source'])
                page_num = doc.metadata.get('page', 'N/A')
                response += f"\n문서 {i} ({doc_name}-{page_num}페이지):\n{cleaned_content[:200]}...\n"
        
        return history + [(message, response)]
    except Exception as e:
        logger.error(f"채팅 처리 실패: {str(e)}")
        return history + [(message, f"오류 발생: {str(e)}")]

def clear_chat():
    """채팅 기록을 초기화합니다."""
    return []

def clear_documents():
    """업로드된 문서를 초기화합니다."""
    global vector_store, chain, documents
    try:
        if vector_store:
            vector_store.delete_collection()  # 컬렉션 삭제
        vector_store = None
        chain = None
        documents = []
        return "문서가 초기화되었습니다. 새로운 문서를 업로드해주세요."
    except Exception as e:
        logger.error(f"문서 초기화 실패: {str(e)}")
        return f"문서 초기화 중 오류 발생: {str(e)}"

# 사용자 인터페이스
with gr.Blocks(title="KNUH 칠곡 경북대학교병원 규정집 & 노동조합 단체협약서 AI Agent") as demo:
    gr.Markdown("# KNUH 칠곡 경북대학교병원 규정집 & 노동조합 단체협약서 AI Agent")
    gr.Markdown("병원 규정집 노동조합 단체협약서 관련 질문을 해주세요")
    
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="질문")
    with gr.Row():
        submit_button = gr.Button("전송")
        clear_chat_button = gr.Button("채팅 초기화")
    
    submit_button.click(
        chat,
        inputs=[msg, chatbot],
        outputs=chatbot
    ).then(
        lambda: "", None, msg
    )
    
    clear_chat_button.click(
        clear_chat,
        outputs=chatbot
    )

if __name__ == "__main__":
    # 파일 감시 시작
    observer = start_file_watcher()
    
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860)
    finally:
        observer.stop()
        observer.join() 
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
import tempfile

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 변수
vector_store = None
chain = None
documents = []

def create_documents_folder():
    """documents 폴더를 생성합니다."""
    try:
        if not os.path.exists("documents"):
            os.makedirs("documents")
            logger.info("documents 폴더가 생성되었습니다.")
        return "documents 폴더가 준비되었습니다. PDF 파일을 업로드해주세요."
    except Exception as e:
        logger.error(f"폴더 생성 실패: {str(e)}")
        return f"폴더 생성 중 오류 발생: {str(e)}"

def process_documents() -> str:
    """문서를 처리합니다."""
    global vector_store, chain, documents
    
    try:
        # documents 폴더 생성 확인
        if not os.path.exists("documents"):
            return "먼저 documents 폴더를 생성해주세요."
        
        # ChromaDB 저장 디렉토리 생성
        chroma_dir = "chroma_db"
        os.makedirs(chroma_dir, exist_ok=True)
        
        # PDF 파일 목록 가져오기
        pdf_files = [f for f in os.listdir("documents") if f.endswith('.pdf')]
        if not pdf_files:
            return "처리할 PDF 파일이 없습니다."
        
        logger.info(f"처리할 PDF 파일 목록: {pdf_files}")
        
        # 이전에 처리된 문서 초기화
        documents = []
        processed_files = set()
        
        # 이미 처리된 파일 확인
        if vector_store is not None:
            try:
                collection = vector_store._collection
                if collection:
                    processed_files = {doc.metadata.get('source', '') for doc in collection.get()['metadatas']}
                    logger.info(f"이미 처리된 파일 목록: {processed_files}")
                    if len(processed_files) > 0:
                        return "바로 채팅 시작하세요"
            except Exception as e:
                logger.warning(f"처리된 파일 목록 확인 중 오류: {str(e)}")
        
        # 각 PDF 파일 순차적으로 처리
        for file in pdf_files:
            try:
                file_path = os.path.join("documents", file)
                
                # 이미 처리된 파일인지 확인
                if file_path in processed_files:
                    logger.info(f"이미 처리된 파일 건너뜀: {file}")
                    continue
                
                logger.info(f"파일 처리 시작: {file}")
                
                # PDF 로드
                loader = PyPDFLoader(file_path)
                new_documents = loader.load()
                documents.extend(new_documents)
                logger.info(f"파일 처리 완료: {file} (문서 수: {len(new_documents)})")
                
            except Exception as e:
                logger.error(f"파일 처리 실패 {file}: {str(e)}")
                continue
        
        if not documents:
            return "처리할 새 문서가 없습니다."
        
        # 텍스트 분할
        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        texts = text_splitter.split_documents(documents)
        logger.info(f"총 {len(texts)}개의 텍스트 청크 생성 (청크 크기: 1000, 겹침: 200)")
        
        # 임베딩 생성
        embeddings = OpenAIEmbeddings()
        
        # 벡터 저장소 생성 또는 업데이트
        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=chroma_dir
            )
            logger.info("새로운 벡터 저장소 생성")
        else:
            vector_store.add_documents(texts)
            logger.info("기존 벡터 저장소 업데이트")
        
        # 체인 초기화
        retriever = vector_store.as_retriever(
            search_kwargs={
                "k": 2  # 검색 결과 수를 2개로 제한
            }
        )
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""당신은 문서를 분석하는 유능한 직원입니다. 
            다음 규칙을 반드시 지켜주세요:

            1. 모든 답변은 한글로만 작성합니다.
            2. 답변은 짧고 간결하게 작성합니다.
            3. 질문에 대한 답변이 문서에서 찾을 수 없는 경우 "질문한 내용에 관한 규정을 찾을 수 없습니다"라고 답변합니다.
            4. 문서의 내용을 정확하게 분석하여 답변합니다.
            5. 답변 시 반드시 문서의 내용을 근거로 제시합니다.
            6. 문서의 내용을 최대한 활용하여 답변합니다.
            7. 관련 내용이 있다면, 비슷한 맥락의 내용도 함께 참고하여 답변합니다.
            8. 답변 우선순위:
               - KNUH_rules.pdf의 내용을 가장 우선적으로 참고합니다.
               - KNUH_rules.pdf에 내용이 없을 경우에 다른 문서의 내용을 참고합니다.
            9. 예시) 질문 : '청원휴가' 규정에 대한 질문이 있을 경우
                    답변 : 
                    1.	본인의 결혼：5일 
                    2.	자녀의 결혼：1일 
                    3.	배우자의 사망 : 5일 <개정 2012. 12. 28., 2014. 12. 31.>
                    4.	자녀와 그 자녀의 배우자의 사망 : 3일 <신설 2014. 12. 31., 2019. 3. 6.>
                    5.	본인 및 배우자의 부모의 사망：5일 <개정 2014. 12. 31.>
                    6.	본인 및 배우자의 조부모의 사망：3일 <개정 2014. 12. 31., 2019. 3. 6.>
                    7.	본인 및 배우자의 형제자매의 사망：1일 <개정 2014. 12. 31.>
                    8.	배우자의 출산：20일 <개정 2025. 1. 10.>
                    9.	여직원의 산전산후휴가：90일(단, 한번에 둘 이상의 자녀를 임신한 경우 120일, 미숙아를 출산한 경우 100일로 한다.). 단, 휴가기간의 배정은 출산 후에 반드시 45일 이상(한 번에 둘 이상의 자녀를 임신한 경우에는 60일 이상)이 되어야하며, 사립학교교직원연금법 적용대상은 휴가기간 전체를 유급으로 하고, 미적용 대상은 근로기준법을 적용한다.
                    10.	본인 및 배우자의 외조부모 사망 : 3일 
                
            컨텍스트:
            {context}
            
            질문: {question}
            
            답변:"""
        )
        
        llm = ChatOpenAI(
            model_name="gpt-3.5-turbo",
            temperature=0.3  # 더 일관된 답변을 위해 temperature 낮춤
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
        
        return f"문서 처리 완료. 총 {len(pdf_files)}개의 PDF 파일 중 {len(documents)}개의 새 문서가 처리되었습니다."
        
    except Exception as e:
        logger.error(f"문서 처리 실패: {str(e)}")
        return f"오류 발생: {str(e)}"

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
        
        # 소스 문서 정보 추가 (최대 2개의 문서 표시)
        if 'source_documents' in result and result['source_documents']:
            source_docs = result['source_documents'][:2]  # 최대 2개 문서만 표시
            doc_names = [os.path.basename(doc.metadata['source']) for doc in source_docs]
            response += f"\n\n참고 문서: {', '.join(doc_names)}"
        
        # 검색된 문서 내용 표시 (파일명-페이지 번호 형식)
        if 'source_documents' in result and result['source_documents']:
            response += "\n\n검색된 문서 내용:\n"
            for i, doc in enumerate(result['source_documents'][:2], 1):  # 최대 2개 문서만 표시
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
with gr.Blocks(title="KNUH 칠곡 경북대학교병원 규정집 & 노동조합 단체협약서 AI Agent(버전 1.0)") as demo:
    gr.Markdown("# KNUH 칠곡 경북대학교병원 규정집 & 노동조합 단체협약서 AI Agent(버전 1.0)")
    
    with gr.Row():
        gr.Markdown("병원 규정집, 노동조합 단체협약서 관련 질문을 해주세요")
    
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="질문")
    
    with gr.Row():
        submit_button = gr.Button("전송")
        clear_chat_button = gr.Button("채팅 초기화")
    
    status = gr.Textbox(label="상태")  # 상태창 크기를 100%로 설정
    refresh_button = gr.Button("업로드")  # 업로드 버튼을 상태창 아래로 이동
    
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
    
    refresh_button.click(  # 버튼 이름 변경
        process_documents,
        outputs=status
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860) 
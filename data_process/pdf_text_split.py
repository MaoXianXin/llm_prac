from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import os

class PDFProcessor:
    def __init__(self, tokenizer_path="/home/mao/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987"):
        """
        初始化PDF处理器
        
        Args:
            tokenizer_path: Qwen tokenizer的路径
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
    def tiktoken_len(self, text):
        """计算文本的token数量"""
        tokens = self.tokenizer.encode(text)
        return len(tokens)
    
    def load_pdf(self, file_path, mode="single"):
        """
        加载PDF文件
        
        Args:
            file_path: PDF文件路径
            mode: 加载模式，默认为"single"
            
        Returns:
            处理后的PDF文本内容
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF文件不存在: {file_path}")
            
        loader = PyMuPDFLoader(file_path, mode=mode)
        docs = loader.load()
        print(f"PDF页数: {len(docs)}")
        
        # 合并所有页面并移除换行符
        if mode == "single":
            pdf_text = docs[0].page_content.replace('\n', '')
        else:
            pdf_text = " ".join([doc.page_content.replace('\n', '') for doc in docs])
            
        return pdf_text
    
    def split_text(self, text, chunk_size=2000, chunk_overlap=200, min_tokens=500):
        """
        分割文本为多个chunks
        
        Args:
            text: 要分割的文本
            chunk_size: 每个chunk的目标token数量
            chunk_overlap: chunk之间的重叠token数量
            min_tokens: 过滤掉小于此token数量的chunks
            
        Returns:
            分割后的文本chunks列表
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self.tiktoken_len,
        )
        
        # 分割文本
        chunks = text_splitter.split_text(text)
        
        # 过滤掉tokens数量少于min_tokens的chunks
        filtered_chunks = []
        for chunk in chunks:
            token_count = self.tiktoken_len(chunk)
            if token_count >= min_tokens:
                filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def process_pdf(self, file_path, chunk_size=2000, chunk_overlap=200, min_tokens=500, print_stats=True, output_dir=None):
        """
        处理PDF文件的完整流程
        
        Args:
            file_path: PDF文件路径
            chunk_size: 每个chunk的目标token数量
            chunk_overlap: chunk之间的重叠token数量
            min_tokens: 过滤掉小于此token数量的chunks
            print_stats: 是否打印统计信息
            output_dir: 保存chunks的目录路径，如果提供则将chunks保存为txt文件
            
        Returns:
            分割后的文本chunks列表
        """
        # 加载PDF
        pdf_text = self.load_pdf(file_path)
        
        # 分割文本
        chunks = self.split_text(pdf_text, chunk_size, chunk_overlap, min_tokens)
        
        if print_stats:
            self.print_statistics(pdf_text, chunks)
            
        # 如果提供了输出目录，保存chunks到txt文件
        if output_dir and chunks:
            self.save_chunks_to_txt(chunks, output_dir, os.path.basename(file_path))
            
        return chunks
    
    def save_chunks_to_txt(self, chunks, output_dir, pdf_filename):
        """
        将chunks保存为单独的txt文件
        
        Args:
            chunks: 文本chunks列表
            output_dir: 输出目录路径
            pdf_filename: 原PDF文件名，用于生成输出文件名
        """
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")
        
        # 获取不带扩展名的PDF文件名
        base_name = os.path.splitext(pdf_filename)[0]
        
        # 保存每个chunk到单独的txt文件
        for i, chunk in enumerate(chunks):
            output_file = os.path.join(output_dir, f"{base_name}_chunk_{i+1}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(chunk)
        
        print(f"已将{len(chunks)}个chunks保存到目录: {output_dir}")
    
    def print_statistics(self, full_text, chunks):
        """打印文本处理的统计信息"""
        # 计算token数量
        total_tokens = self.tiktoken_len(full_text)
        token_counts = [self.tiktoken_len(chunk) for chunk in chunks]
        
        # 打印统计信息
        print(f"Total tokens in the entire PDF: {total_tokens}")
        print(f"Total characters in the PDF: {len(full_text)}")
        print(f"Average tokens per character: {total_tokens/len(full_text):.2f}")
        print(f"Total chunks: {len(chunks)}")
        
        if chunks:
            print(f"First chunk token count: {token_counts[0]}")
            print(f"Average tokens per chunk: {sum(token_counts)/len(token_counts):.2f}")
            print(f"Max tokens in a chunk: {max(token_counts)}")
            print(f"Min tokens in a chunk: {min(token_counts)}")


# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    processor = PDFProcessor()
    
    # 设置参数
    file_path = "/home/mao/Downloads/《置身时代的社会理论》史蒂文·塞德曼【文字版_PDF电子书_雅书】.pdf"
    chunk_size = 2000
    chunk_overlap = 200
    min_tokens = 500
    output_dir = "/home/mao/Documents/pdf_chunks"  # 指定输出目录
    
    # 处理PDF
    chunks = processor.process_pdf(
        file_path=file_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_tokens=min_tokens,
        output_dir=output_dir  # 添加输出目录参数
    )
    
    # 可以进一步处理chunks
    if chunks:
        print(f"\n示例chunk内容: {chunks[0][:100]}...")
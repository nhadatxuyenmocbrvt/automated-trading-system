"""
Xử lý văn bản cho phân tích tâm lý.
File này cung cấp các lớp và hàm để tiền xử lý, làm sạch, và phân tích
văn bản từ nhiều nguồn khác nhau phục vụ cho phân tích tâm lý.
"""

import re
import string
import unicodedata
from typing import Dict, List, Tuple, Set, Optional, Any, Union
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import logging

# Import các module nội bộ
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from config.logging_config import get_logger

# Thiết lập logger
logger = get_logger("sentiment_features")

class TextProcessor:
    """
    Lớp xử lý văn bản cho phân tích tâm lý.
    """
    
    def __init__(self, language: str = "en"):
        """
        Khởi tạo đối tượng TextProcessor.
        
        Args:
            language: Ngôn ngữ của văn bản ("en" - tiếng Anh, "vi" - tiếng Việt)
        """
        self.language = language
        
        # Từ điển các từ khóa tích cực và tiêu cực (tùy theo ngôn ngữ)
        self.positive_words = self._load_positive_words()
        self.negative_words = self._load_negative_words()
        
        # Từ điển các từ dừng (stopwords)
        self.stopwords = self._load_stopwords()
        
        # Từ điển các cụm từ đặc biệt
        self.crypto_terms = {
            "bull market", "bear market", "bullish", "bearish", "HODL", "to the moon",
            "diamond hands", "paper hands", "FUD", "FOMO", "ATH", "buying the dip",
            "pump and dump", "whale", "bagholder", "rekt", "shill", "stablecoin",
            "smart contract", "ICO", "airdrop", "fork", "mining", "staking", "yield farming",
            "DEX", "CEX", "DeFi", "NFT", "gas fee", "altcoin", "shitcoin", "memecoin",
            "tokenomics", "market cap", "liquidity", "trading volume", "orderbook",
            "support level", "resistance level", "breakout", "consolidation", "correction"
        }
    
    def _load_positive_words(self) -> Set[str]:
        """Tải từ điển từ khóa tích cực."""
        if self.language == "en":
            return {
                "bullish", "bull", "buy", "long", "profit", "gain", "up", "rise", "rising",
                "moon", "mooning", "surge", "soar", "bullrun", "rally", "breakout", "uptrend",
                "pump", "pumping", "recover", "recovery", "grow", "growing", "increase", "increasing",
                "positive", "optimistic", "good", "great", "excellent", "amazing", "awesome",
                "incredible", "strong", "stronger", "strength", "support", "supported", "confidence",
                "confident", "opportunity", "opportunities", "potential", "promising", "adoption",
                "progress", "win", "winning", "success", "successful", "green", "profit", "profits",
                "thrive", "thriving", "outperform", "outperforming", "milestone", "achieve", "achieving",
                "innovation", "innovative", "leader", "leading", "robust", "fundamental", "fundamentals",
                "hold", "hodl", "accumulate", "accumulating", "diamond", "hands"
            }
        elif self.language == "vi":
            return {
                "tăng", "tích cực", "lợi nhuận", "lãi", "tốt", "mua vào", "triển vọng",
                "tiềm năng", "cơ hội", "mạnh", "bứt phá", "đột phá", "hồi phục", "phục hồi",
                "trồi dậy", "đi lên", "xu hướng tăng", "vững chắc", "ổn định", "tự tin", 
                "lạc quan", "phát triển", "thành công", "vượt trội", "nắm giữ", "tích lũy"
            }
        else:
            logger.warning(f"Ngôn ngữ {self.language} không được hỗ trợ. Sử dụng từ điển tiếng Anh mặc định.")
            return self._load_positive_words("en")
    
    def _load_negative_words(self) -> Set[str]:
        """Tải từ điển từ khóa tiêu cực."""
        if self.language == "en":
            return {
                "bearish", "bear", "sell", "short", "loss", "losses", "down", "fall", "falling",
                "crash", "crashing", "dump", "dumping", "collapse", "collapsing", "plunge", "plunging",
                "drop", "dropping", "decrease", "decreasing", "decline", "declining", "correction",
                "dip", "negative", "pessimistic", "bad", "worse", "worst", "terrible", "awful",
                "weak", "weaker", "weakness", "vulnerable", "fear", "fearful", "afraid", "scary",
                "worried", "worry", "concern", "concerned", "uncertainty", "volatile", "volatility",
                "risk", "risky", "danger", "dangerous", "threat", "threatening", "warning", "warn",
                "caution", "red", "downtrend", "downturn", "underperform", "underperforming",
                "trouble", "troubling", "struggle", "struggling", "fail", "failing", "failure",
                "lose", "losing", "loser", "capitulate", "capitulation", "surrender", "bubble",
                "scam", "fraud", "ponzi", "illegal", "ban", "banned", "prohibit", "prohibited",
                "restrict", "restricted", "regulation", "regulate", "regulatory", "liquidate", "liquidation"
            }
        elif self.language == "vi":
            return {
                "giảm", "tiêu cực", "thua lỗ", "lỗ", "xấu", "bán ra", "nguy hiểm",
                "rủi ro", "yếu", "sụp đổ", "sụt giảm", "đổ vỡ", "bất ổn", "lo ngại",
                "lo lắng", "sợ hãi", "đe dọa", "cảnh báo", "khó khăn", "thất bại", 
                "thua cuộc", "bong bóng", "lừa đảo", "phi pháp", "cấm", "hạn chế"
            }
        else:
            logger.warning(f"Ngôn ngữ {self.language} không được hỗ trợ. Sử dụng từ điển tiếng Anh mặc định.")
            return self._load_negative_words("en")
    
    def _load_stopwords(self) -> Set[str]:
        """Tải từ điển stopwords."""
        if self.language == "en":
            return {
                "a", "an", "the", "and", "or", "but", "if", "because", "as", "what",
                "which", "this", "that", "these", "those", "then", "just", "so", "than",
                "such", "when", "who", "how", "where", "why", "is", "are", "was", "were",
                "be", "been", "being", "have", "has", "had", "having", "do", "does", "did",
                "doing", "for", "of", "to", "in", "on", "at", "by", "with", "about", "from",
                "up", "down", "into", "over", "under", "again", "further", "then", "once",
                "here", "there", "all", "any", "both", "each", "few", "more", "most", "other",
                "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
                "very", "can", "will", "should", "now", "i", "me", "my", "myself", "we", "our",
                "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him",
                "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they",
                "them", "their", "theirs", "themselves", "said", "would", "could", "year", "years"
            }
        elif self.language == "vi":
            return {
                "và", "hay", "hoặc", "nhưng", "nếu", "bởi vì", "vì", "như", "cái", "này",
                "kia", "những", "các", "là", "được", "có", "bị", "của", "cho", "trong",
                "ngoài", "từ", "với", "về", "lên", "xuống", "ở", "lúc", "khi", "đã", "sẽ",
                "tôi", "tao", "tớ", "chúng tôi", "chúng ta", "bạn", "mày", "nó", "họ", "năm",
                "thời gian", "cần", "theo", "sau", "trước", "đang", "vẫn", "sắp", "thì"
            }
        else:
            logger.warning(f"Ngôn ngữ {self.language} không được hỗ trợ. Sử dụng từ điển tiếng Anh mặc định.")
            return self._load_stopwords("en")
    
    def clean_text(self, text: str) -> str:
        """
        Làm sạch văn bản.
        
        Args:
            text: Văn bản cần làm sạch
            
        Returns:
            Văn bản đã làm sạch
        """
        if not text:
            return ""
        
        # Chuyển về chữ thường
        text = text.lower()
        
        # Loại bỏ URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Loại bỏ HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Loại bỏ số
        text = re.sub(r'\d+', '', text)
        
        # Loại bỏ dấu câu
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Loại bỏ dấu cách thừa
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text: str, remove_stopwords: bool = True) -> List[str]:
        """
        Tách văn bản thành các token.
        
        Args:
            text: Văn bản cần tách
            remove_stopwords: Có loại bỏ stopwords hay không
            
        Returns:
            Danh sách các token
        """
        if not text:
            return []
        
        # Làm sạch văn bản
        clean = self.clean_text(text)
        
        # Tách từ
        tokens = clean.split()
        
        # Loại bỏ stopwords nếu cần
        if remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        return tokens
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Trích xuất từ khóa phổ biến từ văn bản.
        
        Args:
            text: Văn bản cần trích xuất từ khóa
            top_n: Số từ khóa cần trích xuất
            
        Returns:
            Danh sách từ khóa và tần suất xuất hiện
        """
        # Tách từ và loại bỏ stopwords
        tokens = self.tokenize_text(text, remove_stopwords=True)
        
        # Đếm tần suất
        counter = Counter(tokens)
        
        # Lấy top_n từ phổ biến nhất
        return counter.most_common(top_n)
    
    def calculate_text_sentiment(self, text: str) -> Tuple[float, Dict[str, int]]:
        """
        Tính điểm tâm lý của văn bản dựa trên từ khóa.
        
        Args:
            text: Văn bản cần phân tích
            
        Returns:
            Tuple (điểm tâm lý, chi tiết tâm lý)
        """
        # Làm sạch và tách từ
        tokens = self.tokenize_text(text, remove_stopwords=False)
        
        # Đếm số từ tích cực và tiêu cực
        positive_count = sum(1 for token in tokens if token in self.positive_words)
        negative_count = sum(1 for token in tokens if token in self.negative_words)
        
        # Tổng số từ
        total_words = len(tokens)
        
        # Tính điểm tâm lý từ -1 đến 1
        sentiment_score = 0
        if positive_count + negative_count > 0:
            sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
        
        # Chi tiết tâm lý
        sentiment_details = {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "total_words": total_words,
            "positive_ratio": positive_count / total_words if total_words > 0 else 0,
            "negative_ratio": negative_count / total_words if total_words > 0 else 0
        }
        
        return sentiment_score, sentiment_details
    
    def detect_crypto_terms(self, text: str) -> Dict[str, int]:
        """
        Phát hiện các cụm từ tiền điện tử trong văn bản.
        
        Args:
            text: Văn bản cần phân tích
            
        Returns:
            Từ điển (cụm từ: số lần xuất hiện)
        """
        text = text.lower()
        term_counts = {}
        
        for term in self.crypto_terms:
            count = text.count(term)
            if count > 0:
                term_counts[term] = count
        
        return term_counts
    
    def analyze_text_batch(self, texts: List[str]) -> pd.DataFrame:
        """
        Phân tích tâm lý cho một loạt văn bản.
        
        Args:
            texts: Danh sách văn bản cần phân tích
            
        Returns:
            DataFrame chứa kết quả phân tích
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                # Tính tâm lý
                sentiment_score, sentiment_details = self.calculate_text_sentiment(text)
                
                # Phát hiện cụm từ tiền điện tử
                crypto_terms = self.detect_crypto_terms(text)
                
                # Tạo từ điển kết quả
                result = {
                    "text_id": i,
                    "sentiment_score": sentiment_score,
                    "positive_count": sentiment_details["positive_count"],
                    "negative_count": sentiment_details["negative_count"],
                    "total_words": sentiment_details["total_words"],
                    "positive_ratio": sentiment_details["positive_ratio"],
                    "negative_ratio": sentiment_details["negative_ratio"],
                    "crypto_terms_count": len(crypto_terms),
                    "crypto_terms": ", ".join(list(crypto_terms.keys())[:5])  # Chỉ lấy 5 cụm từ đầu tiên
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Lỗi khi phân tích văn bản #{i}: {str(e)}")
                # Thêm bản ghi lỗi
                results.append({
                    "text_id": i,
                    "sentiment_score": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "total_words": 0,
                    "positive_ratio": 0,
                    "negative_ratio": 0,
                    "crypto_terms_count": 0,
                    "crypto_terms": "",
                    "error": str(e)
                })
        
        return pd.DataFrame(results)


def clean_text(text: str) -> str:
    """
    Làm sạch văn bản.
    
    Args:
        text: Văn bản cần làm sạch
        
    Returns:
        Văn bản đã làm sạch
    """
    processor = TextProcessor()
    return processor.clean_text(text)


def tokenize_text(text: str, remove_stopwords: bool = True) -> List[str]:
    """
    Tách văn bản thành các token.
    
    Args:
        text: Văn bản cần tách
        remove_stopwords: Có loại bỏ stopwords hay không
        
    Returns:
        Danh sách các token
    """
    processor = TextProcessor()
    return processor.tokenize_text(text, remove_stopwords)


def extract_keywords(text: str, top_n: int = 10) -> List[Tuple[str, int]]:
    """
    Trích xuất từ khóa phổ biến từ văn bản.
    
    Args:
        text: Văn bản cần trích xuất từ khóa
        top_n: Số từ khóa cần trích xuất
        
    Returns:
        Danh sách từ khóa và tần suất xuất hiện
    """
    processor = TextProcessor()
    return processor.extract_keywords(text, top_n)


def calculate_text_sentiment(text: str) -> Tuple[float, Dict[str, int]]:
    """
    Tính điểm tâm lý của văn bản dựa trên từ khóa.
    
    Args:
        text: Văn bản cần phân tích
        
    Returns:
        Tuple (điểm tâm lý, chi tiết tâm lý)
    """
    processor = TextProcessor()
    return processor.calculate_text_sentiment(text)


def batch_process_texts(texts: List[str], language: str = "en") -> pd.DataFrame:
    """
    Xử lý hàng loạt văn bản.
    
    Args:
        texts: Danh sách văn bản cần xử lý
        language: Ngôn ngữ của văn bản
        
    Returns:
        DataFrame kết quả
    """
    processor = TextProcessor(language=language)
    return processor.analyze_text_batch(texts)


def extract_sentiment_from_articles(articles: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Trích xuất tâm lý từ danh sách bài viết tin tức.
    
    Args:
        articles: Danh sách bài viết (dict với các trường title, description, content)
        
    Returns:
        DataFrame chứa tâm lý của các bài viết
    """
    processor = TextProcessor()
    results = []
    
    for i, article in enumerate(articles):
        try:
            # Ghép nội dung văn bản
            title = article.get('title', '')
            description = article.get('description', '')
            content = article.get('content', '')
            
            # Ghép văn bản với trọng số lớn hơn cho tiêu đề
            combined_text = f"{title} {title} {description} {content}"
            
            # Tính tâm lý
            sentiment_score, sentiment_details = processor.calculate_text_sentiment(combined_text)
            
            # Phát hiện cụm từ tiền điện tử
            crypto_terms = processor.detect_crypto_terms(combined_text)
            
            # Tạo từ điển kết quả
            result = {
                "article_id": article.get('article_id', i),
                "title": title,
                "source": article.get('source', ''),
                "url": article.get('url', ''),
                "published_at": article.get('published_at', ''),
                "sentiment_score": sentiment_score,
                "positive_count": sentiment_details["positive_count"],
                "negative_count": sentiment_details["negative_count"],
                "positive_ratio": sentiment_details["positive_ratio"],
                "negative_ratio": sentiment_details["negative_ratio"],
                "crypto_terms_count": len(crypto_terms),
                "crypto_terms": ", ".join(list(crypto_terms.keys())[:5])
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Lỗi khi phân tích bài viết #{i}: {str(e)}")
            
            # Thêm bản ghi lỗi
            results.append({
                "article_id": article.get('article_id', i),
                "title": article.get('title', ''),
                "source": article.get('source', ''),
                "url": article.get('url', ''),
                "published_at": article.get('published_at', ''),
                "sentiment_score": 0,
                "positive_count": 0,
                "negative_count": 0,
                "positive_ratio": 0,
                "negative_ratio": 0,
                "crypto_terms_count": 0,
                "crypto_terms": "",
                "error": str(e)
            })
    
    return pd.DataFrame(results)
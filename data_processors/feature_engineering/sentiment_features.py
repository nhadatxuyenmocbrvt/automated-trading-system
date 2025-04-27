"""
Trích xuất đặc trưng tâm lý thị trường từ dữ liệu tin tức và mạng xã hội.
File này cung cấp các lớp và phương thức để xử lý dữ liệu văn bản,
phân tích tình cảm, và tạo các đặc trưng tâm lý thị trường.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import re
import datetime
import logging
from pathlib import Path
import json
import os
import sys

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import setup_logger
from config.constants import ErrorCode

class SentimentFeatureExtractor:
    """
    Lớp chính để trích xuất các đặc trưng tâm lý thị trường từ dữ liệu tin tức và mạng xã hội.
    """
    
    def __init__(
        self,
        sentiment_method: str = 'lexicon',  # 'lexicon', 'textblob', 'vader', 'transformers'
        language: str = 'en',                # 'en', 'vi', 'multi'
        use_pretrained_model: bool = False,  # Sử dụng mô hình pretrained cho phân tích tình cảm
        model_path: Optional[str] = None,    # Đường dẫn đến mô hình nếu có
        custom_lexicon_path: Optional[str] = None,  # Đường dẫn đến từ điển tùy chỉnh
        normalize_scores: bool = True,       # Chuẩn hóa điểm tình cảm về khoảng [-1, 1]
        volume_weighted: bool = True,        # Tính trọng số các điểm theo khối lượng tin tức
        cache_dir: Optional[str] = None      # Thư mục lưu cache
    ):
        """
        Khởi tạo trình trích xuất đặc trưng tâm lý.
        
        Args:
            sentiment_method: Phương pháp phân tích tình cảm ('lexicon', 'textblob', 'vader', 'transformers')
            language: Ngôn ngữ của dữ liệu văn bản ('en', 'vi', 'multi')
            use_pretrained_model: Sử dụng mô hình đã huấn luyện sẵn
            model_path: Đường dẫn đến mô hình (nếu sử dụng mô hình tùy chỉnh)
            custom_lexicon_path: Đường dẫn đến từ điển tùy chỉnh (nếu sử dụng phương pháp lexicon)
            normalize_scores: Chuẩn hóa điểm tình cảm về khoảng [-1, 1]
            volume_weighted: Tính trọng số các điểm tình cảm theo khối lượng tin tức
            cache_dir: Thư mục lưu cache kết quả (để tăng tốc độ xử lý)
        """
        self.logger = setup_logger("sentiment_features")
        
        self.sentiment_method = sentiment_method
        self.language = language
        self.use_pretrained_model = use_pretrained_model
        self.model_path = model_path
        self.custom_lexicon_path = custom_lexicon_path
        self.normalize_scores = normalize_scores
        self.volume_weighted = volume_weighted
        
        # Thiết lập thư mục cache
        if cache_dir:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.cache_dir = None
        
        # Khởi tạo các thành phần phân tích tình cảm dựa vào phương pháp đã chọn
        self._init_sentiment_analyzer()
        
        # Các từ điển dùng để chuẩn hóa entity
        self._init_entity_dictionaries()
        
        self.logger.info(f"Đã khởi tạo SentimentFeatureExtractor với phương pháp {self.sentiment_method}, ngôn ngữ {self.language}")
    
    def _init_sentiment_analyzer(self) -> None:
        """
        Khởi tạo công cụ phân tích tình cảm dựa trên phương pháp đã chọn.
        """
        if self.sentiment_method == 'lexicon':
            self._init_lexicon_analyzer()
        elif self.sentiment_method == 'textblob':
            self._init_textblob_analyzer()
        elif self.sentiment_method == 'vader':
            self._init_vader_analyzer()
        elif self.sentiment_method == 'transformers':
            self._init_transformers_analyzer()
        else:
            self.logger.warning(f"Phương pháp phân tích tình cảm {self.sentiment_method} không được hỗ trợ, sử dụng lexicon làm phương pháp mặc định")
            self.sentiment_method = 'lexicon'
            self._init_lexicon_analyzer()
    
    def _init_lexicon_analyzer(self) -> None:
        """
        Khởi tạo bộ phân tích tình cảm dựa trên từ điển lexicon.
        """
        self.lexicon = {}
        
        # Tải từ điển tùy chỉnh nếu có
        if self.custom_lexicon_path and Path(self.custom_lexicon_path).exists():
            try:
                with open(self.custom_lexicon_path, 'r', encoding='utf-8') as f:
                    self.lexicon = json.load(f)
                self.logger.info(f"Đã tải từ điển tình cảm từ {self.custom_lexicon_path} với {len(self.lexicon)} mục")
            except Exception as e:
                self.logger.error(f"Lỗi khi tải từ điển tùy chỉnh: {e}")
        
        # Nếu không có từ điển tùy chỉnh hoặc tải thất bại, sử dụng từ điển mặc định
        if not self.lexicon:
            # Từ điển tiếng Anh cơ bản
            en_positive = ["bullish", "positive", "increase", "rise", "growing", "growth", "gain", "gains",
                         "rally", "rallies", "boom", "soar", "soars", "soaring", "surge", "surging",
                         "jump", "jumping", "uptrend", "outperform", "outperformed", "strong", "stronger",
                         "strongest", "high", "higher", "highest", "record", "top", "best", "better", "good",
                         "great", "excellent", "extraordinary", "outstanding", "profit", "profitable",
                         "success", "successful", "succeed", "succeeding", "opportunity", "opportunities",
                         "optimistic", "optimism", "confident", "confidence", "support", "supporting",
                         "supported", "buy", "buying", "accumulate", "accumulating", "hold", "holding"]
            
            en_negative = ["bearish", "negative", "decrease", "decreasing", "fall", "falling", "fell",
                         "drop", "dropping", "decline", "declining", "loss", "losses", "crash", "crashes",
                         "collapse", "collapsed", "collapsing", "plunge", "plunges", "plunging", "sink",
                         "sinking", "slide", "sliding", "slump", "slumping", "downtrend", "underperform",
                         "underperformed", "weak", "weaker", "weakest", "low", "lower", "lowest", "bottom",
                         "worse", "worst", "bad", "poor", "terrible", "problem", "problematic", "risks",
                         "risky", "danger", "dangerous", "difficult", "pessimistic", "pessimism", "worried",
                         "worry", "worries", "resistance", "resisting", "sell", "selling", "avoid", "sell"]
            
            # Từ điển tiếng Việt cơ bản (nếu language là 'vi' hoặc 'multi')
            vi_positive = ["tăng", "tích cực", "lạc quan", "khởi sắc", "đột phá", "tốt", "lợi nhuận",
                          "thành công", "vững", "mạnh", "cơ hội", "tiềm năng", "phục hồi", "bứt phá",
                          "đà tăng", "hưng phấn", "khả quan", "triển vọng", "đáng mua", "nên mua", "tốt nhất"]
            
            vi_negative = ["giảm", "tiêu cực", "bi quan", "suy giảm", "thua lỗ", "rủi ro", "nguy hiểm",
                          "bấp bênh", "lo ngại", "đáng ngại", "trượt dốc", "lao dốc", "sụt giảm", "yếu",
                          "kém", "tệ", "đà giảm", "áp lực", "nên bán", "thoái lui", "thất bại", "khó khăn"]
            
            # Khởi tạo từ điển dựa trên ngôn ngữ
            if self.language == 'en':
                for word in en_positive:
                    self.lexicon[word] = 1.0
                for word in en_negative:
                    self.lexicon[word] = -1.0
                    
            elif self.language == 'vi':
                for word in vi_positive:
                    self.lexicon[word] = 1.0
                for word in vi_negative:
                    self.lexicon[word] = -1.0
                    
            elif self.language == 'multi':
                for word in en_positive + vi_positive:
                    self.lexicon[word] = 1.0
                for word in en_negative + vi_negative:
                    self.lexicon[word] = -1.0
            
            self.logger.info(f"Đã tạo từ điển tình cảm mặc định với {len(self.lexicon)} mục")
    
    def _init_textblob_analyzer(self) -> None:
        """
        Khởi tạo bộ phân tích tình cảm dựa trên TextBlob.
        """
        try:
            from textblob import TextBlob
            self.textblob_analyzer = TextBlob
            self.logger.info("Đã khởi tạo bộ phân tích TextBlob")
            
            # Nếu ngôn ngữ không phải tiếng Anh, cần cảnh báo
            if self.language != 'en':
                self.logger.warning("TextBlob hoạt động tốt nhất với tiếng Anh. Kết quả với ngôn ngữ khác có thể không chính xác")
            
        except ImportError:
            self.logger.error("Không thể import TextBlob. Vui lòng cài đặt: pip install textblob")
            self.logger.warning("Chuyển sang sử dụng phương pháp lexicon")
            self.sentiment_method = 'lexicon'
            self._init_lexicon_analyzer()
    
    def _init_vader_analyzer(self) -> None:
        """
        Khởi tạo bộ phân tích tình cảm dựa trên VADER (Valence Aware Dictionary and sEntiment Reasoner).
        """
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader_analyzer = SentimentIntensityAnalyzer()
            self.logger.info("Đã khởi tạo bộ phân tích VADER")
            
            # VADER chỉ hỗ trợ tiếng Anh
            if self.language != 'en':
                self.logger.warning("VADER chỉ hỗ trợ tiếng Anh. Kết quả với ngôn ngữ khác có thể không chính xác")
            
        except ImportError:
            self.logger.error("Không thể import VADER. Vui lòng cài đặt: pip install vaderSentiment")
            self.logger.warning("Chuyển sang sử dụng phương pháp lexicon")
            self.sentiment_method = 'lexicon'
            self._init_lexicon_analyzer()
    
    def _init_transformers_analyzer(self) -> None:
        """
        Khởi tạo bộ phân tích tình cảm dựa trên Transformers (Hugging Face).
        """
        try:
            from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
            
            if self.use_pretrained_model and self.model_path:
                try:
                    # Tải mô hình tùy chỉnh
                    model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
                    tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    self.transformer_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
                    self.logger.info(f"Đã tải mô hình tùy chỉnh từ {self.model_path}")
                except Exception as e:
                    self.logger.error(f"Lỗi khi tải mô hình tùy chỉnh: {e}")
                    self.logger.warning("Chuyển sang sử dụng mô hình mặc định")
                    self.transformer_analyzer = pipeline("sentiment-analysis")
            else:
                # Sử dụng mô hình mặc định dựa trên ngôn ngữ
                if self.language == 'en':
                    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
                elif self.language == 'vi':
                    model_name = "nvt01/bert-base-multilingual-uncased-sentiment"
                else:  # multi
                    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
                
                try:
                    self.transformer_analyzer = pipeline("sentiment-analysis", model=model_name)
                    self.logger.info(f"Đã tải mô hình transformers {model_name}")
                except Exception as e:
                    self.logger.error(f"Lỗi khi tải mô hình transformers: {e}")
                    self.logger.warning("Sử dụng mô hình mặc định")
                    self.transformer_analyzer = pipeline("sentiment-analysis")
                
        except ImportError:
            self.logger.error("Không thể import transformers. Vui lòng cài đặt: pip install transformers")
            self.logger.warning("Chuyển sang sử dụng phương pháp lexicon")
            self.sentiment_method = 'lexicon'
            self._init_lexicon_analyzer()
    
    def _init_entity_dictionaries(self) -> None:
        """
        Khởi tạo các từ điển để chuẩn hóa tên thực thể (ví dụ: Bitcoin, BTC, $BTC -> BTC)
        """
        # Từ điển chuẩn hóa tên các cryptocurrencies
        self.crypto_aliases = {
            "bitcoin": "BTC", "btc": "BTC", "$btc": "BTC", "xbt": "BTC",
            "ethereum": "ETH", "eth": "ETH", "$eth": "ETH",
            "binance coin": "BNB", "bnb": "BNB", "$bnb": "BNB",
            "cardano": "ADA", "ada": "ADA", "$ada": "ADA",
            "solana": "SOL", "sol": "SOL", "$sol": "SOL",
            "xrp": "XRP", "ripple": "XRP", "$xrp": "XRP",
            "polkadot": "DOT", "dot": "DOT", "$dot": "DOT",
            "dogecoin": "DOGE", "doge": "DOGE", "$doge": "DOGE",
            "avalanche": "AVAX", "avax": "AVAX", "$avax": "AVAX",
            "shiba inu": "SHIB", "shib": "SHIB", "$shib": "SHIB",
            # Thêm các mapping khác nếu cần
        }
        
        # Từ điển chuẩn hóa tên thực thể liên quan đến thị trường
        self.market_entities = {
            "crypto market": "CRYPTO_MARKET",
            "cryptocurrency market": "CRYPTO_MARKET",
            "bitcoin market": "BTC_MARKET",
            "ethereum market": "ETH_MARKET",
            "bear market": "BEAR_MARKET",
            "bull market": "BULL_MARKET",
            "crypto winter": "BEAR_MARKET",
            "altcoin season": "ALT_SEASON",
            "defi": "DEFI",
            "nft": "NFT",
            "stablecoin": "STABLECOIN",
            "mining": "MINING",
            "staking": "STAKING",
            "dao": "DAO",
            # Thêm các mapping khác nếu cần
        }
    
    def extract_sentiment_from_text(self, text: str) -> Dict[str, float]:
        """
        Phân tích tình cảm từ một đoạn văn bản.
        
        Args:
            text: Đoạn văn bản cần phân tích
            
        Returns:
            Dictionary chứa điểm tình cảm
        """
        if not text or not isinstance(text, str):
            return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 0.0}
        
        # Tiền xử lý văn bản
        cleaned_text = self._preprocess_text(text)
        
        if self.sentiment_method == 'lexicon':
            return self._analyze_sentiment_lexicon(cleaned_text)
        elif self.sentiment_method == 'textblob':
            return self._analyze_sentiment_textblob(cleaned_text)
        elif self.sentiment_method == 'vader':
            return self._analyze_sentiment_vader(cleaned_text)
        elif self.sentiment_method == 'transformers':
            return self._analyze_sentiment_transformers(cleaned_text)
        else:
            # Fallback nếu có lỗi
            return self._analyze_sentiment_lexicon(cleaned_text)
    
    def _preprocess_text(self, text: str) -> str:
        """
        Tiền xử lý văn bản trước khi phân tích tình cảm.
        
        Args:
            text: Đoạn văn bản cần xử lý
            
        Returns:
            Đoạn văn bản đã xử lý
        """
        if not text:
            return ""
        
        # Chuyển về chữ thường
        text = text.lower()
        
        # Loại bỏ URL
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Loại bỏ các ký tự đặc biệt, giữ lại dấu chấm câu cơ bản
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _analyze_sentiment_lexicon(self, text: str) -> Dict[str, float]:
        """
        Phân tích tình cảm sử dụng phương pháp từ điển lexicon.
        
        Args:
            text: Đoạn văn bản cần phân tích
            
        Returns:
            Dictionary chứa điểm tình cảm
        """
        words = text.split()
        
        # Bỏ qua nếu không có từ nào
        if not words:
            return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        positive_score = 0.0
        negative_score = 0.0
        neutral_count = 0
        
        # Đếm số lượng từ có tình cảm tích cực/tiêu cực
        for word in words:
            if word in self.lexicon:
                score = self.lexicon[word]
                if score > 0:
                    positive_score += score
                elif score < 0:
                    negative_score += abs(score)
            else:
                neutral_count += 1
        
        # Tổng số từ được xét
        total_words = len(words)
        
        # Tính tỷ lệ
        positive_ratio = positive_score / total_words if total_words > 0 else 0
        negative_ratio = negative_score / total_words if total_words > 0 else 0
        neutral_ratio = neutral_count / total_words if total_words > 0 else 1.0
        
        # Tính điểm tổng hợp, khoảng giá trị [-1, 1]
        compound = positive_ratio - negative_ratio
        
        return {
            "compound": compound,
            "positive": positive_ratio,
            "negative": negative_ratio,
            "neutral": neutral_ratio
        }
    
    def _analyze_sentiment_textblob(self, text: str) -> Dict[str, float]:
        """
        Phân tích tình cảm sử dụng TextBlob.
        
        Args:
            text: Đoạn văn bản cần phân tích
            
        Returns:
            Dictionary chứa điểm tình cảm
        """
        blob = self.textblob_analyzer(text)
        
        # TextBlob trả về polarity trong khoảng [-1, 1], và subjectivity trong khoảng [0, 1]
        polarity = blob.sentiment.polarity  # [-1, 1]
        subjectivity = blob.sentiment.subjectivity  # [0, 1]
        
        # Tính toán điểm tích cực/tiêu cực
        positive = max(0, polarity)
        negative = abs(min(0, polarity))
        
        # Tính toán điểm trung tính (càng khách quan thì càng trung tính)
        neutral = 1.0 - subjectivity
        
        return {
            "compound": polarity,
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "subjectivity": subjectivity
        }
    
    def _analyze_sentiment_vader(self, text: str) -> Dict[str, float]:
        """
        Phân tích tình cảm sử dụng VADER.
        
        Args:
            text: Đoạn văn bản cần phân tích
            
        Returns:
            Dictionary chứa điểm tình cảm
        """
        sentiment_scores = self.vader_analyzer.polarity_scores(text)
        return sentiment_scores  # VADER trả về {'neg': x, 'neu': y, 'pos': z, 'compound': c}
    
    def _analyze_sentiment_transformers(self, text: str) -> Dict[str, float]:
        """
        Phân tích tình cảm sử dụng Transformers.
        
        Args:
            text: Đoạn văn bản cần phân tích
            
        Returns:
            Dictionary chứa điểm tình cảm
        """
        try:
            # Giới hạn độ dài văn bản để tránh lỗi
            max_length = 512
            if len(text) > max_length:
                text = text[:max_length]
            
            # Phân tích tình cảm
            result = self.transformer_analyzer(text)
            
            if not result:
                return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}
            
            # Transformers thường trả về [{'label': 'POSITIVE/NEGATIVE', 'score': x}]
            sentiment = result[0]
            label = sentiment['label'].upper()
            score = sentiment['score']
            
            # Chuẩn hóa kết quả
            if 'POSITIVE' in label:
                compound = score
                positive = score
                negative = 0.0
                neutral = 1.0 - score
            elif 'NEGATIVE' in label:
                compound = -score
                positive = 0.0
                negative = score
                neutral = 1.0 - score
            else:  # NEUTRAL
                compound = 0.0
                positive = 0.0
                negative = 0.0
                neutral = 1.0
            
            return {
                "compound": compound,
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "label": label,
                "raw_score": score
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi khi phân tích tình cảm với transformers: {e}")
            return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}
    
    def process_news_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'content',
        timestamp_column: str = 'timestamp',
        entity_column: Optional[str] = 'entity',
        source_column: Optional[str] = 'source',
        batch_size: int = 100
    ) -> pd.DataFrame:
        """
        Xử lý DataFrame tin tức, trích xuất tình cảm, và tạo đặc trưng.
        
        Args:
            df: DataFrame chứa dữ liệu tin tức
            text_column: Tên cột chứa nội dung văn bản
            timestamp_column: Tên cột chứa thời gian đăng tin
            entity_column: Tên cột chứa thực thể (ví dụ: "BTC", "ETH", ...), None nếu không có
            source_column: Tên cột chứa nguồn tin (ví dụ: "Twitter", "Reddit", ...), None nếu không có
            batch_size: Kích thước batch để xử lý (giảm memory footprint)
            
        Returns:
            DataFrame với các đặc trưng tình cảm
        """
        if df.empty:
            self.logger.warning("DataFrame rỗng, không có gì để xử lý")
            return df
        
        # Kiểm tra các cột cần thiết
        required_columns = [text_column, timestamp_column]
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            self.logger.error(f"Thiếu các cột cần thiết: {missing_columns}")
            return df
        
        # Đảm bảo timestamp đúng định dạng
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
            try:
                df[timestamp_column] = pd.to_datetime(df[timestamp_column])
            except Exception as e:
                self.logger.error(f"Không thể chuyển đổi cột timestamp: {e}")
                return df
        
        # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
        result_df = df.copy()
        
        # Tạo các cột tình cảm
        sentiment_columns = ['sentiment_compound', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral']
        for col in sentiment_columns:
            result_df[col] = 0.0
        
        # Thêm cột entity nếu không có
        if entity_column and entity_column not in result_df.columns:
            result_df[entity_column] = None
        
        self.logger.info(f"Bắt đầu xử lý {len(result_df)} dòng tin tức")
        
        # Xử lý theo batch để giảm memory footprint
        total_rows = len(result_df)
        num_batches = (total_rows + batch_size - 1) // batch_size  # Làm tròn lên
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, total_rows)
            
            self.logger.debug(f"Xử lý batch {i+1}/{num_batches} (rows {start_idx}-{end_idx})")
            
            # Xử lý từng dòng trong batch
            for idx in range(start_idx, end_idx):
                row = result_df.iloc[idx]
                
                # Trích xuất thực thể từ nội dung nếu không có sẵn
                if entity_column and (not row[entity_column] or pd.isna(row[entity_column])):
                    detected_entities = self._extract_entities_from_text(row[text_column])
                    if detected_entities:
                        result_df.at[idx, entity_column] = ",".join(detected_entities)
                
                # Phân tích tình cảm
                sentiment_scores = self.extract_sentiment_from_text(row[text_column])
                
                # Cập nhật điểm tình cảm
                result_df.at[idx, 'sentiment_compound'] = sentiment_scores.get('compound', 0.0)
                result_df.at[idx, 'sentiment_positive'] = sentiment_scores.get('positive', 0.0)
                result_df.at[idx, 'sentiment_negative'] = sentiment_scores.get('negative', 0.0)
                result_df.at[idx, 'sentiment_neutral'] = sentiment_scores.get('neutral', 0.0)
        
        self.logger.info(f"Đã xử lý {total_rows} dòng tin tức")
        
        # Tạo đặc trưng tổng hợp nếu có thông tin entity
        if entity_column and result_df[entity_column].notna().any():
            result_df = self._create_entity_features(result_df, entity_column, timestamp_column)
        
        return result_df
    
    def _extract_entities_from_text(self, text: str) -> List[str]:
        """
        Trích xuất các thực thể liên quan từ văn bản.
        
        Args:
            text: Đoạn văn bản cần trích xuất
            
        Returns:
            Danh sách các thực thể
        """
        if not text or not isinstance(text, str):
            return []
        
        # Chuyển về chữ thường để dễ so sánh
        text = text.lower()
        
        detected_entities = set()
        
        # Tìm các crypto alias
        for alias, symbol in self.crypto_aliases.items():
            if alias in text:
                detected_entities.add(symbol)
        
        # Tìm các thực thể thị trường
        for entity, normalized in self.market_entities.items():
            if entity in text:
                detected_entities.add(normalized)
        
        # Tìm các mã tiền theo pattern $XXX hoặc #XXX
        ticker_pattern = r'[$#]([A-Za-z]{2,5})'
        tickers = re.findall(ticker_pattern, text)
        for ticker in tickers:
            detected_entities.add(ticker.upper())
        
        return list(detected_entities)
    
    def _create_entity_features(
        self,
        df: pd.DataFrame,
        entity_column: str,
        timestamp_column: str
    ) -> pd.DataFrame:
        """
        Tạo các đặc trưng tình cảm cho từng thực thể.
        
        Args:
            df: DataFrame đã xử lý với cột entity và sentiment
            entity_column: Tên cột chứa thực thể
            timestamp_column: Tên cột chứa thời gian
            
        Returns:
            DataFrame với các đặc trưng tình cảm theo thực thể
        """
        result_df = df.copy()
        
        # Tách các thực thể thành list nếu chúng được lưu dưới dạng chuỗi ngăn cách bởi dấu phẩy
        if result_df[entity_column].dtype == 'object':
            result_df['entity_list'] = result_df[entity_column].fillna('').apply(
                lambda x: [e.strip() for e in x.split(',')] if isinstance(x, str) else []
            )
        
        # Explode để mỗi dòng chỉ có một thực thể
        exploded_df = result_df.explode('entity_list').reset_index(drop=True)
        exploded_df = exploded_df[exploded_df['entity_list'].notna() & (exploded_df['entity_list'] != '')]
        
        # Đặt tên cho thực thể
        exploded_df = exploded_df.rename(columns={'entity_list': 'normalized_entity'})
        
        # Tính giá trị sentiment trung bình theo ngày cho mỗi thực thể
        exploded_df['date'] = exploded_df[timestamp_column].dt.date
        
        # Tạo tính năng sentiment theo thực thể và ngày
        entity_sentiment = exploded_df.groupby(['normalized_entity', 'date']).agg({
            'sentiment_compound': ['mean', 'count', 'std'],
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean'
        }).reset_index()
        
        # Làm phẳng cấu trúc đa cấp của columns
        entity_sentiment.columns = ['normalized_entity', 'date', 
                                   'sentiment_compound_mean', 'sentiment_count', 'sentiment_compound_std',
                                   'sentiment_positive_mean', 'sentiment_negative_mean', 'sentiment_neutral_mean']
        
        # Chuyển date thành datetime
        entity_sentiment['date'] = pd.to_datetime(entity_sentiment['date'])
        
        # Làm trơn giá trị sentiment theo thời gian nếu có đủ dữ liệu
        smoothed_entity_sentiment = []
        
        for entity in entity_sentiment['normalized_entity'].unique():
            entity_data = entity_sentiment[entity_sentiment['normalized_entity'] == entity].sort_values('date')
            
            if len(entity_data) >= 3:  # Cần ít nhất 3 điểm để làm trơn
                # Áp dụng rolling window
                entity_data['sentiment_compound_mean_smooth'] = entity_data['sentiment_compound_mean'].rolling(window=3, min_periods=1).mean()
                entity_data['sentiment_positive_mean_smooth'] = entity_data['sentiment_positive_mean'].rolling(window=3, min_periods=1).mean()
                entity_data['sentiment_negative_mean_smooth'] = entity_data['sentiment_negative_mean'].rolling(window=3, min_periods=1).mean()
            else:
                # Không đủ dữ liệu, giữ nguyên
                entity_data['sentiment_compound_mean_smooth'] = entity_data['sentiment_compound_mean']
                entity_data['sentiment_positive_mean_smooth'] = entity_data['sentiment_positive_mean']
                entity_data['sentiment_negative_mean_smooth'] = entity_data['sentiment_negative_mean']
            
            smoothed_entity_sentiment.append(entity_data)
        
        # Gộp lại
        if smoothed_entity_sentiment:
            entity_sentiment_final = pd.concat(smoothed_entity_sentiment)
        else:
            entity_sentiment_final = entity_sentiment.copy()
            entity_sentiment_final['sentiment_compound_mean_smooth'] = entity_sentiment_final['sentiment_compound_mean']
            entity_sentiment_final['sentiment_positive_mean_smooth'] = entity_sentiment_final['sentiment_positive_mean']
            entity_sentiment_final['sentiment_negative_mean_smooth'] = entity_sentiment_final['sentiment_negative_mean']
        
        # Tính momentum chỉ báo sentiment thay đổi
        entity_sentiment_with_momentum = []
        
        for entity in entity_sentiment_final['normalized_entity'].unique():
            entity_data = entity_sentiment_final[entity_sentiment_final['normalized_entity'] == entity].sort_values('date')
            
            if len(entity_data) >= 2:
                # Tính momentum (thay đổi so với ngày trước)
                entity_data['sentiment_momentum'] = entity_data['sentiment_compound_mean_smooth'].diff()
                # Điền giá trị NA đầu tiên bằng 0
                entity_data['sentiment_momentum'] = entity_data['sentiment_momentum'].fillna(0)
            else:
                entity_data['sentiment_momentum'] = 0
            
            entity_sentiment_with_momentum.append(entity_data)
        
        # Gộp lại
        if entity_sentiment_with_momentum:
            entity_features = pd.concat(entity_sentiment_with_momentum)
        else:
            entity_features = entity_sentiment_final.copy()
            entity_features['sentiment_momentum'] = 0
        
        self.logger.info(f"Đã tạo đặc trưng tình cảm cho {len(entity_features['normalized_entity'].unique())} thực thể")
        
        return entity_features
    
    def create_time_series_features(
        self,
        sentiment_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
        entity: Optional[str] = None,
        resample_freq: str = 'D',  # 'D' for daily, 'H' for hourly, etc.
        window_sizes: List[int] = [1, 3, 7, 14],
        normalize: bool = True
    ) -> pd.DataFrame:
        """
        Tạo các đặc trưng chuỗi thời gian từ dữ liệu tình cảm và tùy chọn gộp với dữ liệu giá.
        
        Args:
            sentiment_df: DataFrame chứa dữ liệu tình cảm theo thời gian
            price_df: DataFrame chứa dữ liệu giá (tùy chọn)
            entity: Lọc theo một thực thể cụ thể (tùy chọn)
            resample_freq: Tần suất resampling ('D', 'H', ...)
            window_sizes: Kích thước cửa sổ để tính các đặc trưng
            normalize: Chuẩn hóa đặc trưng tình cảm
            
        Returns:
            DataFrame với các đặc trưng chuỗi thời gian
        """
        if sentiment_df.empty:
            self.logger.warning("DataFrame tình cảm rỗng, không có gì để xử lý")
            return pd.DataFrame()
        
        # Lọc theo entity nếu được chỉ định
        if entity and 'normalized_entity' in sentiment_df.columns:
            filtered_df = sentiment_df[sentiment_df['normalized_entity'] == entity].copy()
            if filtered_df.empty:
                self.logger.warning(f"Không có dữ liệu cho entity {entity}")
                return pd.DataFrame()
        else:
            filtered_df = sentiment_df.copy()
        
        # Kiểm tra cột thời gian
        if 'date' not in filtered_df.columns:
            self.logger.error("Không tìm thấy cột 'date' trong DataFrame")
            return pd.DataFrame()
        
        # Đảm bảo cột date là datetime
        if not pd.api.types.is_datetime64_any_dtype(filtered_df['date']):
            try:
                filtered_df['date'] = pd.to_datetime(filtered_df['date'])
            except Exception as e:
                self.logger.error(f"Không thể chuyển đổi cột date: {e}")
                return pd.DataFrame()
        
        # Đặt date làm index
        filtered_df = filtered_df.set_index('date')
        
        # Resampling theo tần suất được chỉ định
        if resample_freq:
            # Tính giá trị trung bình cho mỗi khoảng thời gian
            resampled_df = filtered_df.resample(resample_freq).agg({
                'sentiment_compound_mean': 'mean',
                'sentiment_compound_mean_smooth': 'mean',
                'sentiment_positive_mean': 'mean',
                'sentiment_negative_mean': 'mean',
                'sentiment_count': 'sum',
                'sentiment_momentum': 'mean'
            })
        else:
            resampled_df = filtered_df
        
        # Tạo các đặc trưng chuỗi thời gian
        for window in window_sizes:
            # Rolling mean
            resampled_df[f'sentiment_mean_{window}d'] = resampled_df['sentiment_compound_mean'].rolling(window=window).mean()
            
            # Rolling std
            resampled_df[f'sentiment_std_{window}d'] = resampled_df['sentiment_compound_mean'].rolling(window=window).std()
            
            # Rolling momentum (rate of change)
            resampled_df[f'sentiment_roc_{window}d'] = resampled_df['sentiment_compound_mean'].pct_change(periods=window)
            
            # Rolling volume (count)
            resampled_df[f'sentiment_volume_{window}d'] = resampled_df['sentiment_count'].rolling(window=window).sum()
            
            # Rolling sentiment momentum
            resampled_df[f'sentiment_momentum_{window}d'] = resampled_df['sentiment_momentum'].rolling(window=window).mean()
        
        # Thêm các đặc trưng tương đối
        if len(window_sizes) >= 2:
            # Sorted windows
            sorted_windows = sorted(window_sizes)
            
            # Tính chênh lệch giữa các cửa sổ (tạo tín hiệu giao cắt)
            for i in range(len(sorted_windows)-1):
                short_window = sorted_windows[i]
                long_window = sorted_windows[i+1]
                
                # Chênh lệch giữa giá trị trung bình ngắn hạn và dài hạn
                resampled_df[f'sentiment_diff_{short_window}_{long_window}d'] = (
                    resampled_df[f'sentiment_mean_{short_window}d'] - resampled_df[f'sentiment_mean_{long_window}d']
                )
        
        # Thêm các đặc trưng nâng cao
        # MACD-like indicator for sentiment
        if len(window_sizes) >= 2:
            short_window = min(window_sizes)
            long_window = max(window_sizes)
            signal_window = min(9, short_window)  # 9 là giá trị thông thường cho signal line
            
            # MACD line
            resampled_df['sentiment_macd'] = (
                resampled_df[f'sentiment_mean_{short_window}d'] - resampled_df[f'sentiment_mean_{long_window}d']
            )
            
            # Signal line
            resampled_df['sentiment_signal'] = resampled_df['sentiment_macd'].rolling(window=signal_window).mean()
            
            # MACD histogram
            resampled_df['sentiment_macd_hist'] = resampled_df['sentiment_macd'] - resampled_df['sentiment_signal']
        
        # Chuẩn hóa nếu cần
        if normalize:
            # Lấy các cột số
            numeric_cols = resampled_df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Loại bỏ cột count khỏi việc chuẩn hóa
            numeric_cols = [col for col in numeric_cols if 'count' not in col and 'volume' not in col]
            
            for col in numeric_cols:
                # Min-max normalization
                min_val = resampled_df[col].min()
                max_val = resampled_df[col].max()
                
                if max_val > min_val:  # Tránh chia cho 0
                    resampled_df[col] = (resampled_df[col] - min_val) / (max_val - min_val)
                    
                    # Scale về khoảng [-1, 1] nếu có giá trị âm
                    if min_val < 0:
                        resampled_df[col] = resampled_df[col] * 2 - 1
        
        # Reset index để đưa date trở lại thành cột
        result_df = resampled_df.reset_index()
        
        # Gộp với dữ liệu giá nếu có
        if price_df is not None and not price_df.empty:
            # Kiểm tra cột timestamp trong price_df
            if 'timestamp' not in price_df.columns and 'date' not in price_df.columns:
                self.logger.error("Không tìm thấy cột timestamp hoặc date trong DataFrame giá")
                return result_df
            
            # Chuẩn bị dữ liệu giá
            price_data = price_df.copy()
            
            # Đảm bảo có cột date
            timestamp_col = 'timestamp' if 'timestamp' in price_data.columns else 'date'
            if not pd.api.types.is_datetime64_any_dtype(price_data[timestamp_col]):
                try:
                    price_data[timestamp_col] = pd.to_datetime(price_data[timestamp_col])
                except Exception as e:
                    self.logger.error(f"Không thể chuyển đổi cột timestamp trong dữ liệu giá: {e}")
                    return result_df
            
            # Đổi tên để thống nhất
            price_data = price_data.rename(columns={timestamp_col: 'date'})
            
            # Merge dữ liệu
            result_df = result_df.merge(price_data, on='date', how='left')
            
            self.logger.info(f"Đã gộp dữ liệu tình cảm với dữ liệu giá, kết quả có {len(result_df)} dòng")
        
        return result_df
    
    def calculate_correlation_with_price(
        self,
        sentiment_df: pd.DataFrame,
        price_df: pd.DataFrame,
        price_column: str = 'close',
        entity_column: str = 'normalized_entity',
        date_column: str = 'date',
        sentiment_columns: Optional[List[str]] = None,
        lag_periods: List[int] = [0, 1, 2, 3, 5, 7]
    ) -> pd.DataFrame:
        """
        Tính toán tương quan giữa tình cảm và giá.
        
        Args:
            sentiment_df: DataFrame chứa dữ liệu tình cảm
            price_df: DataFrame chứa dữ liệu giá
            price_column: Tên cột giá trong price_df
            entity_column: Tên cột chứa thực thể trong sentiment_df
            date_column: Tên cột ngày trong cả hai DataFrame
            sentiment_columns: Danh sách các cột tình cảm cần phân tích (None sẽ sử dụng các cột mặc định)
            lag_periods: Các khoảng thời gian trễ để phân tích
            
        Returns:
            DataFrame với kết quả phân tích tương quan
        """
        if sentiment_df.empty or price_df.empty:
            self.logger.warning("DataFrame tình cảm hoặc giá rỗng, không thể tính tương quan")
            return pd.DataFrame()
        
        # Kiểm tra các cột cần thiết
        if entity_column not in sentiment_df.columns:
            self.logger.error(f"Không tìm thấy cột {entity_column} trong DataFrame tình cảm")
            return pd.DataFrame()
        
        if date_column not in sentiment_df.columns or date_column not in price_df.columns:
            self.logger.error(f"Không tìm thấy cột {date_column} trong một trong hai DataFrame")
            return pd.DataFrame()
        
        if price_column not in price_df.columns:
            self.logger.error(f"Không tìm thấy cột {price_column} trong DataFrame giá")
            return pd.DataFrame()
        
        # Sử dụng các cột tình cảm mặc định nếu không được chỉ định
        if sentiment_columns is None:
            sentiment_columns = [
                'sentiment_compound_mean', 'sentiment_positive_mean', 
                'sentiment_negative_mean', 'sentiment_momentum'
            ]
            
            # Thêm các cột phái sinh nếu có
            for col in sentiment_df.columns:
                if 'sentiment_mean_' in col or 'sentiment_macd' in col:
                    sentiment_columns.append(col)
        
        # Lọc các cột tình cảm có thực trong DataFrame
        sentiment_columns = [col for col in sentiment_columns if col in sentiment_df.columns]
        
        if not sentiment_columns:
            self.logger.error("Không tìm thấy cột tình cảm hợp lệ để phân tích")
            return pd.DataFrame()
        
        # Đảm bảo cột ngày là datetime
        for df in [sentiment_df, price_df]:
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                try:
                    df[date_column] = pd.to_datetime(df[date_column])
                except Exception as e:
                    self.logger.error(f"Không thể chuyển đổi cột {date_column}: {e}")
                    return pd.DataFrame()
        
        # Kết quả sẽ là DataFrame với các tương quan theo entity và lag period
        result_data = []
        
        # Phân tích cho từng thực thể
        for entity in sentiment_df[entity_column].unique():
            entity_sentiment = sentiment_df[sentiment_df[entity_column] == entity].copy()
            
            # Sắp xếp theo ngày
            entity_sentiment = entity_sentiment.sort_values(date_column)
            price_data = price_df.sort_values(date_column)
            
            # Chuẩn bị dữ liệu giá (chỉ lấy 2 cột cần thiết)
            price_series = price_data[[date_column, price_column]].copy()
            
            # Tính return của giá
            price_series['price_return'] = price_series[price_column].pct_change()
            
            # Ghép dữ liệu tình cảm với giá
            merged_data = entity_sentiment.merge(price_series, on=date_column, how='inner')
            
            if len(merged_data) < 5:  # Cần ít nhất 5 điểm để phân tích
                self.logger.warning(f"Không đủ dữ liệu cho entity {entity} sau khi merge, bỏ qua")
                continue
            
            # Tính tương quan với các độ trễ khác nhau
            for lag in lag_periods:
                # Tạo các cột giá và return với độ trễ
                if lag > 0:
                    merged_data[f'price_{lag}d'] = merged_data[price_column].shift(-lag)
                    merged_data[f'return_{lag}d'] = merged_data['price_return'].shift(-lag)
                else:
                    merged_data[f'price_{lag}d'] = merged_data[price_column]
                    merged_data[f'return_{lag}d'] = merged_data['price_return']
                
                # Tính tương quan giữa tình cảm và giá
                for col in sentiment_columns:
                    if col not in merged_data.columns:
                        continue
                    
                    # Tương quan với giá
                    price_corr = merged_data[[col, f'price_{lag}d']].corr().iloc[0, 1]
                    
                    # Tương quan với return
                    return_corr = merged_data[[col, f'return_{lag}d']].corr().iloc[0, 1]
                    
                    # Thêm vào kết quả
                    result_data.append({
                        'entity': entity,
                        'sentiment_feature': col,
                        'lag_period': lag,
                        'price_correlation': price_corr,
                        'return_correlation': return_corr,
                        'data_points': len(merged_data.dropna())
                    })
        
        # Tạo DataFrame kết quả
        if not result_data:
            self.logger.warning("Không có kết quả tương quan nào được tính toán")
            return pd.DataFrame()
        
        result_df = pd.DataFrame(result_data)
        
        # Sắp xếp kết quả
        result_df = result_df.sort_values(['entity', 'sentiment_feature', 'lag_period'])
        
        self.logger.info(f"Đã tính toán {len(result_df)} tương quan giữa tình cảm và giá")
        
        return result_df
    
    def save_to_cache(self, data: Any, cache_key: str) -> bool:
        """
        Lưu dữ liệu vào cache.
        
        Args:
            data: Dữ liệu cần lưu
            cache_key: Khóa cache
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        if not self.cache_dir:
            return False
        
        try:
            cache_path = self.cache_dir / f"{cache_key}.pkl"
            pd.to_pickle(data, cache_path)
            self.logger.debug(f"Đã lưu dữ liệu vào cache {cache_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu cache: {e}")
            return False
    
    def load_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Tải dữ liệu từ cache.
        
        Args:
            cache_key: Khóa cache
            
        Returns:
            Dữ liệu đã lưu hoặc None nếu không tìm thấy
        """
        if not self.cache_dir:
            return None
        
        cache_path = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_path.exists():
            return None
        
        try:
            data = pd.read_pickle(cache_path)
            self.logger.debug(f"Đã tải dữ liệu từ cache {cache_path}")
            return data
        except Exception as e:
            self.logger.error(f"Lỗi khi tải cache: {e}")
            return None
    
    def clear_cache(self, cache_key: Optional[str] = None) -> None:
        """
        Xóa dữ liệu từ cache.
        
        Args:
            cache_key: Khóa cache cụ thể (None để xóa tất cả)
        """
        if not self.cache_dir:
            return
        
        try:
            if cache_key:
                cache_path = self.cache_dir / f"{cache_key}.pkl"
                if cache_path.exists():
                    cache_path.unlink()
                    self.logger.debug(f"Đã xóa cache {cache_path}")
            else:
                # Xóa tất cả các file .pkl trong thư mục cache
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                self.logger.debug(f"Đã xóa tất cả cache trong {self.cache_dir}")
        except Exception as e:
            self.logger.error(f"Lỗi khi xóa cache: {e}")


# Factory để tạo SentimentFeatureExtractor với các cấu hình khác nhau
class SentimentFeatureExtractorFactory:
    """
    Factory để tạo các SentimentFeatureExtractor với cấu hình khác nhau.
    """
    
    @staticmethod
    def create_basic_extractor(language: str = 'en') -> SentimentFeatureExtractor:
        """
        Tạo bộ trích xuất cơ bản sử dụng phương pháp lexicon.
        
        Args:
            language: Ngôn ngữ ('en', 'vi', 'multi')
            
        Returns:
            SentimentFeatureExtractor đã cấu hình
        """
        return SentimentFeatureExtractor(
            sentiment_method='lexicon',
            language=language,
            normalize_scores=True
        )
    
    @staticmethod
    def create_advanced_extractor(language: str = 'en') -> SentimentFeatureExtractor:
        """
        Tạo bộ trích xuất nâng cao sử dụng VADER (cho tiếng Anh) hoặc TextBlob (cho đa ngôn ngữ).
        
        Args:
            language: Ngôn ngữ ('en', 'vi', 'multi')
            
        Returns:
            SentimentFeatureExtractor đã cấu hình
        """
        if language == 'en':
            return SentimentFeatureExtractor(
                sentiment_method='vader',
                language=language,
                normalize_scores=True
            )
        else:
            return SentimentFeatureExtractor(
                sentiment_method='textblob',
                language=language,
                normalize_scores=True
            )
    
    @staticmethod
    def create_transformer_extractor(
        language: str = 'en',
        model_path: Optional[str] = None
    ) -> SentimentFeatureExtractor:
        """
        Tạo bộ trích xuất sử dụng mô hình transformers.
        
        Args:
            language: Ngôn ngữ ('en', 'vi', 'multi')
            model_path: Đường dẫn đến mô hình tùy chỉnh
            
        Returns:
            SentimentFeatureExtractor đã cấu hình
        """
        return SentimentFeatureExtractor(
            sentiment_method='transformers',
            language=language,
            use_pretrained_model=True,
            model_path=model_path,
            normalize_scores=True,
            cache_dir='./cache/sentiment'
        )
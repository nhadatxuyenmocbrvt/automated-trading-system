"""
Thu thập dữ liệu tâm lý thị trường.
File này cung cấp các lớp và phương thức để thu thập dữ liệu tâm lý
từ các nguồn khác nhau như mạng xã hội, chỉ số fear & greed, và các chỉ báo tâm lý khác.
"""

import os
import re
import json
import time
import logging
import asyncio
import aiohttp
import datetime
import platform
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, TYPE_CHECKING
from urllib.parse import urlencode
from dataclasses import dataclass
import requests
from requests.exceptions import RequestException
import hashlib

# Import matplotlib có điều kiện để hỗ trợ type hints
if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

# Import các module nội bộ
from config.env import get_env
from config.logging_config import get_logger

# Tạo logger cho module
logger = get_logger("sentiment_collector")

# Kiểm tra nếu đang chạy trên Windows
IS_WINDOWS = platform.system() == 'Windows'

# Số lần retry cho HTTP requests
MAX_RETRIES = 3
# Thời gian chờ giữa các lần retry (giây)
RETRY_DELAY = 2

@dataclass
class SentimentData:
    """Lớp lưu trữ dữ liệu tâm lý thị trường."""
    
    value: float  # Giá trị tâm lý (ví dụ: -1 đến 1, 0 đến 100)
    label: str    # Nhãn (ví dụ: "Fear", "Greed", "Neutral")
    source: str   # Nguồn dữ liệu
    timestamp: datetime.datetime  # Thời điểm thu thập
    asset: Optional[str] = None  # Tài sản liên quan (nếu có)
    timeframe: Optional[str] = None  # Khung thời gian (nếu có)
    metadata: Optional[Dict[str, Any]] = None  # Metadata bổ sung
    
    def to_dict(self) -> Dict[str, Any]:
        """Chuyển đổi thành từ điển."""
        return {
            'value': self.value,
            'label': self.label,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'asset': self.asset,
            'timeframe': self.timeframe,
            'metadata': self.metadata or {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SentimentData':
        """Tạo đối tượng từ từ điển."""
        timestamp = data['timestamp']
        if isinstance(timestamp, str):
            timestamp = datetime.datetime.fromisoformat(timestamp)
        
        return cls(
            value=data['value'],
            label=data['label'],
            source=data['source'],
            timestamp=timestamp,
            asset=data.get('asset'),
            timeframe=data.get('timeframe'),
            metadata=data.get('metadata', {})
        )


class SentimentSource:
    """Lớp cơ sở cho các nguồn dữ liệu tâm lý."""
    
    def __init__(self, name: str, description: str = ""):
        """
        Khởi tạo nguồn dữ liệu tâm lý.
        
        Args:
            name: Tên nguồn dữ liệu
            description: Mô tả nguồn dữ liệu
        """
        self.name = name
        self.description = description
        self.session = None
    
    def get_browser_headers(self) -> Dict[str, str]:
        """
        Tạo headers giả lập trình duyệt đầy đủ.
        
        Returns:
            Headers HTTP với thông tin trình duyệt đầy đủ
        """
        return {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml,application/json;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,vi;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Cache-Control": "max-age=0",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "same-origin",
            "Sec-Fetch-User": "?1",
            "DNT": "1"
        }
    
    async def initialize_session(self) -> None:
        """Khởi tạo phiên HTTP."""
        if not IS_WINDOWS and (self.session is None or self.session.closed):
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close_session(self) -> None:
        """Đóng phiên HTTP."""
        if not IS_WINDOWS and self.session and not self.session.closed:
            await self.session.close()
    
    async def fetch_url(self, url: str, params: Optional[Dict[str, Any]] = None, max_retries: int = MAX_RETRIES) -> Optional[str]:
        """
        Tải nội dung từ URL với cơ chế retry.
        
        Args:
            url: URL cần tải nội dung
            params: Tham số truy vấn (tùy chọn)
            max_retries: Số lần thử lại tối đa nếu request thất bại
        
        Returns:
            Nội dung trang web hoặc None nếu có lỗi
        """
        headers = self.get_browser_headers()
        
        # Đếm số lần retry
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Sử dụng requests đồng bộ trên Windows
                if IS_WINDOWS:
                    response = requests.get(url, headers=headers, params=params, timeout=30)
                    if response.status_code == 200:
                        return response.text
                    elif response.status_code in [403, 404, 429, 500, 502, 503, 504]:
                        # Ghi log lỗi HTTP
                        logger.warning(f"Không tải được URL {url}, mã trạng thái: {response.status_code}")
                        
                        # Nếu là lỗi 404 (không tìm thấy), không cần retry
                        if response.status_code == 404:
                            return None
                            
                        # Đối với các lỗi khác, thử lại
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.info(f"Đang thử lại ({retry_count}/{max_retries}) sau {RETRY_DELAY}s...")
                            time.sleep(RETRY_DELAY)
                            continue
                        return None
                    else:
                        # Các mã trạng thái khác xử lý như lỗi
                        logger.warning(f"Mã trạng thái không xác định: {response.status_code} khi tải {url}")
                        return None
                else:
                    # Sử dụng aiohttp cho các hệ điều hành khác
                    await self.initialize_session()
                    
                    async with self.session.get(url, headers=headers, params=params) as response:
                        if response.status == 200:
                            return await response.text()
                        elif response.status in [403, 404, 429, 500, 502, 503, 504]:
                            # Ghi log lỗi HTTP
                            logger.warning(f"Không tải được URL {url}, mã trạng thái: {response.status}")
                            
                            # Nếu là lỗi 404 (không tìm thấy), không cần retry
                            if response.status == 404:
                                return None
                                
                            # Đối với các lỗi khác, thử lại
                            retry_count += 1
                            if retry_count < max_retries:
                                logger.info(f"Đang thử lại ({retry_count}/{max_retries}) sau {RETRY_DELAY}s...")
                                await asyncio.sleep(RETRY_DELAY)
                                continue
                            return None
                        else:
                            # Các mã trạng thái khác xử lý như lỗi
                            logger.warning(f"Mã trạng thái không xác định: {response.status} khi tải {url}")
                            return None
            except Exception as e:
                logger.error(f"Lỗi khi tải URL {url}: {str(e)}")
                
                # Thử lại nếu có lỗi mạng
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Đang thử lại ({retry_count}/{max_retries}) sau {RETRY_DELAY}s...")
                    if IS_WINDOWS:
                        time.sleep(RETRY_DELAY)
                    else:
                        await asyncio.sleep(RETRY_DELAY)
                    continue
                
                return None
        
        # Nếu tất cả các lần retry đều thất bại
        return None
    
    async def fetch_json(self, url: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """
        Tải dữ liệu JSON từ URL.
        
        Args:
            url: URL cần tải dữ liệu
            params: Tham số truy vấn (tùy chọn)
        
        Returns:
            Dữ liệu JSON hoặc None nếu có lỗi
        """
        if IS_WINDOWS:
            try:
                headers = self.get_browser_headers()
                headers["Accept"] = "application/json"
                
                response = requests.get(url, headers=headers, params=params, timeout=30)
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.warning(f"Không tải được JSON từ {url}, mã trạng thái: {response.status_code}")
                    return None
            except Exception as e:
                logger.error(f"Lỗi khi tải JSON từ {url}: {str(e)}")
                return None
        else:
            content = await self.fetch_url(url, params)
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Lỗi khi phân tích JSON từ {url}: {str(e)}")
            return None
    
    async def get_sentiment(self, **kwargs) -> List[SentimentData]:
        """
        Lấy dữ liệu tâm lý từ nguồn.
        
        Args:
            **kwargs: Các tham số tùy chọn
        
        Returns:
            Danh sách dữ liệu tâm lý
        """
        # Phương thức cần được ghi đè trong lớp con
        raise NotImplementedError("Phương thức này phải được triển khai trong lớp con")


class FearAndGreedIndex(SentimentSource):
    """Lớp thu thập dữ liệu từ chỉ số Fear & Greed."""
    
    def __init__(self):
        """Khởi tạo nguồn dữ liệu Fear & Greed Index."""
        super().__init__(
            name="Fear and Greed Index",
            description="Chỉ số đo lường tâm lý thị trường Bitcoin từ cực kỳ sợ hãi đến cực kỳ tham lam"
        )
        self.api_url = "https://api.alternative.me/fng/"
        
        # Mapping các nhãn
        self.value_to_label = {
            range(0, 25): "Extreme Fear",
            range(25, 47): "Fear",
            range(47, 55): "Neutral",
            range(55, 75): "Greed",
            range(75, 101): "Extreme Greed"
        }
    
    def _get_label(self, value: int) -> str:
        """
        Ánh xạ giá trị thành nhãn.
        
        Args:
            value: Giá trị chỉ số (0-100)
            
        Returns:
            Nhãn tương ứng
        """
        for value_range, label in self.value_to_label.items():
            if value in value_range:
                return label
        return "Unknown"
    
    async def get_sentiment(self, limit: int = 0, format_time: str = "humanized") -> List[SentimentData]:
        """
        Lấy dữ liệu chỉ số Fear & Greed.
        
        Args:
            limit: Số lượng bản ghi lịch sử (0 = hiện tại, 1+ = lịch sử)
            format_time: Định dạng thời gian ('humanized' hoặc 'unix')
            
        Returns:
            Danh sách dữ liệu tâm lý
        """
        params = {
            'limit': limit,
            'format': format_time
        }
        
        try:
            data = await self.fetch_json(self.api_url, params)
            if not data or 'data' not in data:
                logger.error("Không thể lấy dữ liệu Fear & Greed Index")
                return []
            
            results = []
            for item in data['data']:
                try:
                    value = int(item['value'])
                    label = self._get_label(value)
                    
                    # Lấy timestamp
                    if 'timestamp' in item:
                        timestamp = datetime.datetime.fromtimestamp(int(item['timestamp']))
                    else:
                        timestamp = datetime.datetime.now()
                    
                    # Lấy timeframe
                    timeframe = item.get('time_until_update', '')
                    
                    # Tạo đối tượng SentimentData
                    sentiment = SentimentData(
                        value=value,
                        label=label,
                        source=self.name,
                        timestamp=timestamp,
                        asset="BTC",  # Mặc định là BTC vì chỉ số này chỉ theo dõi Bitcoin
                        timeframe="1d",
                        metadata={
                            'classification': item.get('value_classification', ''),
                            'time_until_update': timeframe
                        }
                    )
                    
                    results.append(sentiment)
                    
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý dữ liệu Fear & Greed: {str(e)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu Fear & Greed Index: {str(e)}")
            return []


class TwitterSentimentScraper(SentimentSource):
    """Lớp thu thập dữ liệu tâm lý từ Twitter/X."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Khởi tạo scraper Twitter Sentiment.
        
        Args:
            api_key: API key cho Twitter (tùy chọn)
        """
        super().__init__(
            name="Twitter Sentiment",
            description="Phân tích tâm lý thị trường từ các tweet về tiền điện tử"
        )
        self.api_key = api_key or get_env("TWITTER_API_KEY", "")
        self.api_secret = get_env("TWITTER_API_SECRET", "")
        self.bearer_token = get_env("TWITTER_BEARER_TOKEN", "")
        
        # API v2 endpoint
        self.search_url = "https://api.twitter.com/2/tweets/search/recent"
        
        # Danh sách các hashtag cần theo dõi
        self.default_hashtags = [
            "#bitcoin", "#btc", "#ethereum", "#eth", 
            "#crypto", "#cryptocurrency", "#altcoin"
        ]
    
    async def search_tweets(self, query: str, limit: int = 100) -> Optional[Dict[str, Any]]:
        """
        Tìm kiếm tweets bằng Twitter API v2.
        
        Args:
            query: Truy vấn tìm kiếm
            limit: Số lượng tweet tối đa
            
        Returns:
            Kết quả tìm kiếm hoặc None nếu lỗi
        """
        if not self.bearer_token:
            logger.error("Không tìm thấy Bearer Token cho Twitter API")
            return None
        
        headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json"
        }
        
        params = {
            "query": query,
            "max_results": min(limit, 100),  # API giới hạn 100 kết quả mỗi request
            "tweet.fields": "created_at,public_metrics,lang",
            "expansions": "author_id",
            "user.fields": "name,username,verified,public_metrics"
        }
        
        # Sử dụng phương thức khác nhau tùy thuộc vào hệ điều hành
        if IS_WINDOWS:
            try:
                response = requests.get(self.search_url, headers=headers, params=params)
                
                if response.status_code != 200:
                    logger.error(f"Lỗi Twitter API: {response.status_code}, {response.text}")
                    return None
                
                return response.json()
                
            except Exception as e:
                logger.error(f"Lỗi khi tìm kiếm tweets: {str(e)}")
                return None
        else:
            try:
                # Sử dụng aiohttp cho các hệ điều hành khác
                await self.initialize_session()
                
                async with self.session.get(self.search_url, headers=headers, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Lỗi Twitter API: {response.status}, {await response.text()}")
                        return None
                    
                    return await response.json()
                    
            except Exception as e:
                logger.error(f"Lỗi khi tìm kiếm tweets: {str(e)}")
                return None
    
    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """
        Phân tích tâm lý từ văn bản tweet.
        Đây là phân tích đơn giản dựa trên từ điển, cho mục đích minh họa.
        Trong thực tế, nên sử dụng mô hình NLP để phân tích chính xác hơn.
        
        Args:
            text: Nội dung tweet
            
        Returns:
            Tuple (điểm tâm lý, nhãn)
        """
        # Từ điển đơn giản các từ tích cực/tiêu cực
        positive_words = [
            "bull", "bullish", "buy", "long", "moon", "mooning", "pump", "pumping",
            "gain", "gains", "profit", "profits", "up", "uptrend", "higher", "rise",
            "rising", "good", "great", "excellent", "amazing", "positive", "awesome",
            "perfect", "opportunity", "potential", "confident", "confidence", "strong",
            "strength", "winning", "win", "hodl", "hold", "green", "success", "successful"
        ]
        
        negative_words = [
            "bear", "bearish", "sell", "short", "dump", "dumping", "crash", "crashing",
            "loss", "losses", "down", "downtrend", "lower", "fall", "falling", "bad",
            "terrible", "poor", "negative", "awful", "worst", "fear", "fearful", "weak",
            "weakness", "fail", "failing", "failure", "red", "risk", "risky", "danger",
            "dangerous", "trouble", "worried", "worry", "scam", "fraud", "bubble", "correction"
        ]
        
        # Chuyển văn bản thành chữ thường
        text_lower = text.lower()
        
        # Đếm số từ tích cực và tiêu cực
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Tính điểm tâm lý từ -1 (rất tiêu cực) đến 1 (rất tích cực)
        total_count = positive_count + negative_count
        if total_count == 0:
            sentiment_score = 0  # Trung tính
        else:
            sentiment_score = (positive_count - negative_count) / total_count
        
        # Ánh xạ điểm thành nhãn
        if sentiment_score <= -0.6:
            label = "Very Bearish"
        elif sentiment_score <= -0.2:
            label = "Bearish"
        elif sentiment_score <= 0.2:
            label = "Neutral"
        elif sentiment_score <= 0.6:
            label = "Bullish"
        else:
            label = "Very Bullish"
        
        return sentiment_score, label
    
    async def get_sentiment(self, asset: Optional[str] = None, hashtags: Optional[List[str]] = None, 
                          limit: int = 100) -> List[SentimentData]:
        """
        Lấy dữ liệu tâm lý từ Twitter.
        
        Args:
            asset: Mã tài sản (tùy chọn, ví dụ: "BTC", "ETH")
            hashtags: Danh sách hashtag cần tìm (tùy chọn)
            limit: Số lượng tweet tối đa
            
        Returns:
            Danh sách dữ liệu tâm lý
        """
        if not self.bearer_token:
            logger.error("Không có Twitter Bearer Token. Không thể tiếp tục.")
            return []
        
        # Xác định hashtags cần tìm
        if hashtags is None:
            if asset:
                # Nếu chỉ định tài sản, tìm các hashtag liên quan
                asset_hashtags = {
                    "BTC": ["#bitcoin", "#btc"],
                    "ETH": ["#ethereum", "#eth"],
                    # Thêm các tài sản khác...
                }
                hashtags = asset_hashtags.get(asset.upper(), self.default_hashtags)
            else:
                hashtags = self.default_hashtags
        
        # Tạo truy vấn tìm kiếm
        query = " OR ".join(hashtags) + " lang:en -is:retweet"
        
        try:
            # Tìm kiếm tweets
            search_results = await self.search_tweets(query, limit)
            
            if not search_results or 'data' not in search_results:
                logger.warning("Không tìm thấy tweets hoặc lỗi API")
                return []
            
            # Xử lý kết quả
            tweets = search_results['data']
            users = {user['id']: user for user in search_results.get('includes', {}).get('users', [])}
            
            results = []
            for tweet in tweets:
                try:
                    # Phân tích tâm lý từ nội dung tweet
                    sentiment_score, sentiment_label = self.analyze_sentiment(tweet['text'])
                    
                    # Lấy thông tin người dùng
                    user_id = tweet.get('author_id')
                    user_info = users.get(user_id, {})
                    
                    # Tạo metadata
                    metadata = {
                        'tweet_id': tweet.get('id'),
                        'user_id': user_id,
                        'username': user_info.get('username', ''),
                        'verified': user_info.get('verified', False),
                        'followers': user_info.get('public_metrics', {}).get('followers_count', 0),
                        'retweet_count': tweet.get('public_metrics', {}).get('retweet_count', 0),
                        'like_count': tweet.get('public_metrics', {}).get('like_count', 0),
                        'reply_count': tweet.get('public_metrics', {}).get('reply_count', 0),
                        'tweet_text': tweet.get('text', '')
                    }
                    
                    # Tạo timestamp từ created_at
                    created_at = tweet.get('created_at')
                    timestamp = datetime.datetime.fromisoformat(created_at.replace('Z', '+00:00')) if created_at else datetime.datetime.now()
                    
                    # Tạo đối tượng SentimentData
                    sentiment_data = SentimentData(
                        value=sentiment_score,
                        label=sentiment_label,
                        source=self.name,
                        timestamp=timestamp,
                        asset=asset,
                        timeframe="realtime",
                        metadata=metadata
                    )
                    
                    results.append(sentiment_data)
                    
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý tweet: {str(e)}")
            
            # Tính toán tâm lý tổng thể
            if results:
                avg_sentiment = sum(result.value for result in results) / len(results)
                
                # Xác định nhãn cho tâm lý tổng thể
                if avg_sentiment <= -0.6:
                    overall_label = "Very Bearish"
                elif avg_sentiment <= -0.2:
                    overall_label = "Bearish"
                elif avg_sentiment <= 0.2:
                    overall_label = "Neutral"
                elif avg_sentiment <= 0.6:
                    overall_label = "Bullish"
                else:
                    overall_label = "Very Bullish"
                
                # Thêm tâm lý tổng thể vào kết quả
                overall_sentiment = SentimentData(
                    value=avg_sentiment,
                    label=overall_label,
                    source=f"{self.name} (Overall)",
                    timestamp=datetime.datetime.now(),
                    asset=asset,
                    timeframe="realtime",
                    metadata={
                        'hashtags': hashtags,
                        'tweet_count': len(results),
                        'query': query
                    }
                )
                
                results.append(overall_sentiment)
            
            return results
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu tâm lý từ Twitter: {str(e)}")
            return []


class RedditSentimentScraper(SentimentSource):
    """Lớp thu thập dữ liệu tâm lý từ Reddit."""
    
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        """
        Khởi tạo scraper Reddit Sentiment.
        
        Args:
            client_id: Client ID của Reddit API (tùy chọn)
            client_secret: Client Secret của Reddit API (tùy chọn)
        """
        super().__init__(
            name="Reddit Sentiment",
            description="Phân tích tâm lý thị trường từ các bài viết Reddit về tiền điện tử"
        )
        self.client_id = client_id or get_env("REDDIT_CLIENT_ID", "")
        self.client_secret = client_secret or get_env("REDDIT_CLIENT_SECRET", "")
        self.user_agent = "AutomatedTradingSystem/1.0"
        
        # Danh sách các subreddits liên quan đến tiền điện tử
        self.crypto_subreddits = [
            "CryptoCurrency", "Bitcoin", "Ethereum", "CryptoMarkets",
            "BitcoinMarkets", "altcoin", "SatoshiStreetBets", "CryptoMoonShots"
        ]
        
        # Mapping tài sản với subreddits tương ứng
        self.asset_to_subreddit = {
            "BTC": ["Bitcoin", "BitcoinMarkets"],
            "ETH": ["Ethereum", "ethtrader", "ethfinance"],
            # Thêm các tài sản khác...
        }
    
    async def get_reddit_token(self) -> Optional[str]:
        """
        Lấy access token từ Reddit API.
        
        Returns:
            Access token hoặc None nếu có lỗi
        """
        if not self.client_id or not self.client_secret:
            logger.error("Thiếu Reddit API credentials (client_id hoặc client_secret)")
            return None
        
        auth_url = "https://www.reddit.com/api/v1/access_token"
        auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
        
        data = {
            "grant_type": "client_credentials",
            "duration": "temporary"
        }
        
        headers = {
            "User-Agent": self.user_agent
        }
        
        # Sử dụng phương pháp khác nhau tùy vào hệ điều hành
        if IS_WINDOWS:
            try:
                response = requests.post(
                    auth_url,
                    auth=(self.client_id, self.client_secret),
                    data=data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    return response.json().get("access_token")
                else:
                    logger.error(f"Lỗi khi lấy Reddit token: {response.status_code}, {response.text}")
                    return None
            except Exception as e:
                logger.error(f"Lỗi khi lấy Reddit token: {str(e)}")
                return None
        else:
            await self.initialize_session()
            
            try:
                async with self.session.post(
                    auth_url, 
                    auth=auth,
                    data=data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        json_data = await response.json()
                        return json_data.get("access_token")
                    else:
                        error_text = await response.text()
                        logger.error(f"Lỗi khi lấy Reddit token: {response.status}, {error_text}")
                        return None
            except Exception as e:
                logger.error(f"Lỗi khi lấy Reddit token: {str(e)}")
                return None
    
    async def fetch_subreddit_posts(self, subreddit: str, sort: str = "hot", 
                                   limit: int = 100, token: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Lấy bài viết từ subreddit.
        
        Args:
            subreddit: Tên subreddit
            sort: Cách sắp xếp ("hot", "new", "top", "rising")
            limit: Số lượng bài viết tối đa
            token: Access token (tùy chọn)
            
        Returns:
            Dữ liệu bài viết hoặc None nếu có lỗi
        """
        # Lấy token nếu chưa có
        if not token:
            token = await self.get_reddit_token()
            if not token:
                return None
        
        api_url = f"https://oauth.reddit.com/r/{subreddit}/{sort}"
        
        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": self.user_agent
        }
        
        params = {
            "limit": min(limit, 100)  # Reddit API giới hạn 100 kết quả mỗi request
        }
        
        # Sử dụng phương pháp khác nhau tùy vào hệ điều hành
        if IS_WINDOWS:
            try:
                response = requests.get(api_url, headers=headers, params=params)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Lỗi khi lấy bài viết Reddit từ r/{subreddit}: {response.status_code}, {response.text}")
                    return None
            except Exception as e:
                logger.error(f"Lỗi khi lấy bài viết Reddit từ r/{subreddit}: {str(e)}")
                return None
        else:
            try:
                async with self.session.get(api_url, headers=headers, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Lỗi khi lấy bài viết Reddit từ r/{subreddit}: {response.status}, {error_text}")
                        return None
            except Exception as e:
                logger.error(f"Lỗi khi lấy bài viết Reddit từ r/{subreddit}: {str(e)}")
                return None
    
    def analyze_sentiment(self, text: str) -> Tuple[float, str]:
        """
        Phân tích tâm lý từ văn bản Reddit.
        Tương tự như TwitterSentimentScraper, đây là phân tích đơn giản dựa trên từ điển.
        
        Args:
            text: Nội dung bài viết
            
        Returns:
            Tuple (điểm tâm lý, nhãn)
        """
        # Từ điển đơn giản các từ tích cực/tiêu cực
        positive_words = [
            "bull", "bullish", "buy", "long", "moon", "mooning", "pump", "pumping",
            "gain", "gains", "profit", "profits", "up", "uptrend", "higher", "rise",
            "rising", "good", "great", "excellent", "amazing", "positive", "awesome",
            "perfect", "opportunity", "potential", "confident", "confidence", "strong",
            "strength", "winning", "win", "hodl", "hold", "green", "success", "successful"
        ]
        
        negative_words = [
            "bear", "bearish", "sell", "short", "dump", "dumping", "crash", "crashing",
            "loss", "losses", "down", "downtrend", "lower", "fall", "falling", "bad",
            "terrible", "poor", "negative", "awful", "worst", "fear", "fearful", "weak",
            "weakness", "fail", "failing", "failure", "red", "risk", "risky", "danger",
            "dangerous", "trouble", "worried", "worry", "scam", "fraud", "bubble", "correction"
        ]
        
        # Chuyển văn bản thành chữ thường
        text_lower = text.lower()
        
        # Đếm số từ tích cực và tiêu cực
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Tính điểm tâm lý từ -1 (rất tiêu cực) đến 1 (rất tích cực)
        total_count = positive_count + negative_count
        if total_count == 0:
            sentiment_score = 0  # Trung tính
        else:
            sentiment_score = (positive_count - negative_count) / total_count
        
        # Ánh xạ điểm thành nhãn
        if sentiment_score <= -0.6:
            label = "Very Bearish"
        elif sentiment_score <= -0.2:
            label = "Bearish"
        elif sentiment_score <= 0.2:
            label = "Neutral"
        elif sentiment_score <= 0.6:
            label = "Bullish"
        else:
            label = "Very Bullish"
        
        return sentiment_score, label
    
    async def get_sentiment(self, asset: Optional[str] = None, 
                          subreddits: Optional[List[str]] = None,
                          limit: int = 100) -> List[SentimentData]:
        """
        Lấy dữ liệu tâm lý từ Reddit.
        
        Args:
            asset: Mã tài sản (tùy chọn)
            subreddits: Danh sách subreddit cần quét (tùy chọn)
            limit: Số lượng bài viết tối đa mỗi subreddit
            
        Returns:
            Danh sách dữ liệu tâm lý
        """
        # Xác định subreddits cần quét
        if subreddits is None:
            if asset and asset in self.asset_to_subreddit:
                subreddits = self.asset_to_subreddit[asset]
            else:
                subreddits = self.crypto_subreddits
        
        # Lấy token API
        token = await self.get_reddit_token()
        if not token:
            logger.error("Không thể lấy Reddit API token, dừng thu thập")
            return []
        
        # Thu thập bài viết từ các subreddit
        results = []
        for subreddit in subreddits:
            try:
                # Lấy bài viết "hot"
                posts_data = await self.fetch_subreddit_posts(
                    subreddit=subreddit,
                    sort="hot",
                    limit=limit,
                    token=token
                )
                
                if not posts_data or 'data' not in posts_data or 'children' not in posts_data['data']:
                    logger.warning(f"Không tìm thấy bài viết từ r/{subreddit}")
                    continue
                
                posts = posts_data['data']['children']
                
                # Phân tích tâm lý cho mỗi bài viết
                for post in posts:
                    try:
                        post_data = post['data']
                        
                        # Kết hợp tiêu đề và selftext để phân tích
                        title = post_data.get('title', '')
                        selftext = post_data.get('selftext', '')
                        full_text = f"{title}\n{selftext}"
                        
                        # Phân tích tâm lý
                        sentiment_score, sentiment_label = self.analyze_sentiment(full_text)
                        
                        # Tạo timestamp
                        created_utc = post_data.get('created_utc', 0)
                        timestamp = datetime.datetime.fromtimestamp(created_utc)
                        
                        # Tạo metadata
                        metadata = {
                            'post_id': post_data.get('id', ''),
                            'subreddit': post_data.get('subreddit', ''),
                            'author': post_data.get('author', ''),
                            'title': title,
                            'upvote_ratio': post_data.get('upvote_ratio', 0),
                            'score': post_data.get('score', 0),
                            'num_comments': post_data.get('num_comments', 0),
                            'is_original_content': post_data.get('is_original_content', False),
                            'permalink': post_data.get('permalink', '')
                        }
                        
                        # Tạo đối tượng SentimentData
                        sentiment_data = SentimentData(
                            value=sentiment_score,
                            label=sentiment_label,
                            source=f"Reddit - r/{subreddit}",
                            timestamp=timestamp,
                            asset=asset,
                            timeframe="recent",
                            metadata=metadata
                        )
                        
                        results.append(sentiment_data)
                        
                    except Exception as e:
                        logger.error(f"Lỗi khi xử lý bài viết Reddit: {str(e)}")
                
                # Chờ một chút để tránh quá nhiều request
                if IS_WINDOWS:
                    time.sleep(1)
                else:
                    await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Lỗi khi lấy dữ liệu từ subreddit r/{subreddit}: {str(e)}")
        
        # Tính toán tâm lý tổng thể cho mỗi subreddit
        if results:
            # Nhóm kết quả theo subreddit
            subreddit_results = {}
            for result in results:
                subreddit = result.metadata.get('subreddit', '')
                if subreddit not in subreddit_results:
                    subreddit_results[subreddit] = []
                subreddit_results[subreddit].append(result)
            
            # Tính toán tâm lý tổng thể cho mỗi subreddit
            for subreddit, subreddit_data in subreddit_results.items():
                if subreddit_data:
                    avg_sentiment = sum(data.value for data in subreddit_data) / len(subreddit_data)
                    
                    # Xác định nhãn
                    if avg_sentiment <= -0.6:
                        overall_label = "Very Bearish"
                    elif avg_sentiment <= -0.2:
                        overall_label = "Bearish"
                    elif avg_sentiment <= 0.2:
                        overall_label = "Neutral"
                    elif avg_sentiment <= 0.6:
                        overall_label = "Bullish"
                    else:
                        overall_label = "Very Bullish"
                    
                    # Thêm kết quả tổng thể
                    overall_sentiment = SentimentData(
                        value=avg_sentiment,
                        label=overall_label,
                        source=f"Reddit - r/{subreddit} (Overall)",
                        timestamp=datetime.datetime.now(),
                        asset=asset,
                        timeframe="recent",
                        metadata={
                            'subreddit': subreddit,
                            'post_count': len(subreddit_data)
                        }
                    )
                    
                    results.append(overall_sentiment)
            
            # Tính toán tâm lý tổng thể cho tất cả subreddits
            avg_sentiment_all = sum(result.value for result in results if not result.source.endswith("(Overall)")) / sum(1 for result in results if not result.source.endswith("(Overall)"))
            
            # Xác định nhãn tổng thể
            if avg_sentiment_all <= -0.6:
                overall_label_all = "Very Bearish"
            elif avg_sentiment_all <= -0.2:
                overall_label_all = "Bearish"
            elif avg_sentiment_all <= 0.2:
                overall_label_all = "Neutral"
            elif avg_sentiment_all <= 0.6:
                overall_label_all = "Bullish"
            else:
                overall_label_all = "Very Bullish"
            
            # Thêm kết quả tổng thể cho tất cả subreddits
            overall_sentiment_all = SentimentData(
                value=avg_sentiment_all,
                label=overall_label_all,
                source="Reddit (Overall)",
                timestamp=datetime.datetime.now(),
                asset=asset,
                timeframe="recent",
                metadata={
                    'subreddits': subreddits,
                    'post_count': sum(1 for result in results if not result.source.endswith("(Overall)"))
                }
            )
            
            results.append(overall_sentiment_all)
        
        return results


class GlassNodeSentiment(SentimentSource):
    """Lớp thu thập dữ liệu tâm lý từ GlassNode."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Khởi tạo nguồn dữ liệu GlassNode.
        
        Args:
            api_key: API key cho GlassNode (tùy chọn)
        """
        super().__init__(
            name="GlassNode",
            description="Dữ liệu on-chain và chỉ số tâm lý từ GlassNode"
        )
        self.api_key = api_key or get_env("GLASSNODE_API_KEY", "")
        self.api_url = "https://api.glassnode.com/v1/metrics"
    
    async def fetch_metric(self, metric: str, asset: str = "BTC", 
                          since: Optional[int] = None, until: Optional[int] = None,
                          interval: str = "24h") -> Optional[List[Dict[str, Any]]]:
        """
        Lấy dữ liệu metric từ GlassNode API.
        
        Args:
            metric: Tên metric (ví dụ: "sopr", "nupl", "puell_multiple")
            asset: Mã tài sản (mặc định: "BTC")
            since: Timestamp bắt đầu (Unix timestamp)
            until: Timestamp kết thúc (Unix timestamp)
            interval: Khoảng thời gian ("10m", "1h", "24h", "1w", "1month")
            
        Returns:
            Dữ liệu metric hoặc None nếu có lỗi
        """
        if not self.api_key:
            logger.error("Không tìm thấy GlassNode API key")
            return None
        
        endpoint = f"{self.api_url}/{metric}"
        
        params = {
            "a": asset,
            "api_key": self.api_key,
            "i": interval
        }
        
        if since:
            params["s"] = since
            
        if until:
            params["u"] = until
        
        # Phương pháp khác nhau tùy vào hệ điều hành
        if IS_WINDOWS:
            try:
                headers = self.get_browser_headers()
                headers["Accept"] = "application/json"
                
                response = requests.get(endpoint, params=params, headers=headers)
                
                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Lỗi khi lấy dữ liệu GlassNode {metric}: {response.status_code}, {response.text}")
                    return None
            except Exception as e:
                logger.error(f"Lỗi khi lấy dữ liệu GlassNode {metric}: {str(e)}")
                return None
        else:
            try:
                await self.initialize_session()
                
                async with self.session.get(endpoint, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        error_text = await response.text()
                        logger.error(f"Lỗi khi lấy dữ liệu GlassNode {metric}: {response.status}, {error_text}")
                        return None
            except Exception as e:
                logger.error(f"Lỗi khi lấy dữ liệu GlassNode {metric}: {str(e)}")
                return None
    
    def _get_sopr_label(self, value: float) -> str:
        """
        Ánh xạ giá trị SOPR thành nhãn tâm lý.
        
        Args:
            value: Giá trị SOPR
            
        Returns:
            Nhãn tâm lý
        """
        if value < 0.95:
            return "Bearish"
        elif value < 1.0:
            return "Slightly Bearish"
        elif value < 1.05:
            return "Slightly Bullish"
        else:
            return "Bullish"
    
    def _get_nupl_label(self, value: float) -> str:
        """
        Ánh xạ giá trị NUPL thành nhãn tâm lý.
        
        Args:
            value: Giá trị NUPL
            
        Returns:
            Nhãn tâm lý
        """
        if value < -0.2:
            return "Capitulation"
        elif value < 0:
            return "Fear"
        elif value < 0.25:
            return "Hope/Fear"
        elif value < 0.5:
            return "Optimism/Belief"
        elif value < 0.75:
            return "Euphoria/Greed"
        else:
            return "Extreme Greed"
    
    def _get_puell_multiple_label(self, value: float) -> str:
        """
        Ánh xạ giá trị Puell Multiple thành nhãn tâm lý.
        
        Args:
            value: Giá trị Puell Multiple
            
        Returns:
            Nhãn tâm lý
        """
        if value < 0.5:
            return "Market Bottom"
        elif value < 0.8:
            return "Buy Zone"
        elif value < 1.5:
            return "Fair Value"
        elif value < 2.5:
            return "Sell Zone"
        else:
            return "Market Top"
    
    async def get_sentiment(self, asset: str = "BTC", 
                          metrics: Optional[List[str]] = None,
                          days: int = 30) -> List[SentimentData]:
        """
        Lấy dữ liệu tâm lý từ GlassNode.
        
        Args:
            asset: Mã tài sản (mặc định: "BTC")
            metrics: Danh sách metrics cần lấy (tùy chọn)
            days: Số ngày lấy dữ liệu lịch sử
            
        Returns:
            Danh sách dữ liệu tâm lý
        """
        if not self.api_key:
            logger.error("Không tìm thấy GlassNode API key. Không thể tiếp tục.")
            return []
        
        await self.initialize_session()
        
        # Metrics mặc định nếu không chỉ định
        if metrics is None:
            metrics = ["sopr", "nupl", "puell_multiple"]
        
        # Tính toán timestamp
        until = int(time.time())
        since = until - (days * 24 * 60 * 60)
        
        results = []
        
        for metric in metrics:
            try:
                # Lấy dữ liệu metric
                data = await self.fetch_metric(
                    metric=metric,
                    asset=asset,
                    since=since,
                    until=until,
                    interval="24h"
                )
                
                if not data:
                    logger.warning(f"Không có dữ liệu cho metric {metric}")
                    continue
                
                # Xử lý dữ liệu
                for item in data:
                    try:
                        timestamp_ms = item.get('t', 0)
                        value = item.get('v', 0)
                        
                        # Chuyển đổi timestamp từ milliseconds sang datetime
                        timestamp = datetime.datetime.fromtimestamp(timestamp_ms / 1000)
                        
                        # Xác định nhãn dựa trên loại metric
                        if metric == "sopr":
                            label = self._get_sopr_label(value)
                            display_name = "SOPR (Spent Output Profit Ratio)"
                        elif metric == "nupl":
                            label = self._get_nupl_label(value)
                            display_name = "NUPL (Net Unrealized Profit/Loss)"
                        elif metric == "puell_multiple":
                            label = self._get_puell_multiple_label(value)
                            display_name = "Puell Multiple"
                        else:
                            # Nhãn chung cho các metrics khác
                            if value < 0:
                                label = "Bearish"
                            elif value == 0:
                                label = "Neutral"
                            else:
                                label = "Bullish"
                            display_name = metric
                        
                        # Tạo đối tượng SentimentData
                        sentiment_data = SentimentData(
                            value=value,
                            label=label,
                            source=f"GlassNode - {display_name}",
                            timestamp=timestamp,
                            asset=asset,
                            timeframe="1d",
                            metadata={
                                'metric': metric,
                                'display_name': display_name
                            }
                        )
                        
                        results.append(sentiment_data)
                        
                    except Exception as e:
                        logger.error(f"Lỗi khi xử lý dữ liệu GlassNode {metric}: {str(e)}")
                
            except Exception as e:
                logger.error(f"Lỗi khi lấy dữ liệu GlassNode cho metric {metric}: {str(e)}")
        
        return results


class SantimentDataCollector(SentimentSource):
    """Lớp thu thập dữ liệu tâm lý từ Santiment API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Khởi tạo collector Santiment.
        
        Args:
            api_key: API key cho Santiment (tùy chọn)
        """
        super().__init__(
            name="Santiment",
            description="Dữ liệu tâm lý thị trường từ Santiment API"
        )
        self.api_key = api_key or get_env("SANTIMENT_API_KEY", "")
        self.base_url = "https://api.santiment.net/graphql"
        self.api_call_counter = 0
        self.monthly_limit = 5000  # Giới hạn 5000 cuộc gọi/tháng
        
        # Tạo thư mục cache nếu chưa tồn tại
        self.cache_dir = Path("data/cache/santiment")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Thời gian cache (mặc định: 24 giờ)
        self.cache_duration = 24 * 60 * 60  # seconds
    
    def _get_cache_key(self, query_type: str, params: Dict[str, Any]) -> str:
        """
        Tạo khóa cache từ loại truy vấn và tham số.
        
        Args:
            query_type: Loại truy vấn (ví dụ: 'social_volume', 'news')
            params: Tham số truy vấn
            
        Returns:
            Khóa cache
        """
        # Tạo chuỗi đại diện cho tham số
        param_str = json.dumps(params, sort_keys=True)
        
        # Tạo hash từ query_type và param_str
        cache_key = hashlib.md5(f"{query_type}_{param_str}".encode()).hexdigest()
        
        return cache_key
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Lấy đường dẫn tệp cache.
        
        Args:
            cache_key: Khóa cache
            
        Returns:
            Đường dẫn tệp cache
        """
        return self.cache_dir / f"{cache_key}.json"
    
    def _save_to_cache(self, cache_key: str, data: Any) -> bool:
        """
        Lưu dữ liệu vào cache.
        
        Args:
            cache_key: Khóa cache
            data: Dữ liệu cần lưu
            
        Returns:
            True nếu lưu thành công, False nếu có lỗi
        """
        try:
            cache_path = self._get_cache_path(cache_key)
            
            # Thêm timestamp vào dữ liệu cache
            cache_data = {
                "timestamp": time.time(),
                "data": data
            }
            
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f)
                
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu dữ liệu vào cache: {str(e)}")
            return False
    
    def _load_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Tải dữ liệu từ cache.
        
        Args:
            cache_key: Khóa cache
            
        Returns:
            Dữ liệu từ cache hoặc None nếu không có cache hợp lệ
        """
        try:
            cache_path = self._get_cache_path(cache_key)
            
            # Kiểm tra nếu cache tồn tại
            if not cache_path.exists():
                return None
                
            # Đọc dữ liệu cache
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            # Kiểm tra xem cache còn hiệu lực không
            if time.time() - cache_data["timestamp"] > self.cache_duration:
                logger.debug(f"Cache cho {cache_key} đã hết hạn")
                return None
                
            return cache_data["data"]
        except Exception as e:
            logger.error(f"Lỗi khi tải dữ liệu từ cache: {str(e)}")
            return None
    
    async def execute_graphql(self, query: str, variables: Dict[str, Any] = None, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Thực thi truy vấn GraphQL.
        
        Args:
            query: Truy vấn GraphQL
            variables: Biến truy vấn
            use_cache: Sử dụng cache hay không
            
        Returns:
            Kết quả truy vấn hoặc None nếu có lỗi
        """
        if not self.api_key:
            logger.error("Không tìm thấy Santiment API key")
            return None
        
        # Tạo khóa cache
        cache_key = self._get_cache_key(query[:50], variables or {})
        
        # Kiểm tra cache nếu được yêu cầu
        if use_cache:
            cached_data = self._load_from_cache(cache_key)
            if cached_data is not None:
                logger.debug(f"Đã tải dữ liệu từ cache cho truy vấn {cache_key}")
                return cached_data
        
        # Kiểm tra giới hạn API
        if self.api_call_counter >= self.monthly_limit:
            logger.error(f"Đã đạt giới hạn {self.monthly_limit} cuộc gọi API Santiment trong tháng")
            return None
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Apikey {self.api_key}"
        }
        
        payload = {
            "query": query,
            "variables": variables or {}
        }
        
        try:
            # Sử dụng requests đồng bộ trên Windows
            if IS_WINDOWS:
                response = requests.post(
                    self.base_url,
                    json=payload,
                    headers=headers
                )
                
                self.api_call_counter += 1
                
                if response.status_code != 200:
                    logger.error(f"Lỗi khi thực thi GraphQL: {response.status_code}, {response.text}")
                    return None
                
                data = response.json()
            else:
                # Sử dụng aiohttp cho các hệ điều hành khác
                await self.initialize_session()
                
                async with self.session.post(
                    self.base_url,
                    json=payload,
                    headers=headers
                ) as response:
                    self.api_call_counter += 1
                    
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Lỗi khi thực thi GraphQL: {response.status}, {error_text}")
                        return None
                        
                    data = await response.json()
            
            # Kiểm tra lỗi trong phản hồi
            if "errors" in data:
                errors = data["errors"]
                logger.error(f"GraphQL trả về lỗi: {errors}")
                return None
                
            # Lưu vào cache nếu được yêu cầu
            if use_cache:
                self._save_to_cache(cache_key, data)
                
            return data
            
        except Exception as e:
            logger.error(f"Lỗi khi thực thi truy vấn GraphQL: {str(e)}")
            return None
    
    async def get_social_volume(self, asset: str, from_date: str, to_date: str, 
                              source: str = "telegram", use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Lấy dữ liệu về khối lượng đề cập trên mạng xã hội.
        
        Args:
            asset: Mã tài sản (ví dụ: "bitcoin", "ethereum")
            from_date: Ngày bắt đầu (định dạng: "YYYY-MM-DD")
            to_date: Ngày kết thúc (định dạng: "YYYY-MM-DD")
            source: Nguồn dữ liệu ("telegram", "twitter", "reddit", "professional_traders_chat")
            use_cache: Sử dụng cache hay không
            
        Returns:
            Dữ liệu khối lượng mạng xã hội hoặc None nếu có lỗi
        """
        query = """
        query socialVolume($slug: String!, $from: DateTime!, $to: DateTime!, $source: String!) {
          socialVolume(slug: $slug, from: $from, to: $to, source: $source, interval: "1d") {
            datetime
            mentionsCount
          }
        }
        """
        
        variables = {
            "slug": asset.lower(),
            "from": from_date,
            "to": to_date,
            "source": source
        }
        
        result = await self.execute_graphql(query, variables, use_cache)
        
        if result and "data" in result and "socialVolume" in result["data"]:
            return result["data"]["socialVolume"]
        
        return None
    
    async def get_social_sentiment(self, asset: str, from_date: str, to_date: str, 
                                 source: str = "telegram", use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Lấy dữ liệu về tâm lý mạng xã hội.
        
        Args:
            asset: Mã tài sản (ví dụ: "bitcoin", "ethereum")
            from_date: Ngày bắt đầu (định dạng: "YYYY-MM-DD")
            to_date: Ngày kết thúc (định dạng: "YYYY-MM-DD")
            source: Nguồn dữ liệu ("telegram", "twitter", "reddit", "professional_traders_chat")
            use_cache: Sử dụng cache hay không
            
        Returns:
            Dữ liệu tâm lý mạng xã hội hoặc None nếu có lỗi
        """
        query = """
        query socialSentiment($slug: String!, $from: DateTime!, $to: DateTime!, $source: String!) {
          socialSentiment(slug: $slug, from: $from, to: $to, source: $source, interval: "1d") {
            datetime
            sentiment
            sentimentPositive
            sentimentNegative
            sentimentBearish
            sentimentBullish
            sentimentVolume
          }
        }
        """
        
        variables = {
            "slug": asset.lower(),
            "from": from_date,
            "to": to_date,
            "source": source
        }
        
        result = await self.execute_graphql(query, variables, use_cache)
        
        if result and "data" in result and "socialSentiment" in result["data"]:
            return result["data"]["socialSentiment"]
        
        return None
    
    async def get_news_sentiment(self, asset: str, from_date: str, to_date: str, 
                              use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Lấy dữ liệu về tâm lý tin tức.
        
        Args:
            asset: Mã tài sản (ví dụ: "bitcoin", "ethereum")
            from_date: Ngày bắt đầu (định dạng: "YYYY-MM-DD")
            to_date: Ngày kết thúc (định dạng: "YYYY-MM-DD")
            use_cache: Sử dụng cache hay không
            
        Returns:
            Dữ liệu tâm lý tin tức hoặc None nếu có lỗi
        """
        query = """
        query newsSentiment($slug: String!, $from: DateTime!, $to: DateTime!) {
          newsSentiment(slug: $slug, from: $from, to: $to, interval: "1d") {
            datetime
            sentiment
            sentimentPositive
            sentimentNegative
            mentionsCount
          }
        }
        """
        
        variables = {
            "slug": asset.lower(),
            "from": from_date,
            "to": to_date
        }
        
        result = await self.execute_graphql(query, variables, use_cache)
        
        if result and "data" in result and "newsSentiment" in result["data"]:
            return result["data"]["newsSentiment"]
        
        return None
    
    async def get_price_sentiment(self, asset: str, from_date: str, to_date: str, 
                                use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Lấy dữ liệu về phân tích tâm lý giá.
        
        Args:
            asset: Mã tài sản (ví dụ: "bitcoin", "ethereum")
            from_date: Ngày bắt đầu (định dạng: "YYYY-MM-DD")
            to_date: Ngày kết thúc (định dạng: "YYYY-MM-DD")
            use_cache: Sử dụng cache hay không
            
        Returns:
            Dữ liệu phân tích tâm lý giá hoặc None nếu có lỗi
        """
        query = """
        query priceSentiment($slug: String!, $from: DateTime!, $to: DateTime!) {
          priceSentiment(slug: $slug, from: $from, to: $to, interval: "1d") {
            datetime
            value
          }
        }
        """
        
        variables = {
            "slug": asset.lower(),
            "from": from_date,
            "to": to_date
        }
        
        result = await self.execute_graphql(query, variables, use_cache)
        
        if result and "data" in result and "priceSentiment" in result["data"]:
            return result["data"]["priceSentiment"]
        
        return None
    
    def _map_sentiment_to_label(self, sentiment: float) -> str:
        """
        Ánh xạ giá trị tâm lý thành nhãn.
        
        Args:
            sentiment: Giá trị tâm lý (thang đo từ -1 đến 1)
            
        Returns:
            Nhãn tâm lý
        """
        if sentiment <= -0.6:
            return "Very Bearish"
        elif sentiment <= -0.2:
            return "Bearish"
        elif sentiment <= 0.2:
            return "Neutral"
        elif sentiment <= 0.6:
            return "Bullish"
        else:
            return "Very Bullish"
    
    def _parse_santiment_data(self, data: List[Dict[str, Any]], asset: str, source: str) -> List[SentimentData]:
        """
        Chuyển đổi dữ liệu Santiment sang SentimentData.
        
        Args:
            data: Dữ liệu từ Santiment API
            asset: Mã tài sản
            source: Nguồn dữ liệu
            
        Returns:
            Danh sách SentimentData
        """
        results = []
        
        for item in data:
            try:
                # Lấy timestamp
                datetime_str = item.get("datetime")
                if not datetime_str:
                    continue
                    
                timestamp = datetime.datetime.fromisoformat(datetime_str.replace("Z", "+00:00"))
                
                # Lấy giá trị tâm lý
                if "sentiment" in item:
                    sentiment_value = item["sentiment"]
                elif "value" in item:
                    sentiment_value = item["value"]
                else:
                    # Tính tâm lý từ tỷ lệ positive/negative nếu có
                    positive = item.get("sentimentPositive", 0)
                    negative = item.get("sentimentNegative", 0)
                    
                    if positive != 0 or negative != 0:
                        sentiment_value = (positive - negative) / (positive + negative)
                    else:
                        sentiment_value = 0
                
                # Ánh xạ giá trị thành nhãn
                sentiment_label = self._map_sentiment_to_label(sentiment_value)
                
                # Tạo metadata
                metadata = {}
                for key, value in item.items():
                    if key not in ["datetime", "sentiment", "value"]:
                        metadata[key] = value
                
                # Tạo đối tượng SentimentData
                sentiment_data = SentimentData(
                    value=sentiment_value,
                    label=sentiment_label,
                    source=f"Santiment - {source}",
                    timestamp=timestamp,
                    asset=asset,
                    timeframe="1d",
                    metadata=metadata
                )
                
                results.append(sentiment_data)
                
            except Exception as e:
                logger.error(f"Lỗi khi xử lý dữ liệu Santiment: {str(e)}")
        
        return results
    
    async def get_sentiment(self, asset: str = "bitcoin", days: int = 30, 
                          data_source: str = "all", **kwargs) -> List[SentimentData]:
        """
        Lấy dữ liệu tâm lý từ Santiment.
        
        Args:
            asset: Mã tài sản (mặc định: "bitcoin")
            days: Số ngày lịch sử (mặc định: 30)
            data_source: Nguồn dữ liệu ("social", "news", "price", "all")
            
        Returns:
            Danh sách dữ liệu tâm lý
        """
        if not self.api_key:
            logger.error("Không tìm thấy Santiment API key. Không thể tiếp tục.")
            return []
        
        await self.initialize_session()
        
        # Tính toán khoảng thời gian
        to_date = datetime.datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        
        results = []
        
        # Xác định các nguồn dữ liệu cần lấy
        if data_source in ["social", "all"]:
            # Lấy dữ liệu tâm lý từ Telegram
            telegram_data = await self.get_social_sentiment(asset, from_date, to_date, "telegram")
            if telegram_data:
                results.extend(self._parse_santiment_data(telegram_data, asset, "Telegram"))
            
            # Lấy dữ liệu tâm lý từ Twitter/X
            twitter_data = await self.get_social_sentiment(asset, from_date, to_date, "twitter")
            if twitter_data:
                results.extend(self._parse_santiment_data(twitter_data, asset, "Twitter"))
            
            # Lấy dữ liệu tâm lý từ Reddit
            reddit_data = await self.get_social_sentiment(asset, from_date, to_date, "reddit")
            if reddit_data:
                results.extend(self._parse_santiment_data(reddit_data, asset, "Reddit"))
        
        # Lấy dữ liệu tâm lý từ tin tức
        if data_source in ["news", "all"]:
            news_data = await self.get_news_sentiment(asset, from_date, to_date)
            if news_data:
                results.extend(self._parse_santiment_data(news_data, asset, "News"))
        
        # Lấy dữ liệu phân tích tâm lý giá
        if data_source in ["price", "all"]:
            price_data = await self.get_price_sentiment(asset, from_date, to_date)
            if price_data:
                results.extend(self._parse_santiment_data(price_data, asset, "Price Analysis"))
        
        # Tính toán tâm lý tổng thể từ tất cả nguồn
        if results:
            avg_sentiment = sum(result.value for result in results) / len(results)
            
            # Ánh xạ giá trị thành nhãn
            overall_label = self._map_sentiment_to_label(avg_sentiment)
            
            # Thêm kết quả tổng thể vào danh sách
            overall_sentiment = SentimentData(
                value=avg_sentiment,
                label=overall_label,
                source="Santiment (Overall)",
                timestamp=datetime.datetime.now(),
                asset=asset,
                timeframe="1d",
                metadata={
                    'data_sources': data_source,
                    'days': days,
                    'data_points': len(results)
                }
            )
            
            results.append(overall_sentiment)
        
        return results


class SentimentCollector:
    """Lớp chính để thu thập và quản lý dữ liệu tâm lý thị trường từ nhiều nguồn."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Khởi tạo collector tâm lý.
        
        Args:
            data_dir: Thư mục lưu trữ dữ liệu (tùy chọn)
        """
        self.sources = {}
        self.data_dir = data_dir
        
        if self.data_dir is None:
            # Sử dụng thư mục mặc định nếu không được chỉ định
            from config.system_config import BASE_DIR
            self.data_dir = BASE_DIR / "data" / "sentiment"
        
        # Tạo thư mục nếu chưa tồn tại
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Tạo logger
        self.logger = get_logger("sentiment_collector")
        
        # Khởi tạo các nguồn dữ liệu tâm lý
        self._initialize_sources()
    
    def _initialize_sources(self) -> None:
        """Khởi tạo các nguồn dữ liệu tâm lý mặc định."""
        # Thêm Fear & Greed Index
        self.add_source(FearAndGreedIndex())
        
        # Thêm Santiment nếu có API key
        santiment_api_key = get_env("SANTIMENT_API_KEY", "")
        if santiment_api_key:
            self.add_source(SantimentDataCollector(api_key=santiment_api_key))
        
        # Thêm Twitter Sentiment nếu có API key
        twitter_bearer_token = get_env("TWITTER_BEARER_TOKEN", "")
        if twitter_bearer_token:
            self.add_source(TwitterSentimentScraper())
        
        # Thêm Reddit Sentiment nếu có API keys
        reddit_client_id = get_env("REDDIT_CLIENT_ID", "")
        reddit_client_secret = get_env("REDDIT_CLIENT_SECRET", "")
        if reddit_client_id and reddit_client_secret:
            self.add_source(RedditSentimentScraper())
        
        # Thêm GlassNode nếu có API key
        glassnode_api_key = get_env("GLASSNODE_API_KEY", "")
        if glassnode_api_key:
            self.add_source(GlassNodeSentiment())
    
    def add_source(self, source: SentimentSource) -> None:
        """
        Thêm nguồn dữ liệu tâm lý vào collector.
        
        Args:
            source: Đối tượng nguồn dữ liệu tâm lý
        """
        self.sources[source.name] = source
        self.logger.info(f"Đã thêm nguồn dữ liệu tâm lý: {source.name}")
    
    def remove_source(self, source_name: str) -> bool:
        """
        Xóa nguồn dữ liệu tâm lý khỏi collector.
        
        Args:
            source_name: Tên nguồn dữ liệu tâm lý
            
        Returns:
            True nếu xóa thành công, False nếu không tìm thấy
        """
        if source_name in self.sources:
            del self.sources[source_name]
            self.logger.info(f"Đã xóa nguồn dữ liệu tâm lý: {source_name}")
            return True
        return False
    
    def get_sources(self) -> Dict[str, SentimentSource]:
        """
        Lấy danh sách các nguồn dữ liệu tâm lý.
        
        Returns:
            Từ điển các nguồn dữ liệu tâm lý
        """
        return self.sources
    
    async def collect_from_source(self, source_name: str, **kwargs) -> List[SentimentData]:
        """
        Thu thập dữ liệu tâm lý từ một nguồn cụ thể.
        
        Args:
            source_name: Tên nguồn dữ liệu tâm lý
            **kwargs: Tham số bổ sung cho nguồn
            
        Returns:
            Danh sách dữ liệu tâm lý thu thập được
        """
        if source_name not in self.sources:
            self.logger.error(f"Không tìm thấy nguồn dữ liệu tâm lý: {source_name}")
            return []
        
        source = self.sources[source_name]
        try:
            # Xử lý đặc biệt cho Fear and Greed Index
            if source_name == "Fear and Greed Index":
                # Sao chép kwargs và loại bỏ tham số không cần thiết
                kwargs_copy = {k: v for k, v in kwargs.items() if k not in ['asset']} 
                data = await source.get_sentiment(**kwargs_copy)
            else:
                data = await source.get_sentiment(**kwargs)
                
            self.logger.info(f"Đã thu thập {len(data)} bản ghi tâm lý từ {source_name}")
            return data
        except Exception as e:
            self.logger.error(f"Lỗi khi thu thập dữ liệu tâm lý từ {source_name}: {str(e)}")
            return []
        finally:
            # Đảm bảo đóng session
            await source.close_session()
    
    async def collect_specific_sentiment(self, data_source: str, **kwargs) -> List[SentimentData]:
        """
        Thu thập dữ liệu tâm lý từ nguồn cụ thể dựa vào loại dữ liệu.
        
        Args:
            data_source: Loại nguồn dữ liệu ('news', 'social')
            **kwargs: Tham số bổ sung cho nguồn
            
        Returns:
            Danh sách dữ liệu tâm lý thu thập được
        """
        results = []
        
        # Ưu tiên sử dụng Santiment API nếu có
        if "Santiment" in self.sources:
            # Chuẩn bị tham số cho Santiment
            source_params = kwargs.copy()
            
            # Chuyển đổi mã tài sản nếu cần
            if "asset" in source_params and source_params["asset"] in ["BTC", "ETH"]:
                asset_map = {"BTC": "bitcoin", "ETH": "ethereum"}
                source_params["asset"] = asset_map.get(source_params["asset"], source_params["asset"])
            
            # Thiết lập data_source phù hợp
            source_params["data_source"] = data_source
            
            # Thu thập dữ liệu từ Santiment
            santiment_data = await self.collect_from_source("Santiment", **source_params)
            results.extend(santiment_data)
            
            if results:
                self.logger.info(f"Đã thu thập {len(results)} bản ghi tâm lý từ Santiment với loại dữ liệu {data_source}")
                return results
        
        # Fallback về các nguồn khác nếu không có dữ liệu từ Santiment
        if data_source == "news":
            # Thu thập dữ liệu tâm lý tin tức từ các nguồn khác
            # Hiện tại chúng ta chưa có nguồn tin tức khác trong danh sách
            self.logger.warning("Không tìm thấy Santiment API hoặc nguồn tin tức khác")
        
        elif data_source == "social":
            # Thu thập dữ liệu từ Twitter nếu có
            if "Twitter Sentiment" in self.sources:
                twitter_data = await self.collect_from_source("Twitter Sentiment", **kwargs)
                results.extend(twitter_data)
            
            # Thu thập dữ liệu từ Reddit nếu có
            if "Reddit Sentiment" in self.sources:
                reddit_data = await self.collect_from_source("Reddit Sentiment", **kwargs)
                results.extend(reddit_data)
            
            if not results:
                self.logger.warning("Không tìm thấy Santiment API hoặc nguồn mạng xã hội khác")
        
        return results
    
    async def collect_from_all_sources(self, **kwargs) -> Dict[str, List[SentimentData]]:
        """
        Thu thập dữ liệu tâm lý từ tất cả các nguồn.
        
        Args:
            **kwargs: Tham số bổ sung cho các nguồn
            
        Returns:
            Từ điển với khóa là tên nguồn và giá trị là danh sách dữ liệu tâm lý
        """
        results = {}
        
        # Trên Windows, xử lý tuần tự để tránh vấn đề với asyncio
        if IS_WINDOWS:
            for source_name in self.sources:
                try:
                    # Xử lý đặc biệt cho Fear and Greed Index
                    if source_name == "Fear and Greed Index":
                        # Sao chép kwargs và loại bỏ tham số 'asset'
                        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['asset']}
                        data = await self.collect_from_source(source_name, **filtered_kwargs)
                    else:
                        data = await self.collect_from_source(source_name, **kwargs)
                    results[source_name] = data
                except Exception as e:
                    self.logger.error(f"Lỗi khi thu thập dữ liệu tâm lý từ {source_name}: {str(e)}")
                    results[source_name] = []
        else:
            # Trên các hệ điều hành khác, sử dụng asyncio để xử lý song song
            tasks = []
            for source_name in self.sources:
                # Xử lý đặc biệt cho Fear and Greed Index
                if source_name == "Fear and Greed Index":
                    # Sao chép kwargs và loại bỏ tham số 'asset'
                    filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ['asset']}
                    task = self.collect_from_source(source_name, **filtered_kwargs)
                else:
                    task = self.collect_from_source(source_name, **kwargs)
                tasks.append((source_name, task))
            
            for source_name, task in tasks:
                try:
                    data = await task
                    results[source_name] = data
                except Exception as e:
                    self.logger.error(f"Lỗi khi thu thập dữ liệu tâm lý từ {source_name}: {str(e)}")
                    results[source_name] = []
        
        return results
    
    def save_to_json(self, data: List[SentimentData], filename: Optional[str] = None) -> str:
        """
        Lưu dữ liệu tâm lý vào file JSON.
        
        Args:
            data: Danh sách dữ liệu tâm lý cần lưu
            filename: Tên file (tùy chọn)
            
        Returns:
            Đường dẫn đến file đã lưu
        """
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crypto_sentiment_{timestamp}.json"
        
        file_path = self.data_dir / filename
        
        try:
            # Chuyển đổi đối tượng SentimentData thành dict
            serializable_data = [item.to_dict() for item in data]
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Đã lưu {len(data)} bản ghi tâm lý vào {file_path}")
            return str(file_path)
        
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu dữ liệu tâm lý vào file JSON: {str(e)}")
            return ""
    
    def save_to_csv(self, data: List[SentimentData], filename: Optional[str] = None) -> str:
        """
        Lưu dữ liệu tâm lý vào file CSV.
        
        Args:
            data: Danh sách dữ liệu tâm lý cần lưu
            filename: Tên file (tùy chọn)
            
        Returns:
            Đường dẫn đến file đã lưu
        """
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crypto_sentiment_{timestamp}.csv"
        
        if not filename.endswith('.csv'):
            filename += '.csv'
        
        file_path = self.data_dir / filename
        
        try:
            # Chuyển đổi thành DataFrame
            df = self.convert_to_dataframe(data)
            
            # Lưu vào CSV
            df.to_csv(file_path, index=False)
            
            self.logger.info(f"Đã lưu {len(data)} bản ghi tâm lý vào {file_path}")
            return str(file_path)
        
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu dữ liệu tâm lý vào file CSV: {str(e)}")
            return ""
    
    def save_to_file(self, data_list, filename_prefix, format='csv'):
        """
        Lưu danh sách dữ liệu tâm lý vào file với định dạng chỉ định.
        
        Args:
            data_list: Danh sách các đối tượng SentimentData
            filename_prefix: Tiền tố tên file
            format: Định dạng file ('csv', 'json', 'parquet')
            
        Returns:
            Path: Đường dẫn đến file đã lưu
        """
        if not data_list:
            return None
            
        output_dir = self.data_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Chuyển đổi danh sách đối tượng thành danh sách dicts
        data_dicts = [item.to_dict() for item in data_list]
        
        if format.lower() == 'json':
            file_path = output_dir / f"{filename_prefix}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data_dicts, f, ensure_ascii=False, indent=2)
        elif format.lower() == 'parquet':
            file_path = output_dir / f"{filename_prefix}.parquet"
            df = pd.DataFrame(data_dicts)
            df.to_parquet(file_path)
        else:  # csv
            file_path = output_dir / f"{filename_prefix}.csv"
            self.save_to_csv(data_list, file_path)
            
        return file_path

    def load_from_json(self, file_path: Union[str, Path]) -> List[SentimentData]:
        """
        Tải dữ liệu tâm lý từ file JSON.
        
        Args:
            file_path: Đường dẫn đến file JSON
            
        Returns:
            Danh sách dữ liệu tâm lý
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            # Chuyển đổi từ dict sang đối tượng SentimentData
            data = [SentimentData.from_dict(item) for item in json_data]
            
            self.logger.info(f"Đã tải {len(data)} bản ghi tâm lý từ {file_path}")
            return data
        
        except Exception as e:
            self.logger.error(f"Lỗi khi tải dữ liệu tâm lý từ file JSON: {str(e)}")
            return []
    
    async def collect_and_save(self, filename: Optional[str] = None, format: str = "json", **kwargs) -> str:
        """
        Thu thập dữ liệu tâm lý từ tất cả các nguồn và lưu vào file.
        
        Args:
            filename: Tên file (tùy chọn)
            format: Định dạng file ('json' hoặc 'csv')
            **kwargs: Tham số bổ sung cho các nguồn
            
        Returns:
            Đường dẫn đến file đã lưu
        """
        results = await self.collect_from_all_sources(**kwargs)
        
        # Ghép tất cả dữ liệu thành một danh sách
        all_data = []
        for source_data in results.values():
            all_data.extend(source_data)
        
        # Sắp xếp theo thời gian từ mới đến cũ
        sorted_data = sorted(
            all_data,
            key=lambda x: x.timestamp,
            reverse=True
        )
        
        # Lưu theo định dạng tương ứng
        if format.lower() == 'csv':
            return self.save_to_csv(sorted_data, filename)
        else:
            return self.save_to_json(sorted_data, filename)
    
    async def collect_news_sentiment(self, asset: Optional[str] = None, days: int = 30, 
                                   output_file: Optional[str] = None) -> str:
        """
        Thu thập và lưu dữ liệu tâm lý từ tin tức.
        
        Args:
            asset: Mã tài sản (tùy chọn)
            days: Số ngày lịch sử
            output_file: Tên file đầu ra (tùy chọn)
            
        Returns:
            Đường dẫn đến file đã lưu
        """
        # Chuẩn bị tham số
        params = {
            "days": days
        }
        
        if asset:
            params["asset"] = asset
        
        # Thu thập dữ liệu tâm lý tin tức
        results = await self.collect_specific_sentiment("news", **params)
        
        # Xác định tên file nếu không được chỉ định
        if not output_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if asset:
                output_file = f"news_sentiment_{asset}_{timestamp}.csv"
            else:
                output_file = f"news_sentiment_{timestamp}.csv"
        
        # Lưu kết quả
        if not results:
            self.logger.warning("Không có dữ liệu tâm lý tin tức để lưu")
            return ""
        
        return self.save_to_csv(results, output_file)
    
    async def collect_social_sentiment(self, asset: Optional[str] = None, days: int = 30, 
                                     platforms: Optional[List[str]] = None,
                                     output_file: Optional[str] = None) -> str:
        """
        Thu thập và lưu dữ liệu tâm lý từ mạng xã hội.
        
        Args:
            asset: Mã tài sản (tùy chọn)
            days: Số ngày lịch sử
            platforms: Danh sách nền tảng mạng xã hội ('twitter', 'reddit', 'telegram')
            output_file: Tên file đầu ra (tùy chọn)
            
        Returns:
            Đường dẫn đến file đã lưu
        """
        # Chuẩn bị tham số
        params = {
            "days": days
        }
        
        if asset:
            params["asset"] = asset
        
        # Thu thập dữ liệu tâm lý mạng xã hội
        results = await self.collect_specific_sentiment("social", **params)
        
        # Lọc theo nền tảng nếu được chỉ định
        if platforms and results:
            filtered_results = []
            for item in results:
                source_lower = item.source.lower()
                if any(platform.lower() in source_lower for platform in platforms):
                    filtered_results.append(item)
            
            results = filtered_results
        
        # Xác định tên file nếu không được chỉ định
        if not output_file:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            if asset:
                output_file = f"social_sentiment_{asset}_{timestamp}.csv"
            else:
                output_file = f"social_sentiment_{timestamp}.csv"
        
        # Lưu kết quả
        if not results:
            self.logger.warning("Không có dữ liệu tâm lý mạng xã hội để lưu")
            return ""
        
        return self.save_to_csv(results, output_file)
    
    def convert_to_dataframe(self, data: List[SentimentData]) -> pd.DataFrame:
        """
        Chuyển đổi danh sách dữ liệu tâm lý thành DataFrame.
        
        Args:
            data: Danh sách dữ liệu tâm lý
            
        Returns:
            DataFrame chứa dữ liệu tâm lý
        """
        # Chuyển đổi sang dạng từ điển
        records = []
        for item in data:
            record = item.to_dict()
            
            # Lấy các trường metadata phổ biến
            if item.metadata:
                for key, value in item.metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        record[f"metadata_{key}"] = value
            
            records.append(record)
        
        # Tạo DataFrame
        return pd.DataFrame(records)
    
    def get_aggregated_sentiment(self, data: List[SentimentData], 
                                group_by: str = 'source',
                                timeframe: Optional[str] = None) -> pd.DataFrame:
        """
        Tính toán tâm lý tổng hợp từ nhiều nguồn.
        
        Args:
            data: Danh sách dữ liệu tâm lý
            group_by: Trường để nhóm dữ liệu (source, asset, timeframe)
            timeframe: Khoảng thời gian để nhóm dữ liệu
            
        Returns:
            DataFrame chứa dữ liệu tâm lý tổng hợp
        """
        df = self.convert_to_dataframe(data)
        
        # Chuyển timestamp sang datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Nhóm theo timeframe nếu có
        if timeframe:
            df['timestamp_group'] = df['timestamp'].dt.floor(timeframe)
            
            # Nhóm dữ liệu
            grouped = df.groupby([group_by, 'timestamp_group'])
        else:
            # Chỉ nhóm theo nguồn
            grouped = df.groupby(group_by)
        
        # Tính toán các chỉ số tổng hợp
        aggregated = grouped.agg({
            'value': ['mean', 'min', 'max', 'std'],
            'asset': lambda x: x.iloc[0] if x.iloc[0] else None,
            'timestamp': 'max'  # Lấy timestamp mới nhất
        })
        
        # Làm phẳng cấu trúc đa cấp
        aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
        aggregated = aggregated.reset_index()
        
        # Thêm nhãn tổng hợp dựa trên value_mean
        def get_label(value):
            if value <= -0.6:
                return "Very Bearish"
            elif value <= -0.2:
                return "Bearish"
            elif value <= 0.2:
                return "Neutral"
            elif value <= 0.6:
                return "Bullish"
            else:
                return "Very Bullish"
        
        aggregated['label'] = aggregated['value_mean'].apply(get_label)
        
        return aggregated
    
    def get_latest_sentiment(self, n: int = 10) -> List[SentimentData]:
        """
        Lấy n bản ghi tâm lý mới nhất từ tất cả các file JSON trong thư mục dữ liệu.
        
        Args:
            n: Số lượng bản ghi muốn lấy
            
        Returns:
            Danh sách bản ghi tâm lý mới nhất
        """
        all_data = []
        
        # Tìm tất cả các file JSON trong thư mục dữ liệu
        json_files = list(self.data_dir.glob("*.json"))
        
        # Sắp xếp theo thời gian tạo file (mới nhất trước)
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Đọc các file cho đến khi đủ n bản ghi
        for file_path in json_files:
            if len(all_data) >= n:
                break
                
            try:
                data = self.load_from_json(file_path)
                all_data.extend(data)
            except Exception as e:
                self.logger.error(f"Lỗi khi đọc file {file_path}: {str(e)}")
        
        # Sắp xếp theo thời gian và lấy n bản ghi mới nhất
        all_data.sort(key=lambda x: x.timestamp, reverse=True)
        
        return all_data[:n]
    
    def get_sentiment_by_asset(self, asset: str, data: Optional[List[SentimentData]] = None) -> List[SentimentData]:
        """
        Lọc dữ liệu tâm lý theo tài sản.
        
        Args:
            asset: Mã tài sản cần lọc
            data: Danh sách dữ liệu tâm lý (tùy chọn, nếu None sẽ tải từ file)
            
        Returns:
            Danh sách dữ liệu tâm lý cho tài sản cụ thể
        """
        # Nếu không có dữ liệu, tải từ tất cả các file
        if data is None:
            data = self.get_latest_sentiment(1000)  # Giới hạn một số lượng hợp lý
        
        # Lọc theo tài sản
        asset = asset.upper()
        return [item for item in data if item.asset == asset]
    
    def plot_sentiment_trends(self, data: List[SentimentData], 
                             source_filter: Optional[str] = None,
                             asset_filter: Optional[str] = None,
                             days: int = 30) -> Optional["Figure"]:
        """
        Tạo biểu đồ xu hướng tâm lý theo thời gian.
        
        Args:
            data: Danh sách dữ liệu tâm lý
            source_filter: Lọc theo nguồn (tùy chọn)
            asset_filter: Lọc theo tài sản (tùy chọn)
            days: Số ngày hiển thị
            
        Returns:
            Đối tượng Figure matplotlib hoặc None nếu không có dữ liệu
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            import matplotlib.dates as mdates
            
            # Lọc dữ liệu
            filtered_data = data
            
            if source_filter:
                filtered_data = [item for item in filtered_data if source_filter in item.source]
                
            if asset_filter:
                filtered_data = [item for item in filtered_data if item.asset == asset_filter]
            
            # Chuyển thành DataFrame
            df = self.convert_to_dataframe(filtered_data)
            
            if df.empty:
                return None
            
            # Chuyển timestamp sang datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Lọc theo số ngày
            min_date = datetime.datetime.now() - datetime.timedelta(days=days)
            df = df[df['timestamp'] >= min_date]
            
            # Nhóm theo nguồn và ngày
            df['date'] = df['timestamp'].dt.date
            grouped = df.groupby(['source', 'date'])['value'].mean().reset_index()
            
            # Tạo biểu đồ
            fig, ax = plt.subplots(figsize=(12, 6))
            
            for source, group in grouped.groupby('source'):
                ax.plot(group['date'], group['value'], 'o-', label=source)
            
            # Định dạng biểu đồ
            ax.set_title('Xu hướng tâm lý thị trường theo thời gian')
            ax.set_xlabel('Ngày')
            ax.set_ylabel('Điểm tâm lý')
            ax.grid(True, alpha=0.3)
            
            # Định dạng trục ngày
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
            
            # Thêm chú thích
            ax.legend(loc='best')
            
            # Thêm đường tham chiếu
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
            ax.axhline(y=0.6, color='g', linestyle='--', alpha=0.2)
            ax.axhline(y=-0.6, color='r', linestyle='--', alpha=0.2)
            
            # Thêm nhãn tâm lý
            ylim = ax.get_ylim()
            ax.text(min_date, 0.8, 'Very Bullish', color='g', fontsize=8)
            ax.text(min_date, 0, 'Neutral', color='k', fontsize=8)
            ax.text(min_date, -0.8, 'Very Bearish', color='r', fontsize=8)
            
            plt.tight_layout()
            return fig
        
        except ImportError:
            self.logger.warning("Thư viện matplotlib không được cài đặt. Không thể tạo biểu đồ.")
            return None
        except Exception as e:
            self.logger.error(f"Lỗi khi tạo biểu đồ: {str(e)}")
            return None


# Phương thức chạy chính
async def main():
    """Hàm test chạy trực tiếp module."""
    collector = SentimentCollector()
    
    # Thu thập dữ liệu từ tất cả các nguồn
    print("Đang thu thập dữ liệu tâm lý thị trường...")
    results = await collector.collect_from_all_sources(asset="BTC")
    
    # In số lượng bản ghi từ mỗi nguồn
    for source_name, data in results.items():
        print(f"{source_name}: {len(data)} bản ghi tâm lý")
    
    # Lưu các bản ghi vào file
    all_data = []
    for data in results.values():
        all_data.extend(data)
    
    if all_data:
        file_path = collector.save_to_json(all_data)
        print(f"Đã lưu tất cả dữ liệu tâm lý vào {file_path}")
        
        # Chuyển đổi sang DataFrame
        df = collector.convert_to_dataframe(all_data)
        print(f"Tổng cộng: {len(df)} bản ghi trong DataFrame")
        print(df[['source', 'value', 'label', 'asset', 'timestamp']].head())
        
        # Tính toán tâm lý tổng hợp
        agg_df = collector.get_aggregated_sentiment(all_data)
        print("\nTâm lý tổng hợp theo nguồn:")
        print(agg_df[['source', 'value_mean', 'label', 'timestamp_max']].head())

# Cách khởi chạy khác nhau tùy thuộc vào hệ điều hành
if __name__ == "__main__":
    if IS_WINDOWS:
        # Trên Windows, sử dụng asyncio.run trong ProactorEventLoopPolicy
        # để tránh các vấn đề với SelectorEventLoop
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Chạy hàm main bất đồng bộ
    asyncio.run(main())
"""
Thu thập dữ liệu tin tức tiền điện tử.
File này cung cấp các lớp và phương thức để thu thập tin tức
từ các nguồn trực tuyến về thị trường tiền điện tử.
"""

import os
import re
import json
import time
import logging
import asyncio
import aiohttp
import hashlib
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from urllib.parse import urlparse, quote_plus
from bs4 import BeautifulSoup
import requests
from requests.exceptions import RequestException
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import các module nội bộ
from config.env import get_env
from config.logging_config import get_logger
from config.utils.validators import is_valid_url

# Tạo logger cho module
logger = get_logger("news_collector")

class CryptoNewsSource:
    """Lớp cơ sở định nghĩa nguồn tin tức tiền điện tử."""
    
    def __init__(self, name: str, base_url: str, 
                language: str = "en", logo_url: Optional[str] = None,
                categories: Optional[List[str]] = None):
        """
        Khởi tạo nguồn tin tức.
        
        Args:
            name: Tên nguồn tin
            base_url: URL cơ sở của trang web tin tức
            language: Ngôn ngữ chính của nguồn tin (mặc định: en)
            logo_url: URL logo của nguồn tin (tùy chọn)
            categories: Danh sách các category tin tức (tùy chọn)
        """
        self.name = name
        self.base_url = base_url
        self.language = language
        self.logo_url = logo_url
        self.categories = categories if categories else []
        self.session = None
    
    async def initialize_session(self) -> None:
        """Khởi tạo phiên HTTP."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
    
    async def close_session(self) -> None:
        """Đóng phiên HTTP."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def fetch_url(self, url: str) -> Optional[str]:
        """
        Tải nội dung từ URL sử dụng aiohttp.
        
        Args:
            url: URL cần tải nội dung
        
        Returns:
            Nội dung trang web hoặc None nếu có lỗi
        """
        await self.initialize_session()
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        
        try:
            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.warning(f"Không tải được URL {url}, mã trạng thái: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Lỗi khi tải URL {url}: {str(e)}")
            return None
    
    async def parse_article_list(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Phân tích HTML để lấy danh sách bài viết.
        
        Args:
            html_content: Nội dung HTML của trang danh sách bài viết
        
        Returns:
            Danh sách các bài viết với các trường cơ bản
        """
        # Phương thức cần được ghi đè trong lớp con
        raise NotImplementedError("Phương thức này phải được triển khai trong lớp con")
    
    async def parse_article_content(self, html_content: str, url: str) -> Dict[str, Any]:
        """
        Phân tích HTML để lấy nội dung đầy đủ của bài viết.
        
        Args:
            html_content: Nội dung HTML của trang bài viết
            url: URL của bài viết
        
        Returns:
            Thông tin chi tiết của bài viết
        """
        # Phương thức cần được ghi đè trong lớp con
        raise NotImplementedError("Phương thức này phải được triển khai trong lớp con")
    
    async def fetch_latest_articles(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Lấy các bài viết mới nhất từ nguồn tin.
        
        Args:
            limit: Số lượng bài viết tối đa cần lấy
        
        Returns:
            Danh sách các bài viết mới nhất
        """
        # Phương thức cần được ghi đè trong lớp con
        raise NotImplementedError("Phương thức này phải được triển khai trong lớp con")
    
    def generate_article_id(self, url: str, title: str) -> str:
        """
        Tạo ID duy nhất cho bài viết từ URL và tiêu đề.
        
        Args:
            url: URL của bài viết
            title: Tiêu đề bài viết
        
        Returns:
            ID duy nhất dạng chuỗi
        """
        # Tạo chuỗi kết hợp
        combined = f"{url}|{title}"
        # Tạo hash MD5
        return hashlib.md5(combined.encode()).hexdigest()
    
    def extract_publish_date(self, date_str: str) -> Optional[datetime.datetime]:
        """
        Chuyển đổi chuỗi ngày thành đối tượng datetime.
        
        Args:
            date_str: Chuỗi ngày từ bài viết
        
        Returns:
            Đối tượng datetime hoặc None nếu không thể phân tích
        """
        formats = [
            "%Y-%m-%dT%H:%M:%S%z",      # ISO 8601 với timezone
            "%Y-%m-%dT%H:%M:%S.%f%z",   # ISO 8601 với timezone và microseconds
            "%Y-%m-%d %H:%M:%S",        # Format chuẩn không timezone
            "%B %d, %Y %H:%M:%S",       # Ví dụ: "January 1, 2022 12:00:00"
            "%d %B %Y %H:%M",           # Ví dụ: "1 January 2022 12:00"
            "%Y-%m-%d",                 # Chỉ ngày
            "%d/%m/%Y",                 # Định dạng ngày/tháng/năm
            "%m/%d/%Y",                 # Định dạng tháng/ngày/năm
        ]
        
        for fmt in formats:
            try:
                return datetime.datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        # Thử phân tích định dạng "X hours/minutes ago"
        try:
            now = datetime.datetime.now()
            if "hour" in date_str.lower():
                hours = int(re.search(r'(\d+)\s+hour', date_str.lower()).group(1))
                return now - datetime.timedelta(hours=hours)
            elif "minute" in date_str.lower():
                minutes = int(re.search(r'(\d+)\s+minute', date_str.lower()).group(1))
                return now - datetime.timedelta(minutes=minutes)
            elif "day" in date_str.lower():
                days = int(re.search(r'(\d+)\s+day', date_str.lower()).group(1))
                return now - datetime.timedelta(days=days)
        except (AttributeError, ValueError):
            pass
        
        logger.warning(f"Không thể phân tích chuỗi ngày: {date_str}")
        return None


class CoinDeskScraper(CryptoNewsSource):
    """Lớp thu thập tin tức từ CoinDesk."""
    
    def __init__(self):
        """Khởi tạo scraper CoinDesk."""
        super().__init__(
            name="CoinDesk",
            base_url="https://www.coindesk.com",
            language="en",
            logo_url="https://www.coindesk.com/pf/resources/images/logo-full-black.svg",
            categories=["bitcoin", "ethereum", "policy", "business", "markets"]
        )
    
    async def parse_article_list(self, html_content: str) -> List[Dict[str, Any]]:
        """
        Phân tích trang danh sách bài viết CoinDesk.
        
        Args:
            html_content: Nội dung HTML
            
        Returns:
            Danh sách các bài viết
        """
        articles = []
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Tìm các thẻ article trong trang
        article_elements = soup.find_all('article')
        
        for article in article_elements:
            try:
                # Tìm liên kết và tiêu đề
                link_element = article.find('a', href=True)
                if not link_element:
                    continue
                
                link = link_element.get('href', '')
                if not link.startswith('http'):
                    link = self.base_url + link
                
                # Lấy tiêu đề
                title_element = article.find('h4') or article.find('h3') or article.find('h2')
                if not title_element:
                    continue
                
                title = title_element.text.strip()
                
                # Tìm description nếu có
                description_element = article.find('p')
                description = description_element.text.strip() if description_element else ""
                
                # Tìm thời gian xuất bản nếu có
                timestamp_element = article.find('time')
                published_at = None
                if timestamp_element:
                    published_at = timestamp_element.get('datetime') or timestamp_element.text.strip()
                    if published_at:
                        published_at = self.extract_publish_date(published_at)
                
                # Lấy hình ảnh nếu có
                img_element = article.find('img', src=True)
                image_url = img_element.get('src', '') if img_element else None
                
                # Tạo bản ghi bài viết
                article_data = {
                    'source': self.name,
                    'title': title,
                    'url': link,
                    'description': description,
                    'image_url': image_url,
                    'published_at': published_at.isoformat() if published_at else None,
                    'collected_at': datetime.datetime.now().isoformat(),
                    'article_id': self.generate_article_id(link, title),
                    'category': None,  # Sẽ được điền sau
                    'content': None,  # Sẽ được điền sau khi tải nội dung đầy đủ
                }
                
                articles.append(article_data)
                
            except Exception as e:
                logger.error(f"Lỗi khi phân tích bài viết CoinDesk: {str(e)}")
        
        return articles
    
    async def parse_article_content(self, html_content: str, url: str) -> Dict[str, Any]:
        """
        Phân tích trang chi tiết bài viết CoinDesk.
        
        Args:
            html_content: Nội dung HTML
            url: URL của bài viết
            
        Returns:
            Thông tin chi tiết của bài viết
        """
        soup = BeautifulSoup(html_content, 'html.parser')
        
        try:
            # Lấy tiêu đề
            title_element = soup.find('h1')
            title = title_element.text.strip() if title_element else ""
            
            # Lấy mô tả/tóm tắt
            description_element = soup.find('meta', {'name': 'description'}) or soup.find('meta', {'property': 'og:description'})
            description = description_element.get('content', '') if description_element else ""
            
            # Lấy thời gian xuất bản
            time_element = soup.find('time')
            published_at = None
            if time_element:
                datetime_str = time_element.get('datetime') or time_element.text.strip()
                published_at = self.extract_publish_date(datetime_str)
            
            # Lấy tác giả
            author_element = soup.find('a', {'rel': 'author'}) or soup.find('meta', {'name': 'author'})
            author = ""
            if author_element:
                if author_element.name == 'a':
                    author = author_element.text.strip()
                else:
                    author = author_element.get('content', '')
            
            # Lấy hình ảnh
            image_element = soup.find('meta', {'property': 'og:image'})
            image_url = image_element.get('content', '') if image_element else None
            
            # Lấy nội dung chính của bài viết
            content_elements = soup.find('div', class_=['article-content', 'entry-content', 'post-content'])
            
            # Nếu không tìm thấy container cụ thể, tìm tất cả các thẻ p trong phần thân
            if not content_elements:
                content_elements = soup.find('article')
            
            content = ""
            if content_elements:
                # Lấy text từ tất cả thẻ p
                paragraphs = content_elements.find_all('p')
                content = "\n\n".join([p.text.strip() for p in paragraphs])
            
            # Lấy tags/categories nếu có
            tags = []
            tags_elements = soup.find_all('a', href=lambda href: href and '/tag/' in href)
            if tags_elements:
                tags = [tag.text.strip() for tag in tags_elements]
            
            # Xác định category từ URL
            category = None
            for cat in self.categories:
                if f"/{cat}/" in url:
                    category = cat
                    break
            
            # Tạo ID duy nhất
            article_id = self.generate_article_id(url, title)
            
            # Tạo đối tượng bài viết đầy đủ
            article = {
                'article_id': article_id,
                'source': self.name,
                'title': title,
                'url': url,
                'description': description,
                'content': content,
                'published_at': published_at.isoformat() if published_at else None,
                'author': author,
                'image_url': image_url,
                'category': category,
                'tags': tags,
                'collected_at': datetime.datetime.now().isoformat(),
            }
            
            return article
            
        except Exception as e:
            logger.error(f"Lỗi khi phân tích nội dung bài viết CoinDesk {url}: {str(e)}")
            return {
                'article_id': self.generate_article_id(url, ""),
                'source': self.name,
                'url': url,
                'error': str(e),
                'collected_at': datetime.datetime.now().isoformat(),
            }
    
    async def fetch_latest_articles(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Lấy các bài viết mới nhất từ CoinDesk.
        
        Args:
            limit: Số lượng bài viết tối đa cần lấy
        
        Returns:
            Danh sách các bài viết mới nhất
        """
        latest_url = f"{self.base_url}/news/"
        
        try:
            # Tải trang tin tức mới nhất
            html_content = await self.fetch_url(latest_url)
            if not html_content:
                logger.error(f"Không thể tải trang tin tức từ {latest_url}")
                return []
            
            # Phân tích danh sách bài viết
            articles = await self.parse_article_list(html_content)
            
            # Giới hạn số lượng bài viết
            articles = articles[:limit]
            
            # Tải và phân tích nội dung đầy đủ cho mỗi bài viết
            full_articles = []
            for article in articles:
                try:
                    # Tải nội dung bài viết
                    article_html = await self.fetch_url(article['url'])
                    if article_html:
                        # Phân tích nội dung đầy đủ
                        full_article = await self.parse_article_content(article_html, article['url'])
                        full_articles.append(full_article)
                    else:
                        # Nếu không tải được, sử dụng thông tin cơ bản
                        article['error'] = "Không thể tải nội dung bài viết"
                        full_articles.append(article)
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý bài viết {article['url']}: {str(e)}")
                    article['error'] = str(e)
                    full_articles.append(article)
                
                # Chờ một chút để tránh gửi quá nhiều request
                await asyncio.sleep(1)
            
            return full_articles
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy bài viết mới nhất từ CoinDesk: {str(e)}")
            return []


class CryptoPanicScraper(CryptoNewsSource):
    """Lớp thu thập tin tức từ Crypto Panic."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Khởi tạo scraper Crypto Panic.
        
        Args:
            api_key: API key cho Crypto Panic (tùy chọn)
        """
        super().__init__(
            name="CryptoPanic",
            base_url="https://cryptopanic.com",
            language="en",
            logo_url="https://cryptopanic.com/static/images/cryptopanic-logo.png",
            categories=["news", "media", "hot", "rising"]
        )
        self.api_key = api_key or get_env("CRYPTOPANIC_API_KEY", "")
        self.api_url = "https://cryptopanic.com/api/v1/posts/"
    
    async def fetch_latest_articles(self, limit: int = 50, currencies: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Lấy các bài viết mới nhất từ Crypto Panic API.
        
        Args:
            limit: Số lượng bài viết tối đa cần lấy
            currencies: Danh sách các mã tiền điện tử cần lọc
            
        Returns:
            Danh sách các bài viết mới nhất
        """
        await self.initialize_session()
        
        # Chuẩn bị tham số
        params = {
            'limit': min(limit, 100),  # API chỉ cho phép tối đa 100 bài viết mỗi lần
            'public': 'true'
        }
        
        if self.api_key:
            params['auth_token'] = self.api_key
        
        if currencies:
            params['currencies'] = ','.join(currencies)
        
        try:
            async with self.session.get(self.api_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = data.get('results', [])
                    
                    articles = []
                    for item in results:
                        try:
                            # Trích xuất thông tin từ JSON
                            title = item.get('title', '')
                            url = item.get('url', '')
                            source = item.get('source', {}).get('title', 'Unknown')
                            domain = item.get('domain', '')
                            published_at = item.get('published_at')
                            
                            # Xác định currencies được đề cập
                            currencies_mentioned = []
                            for currency in item.get('currencies', []):
                                currencies_mentioned.append({
                                    'code': currency.get('code', ''),
                                    'title': currency.get('title', ''),
                                    'slug': currency.get('slug', '')
                                })
                            
                            # Lấy sentiment nếu có
                            votes = item.get('votes', {})
                            sentiment = {
                                'positive': votes.get('positive', 0),
                                'negative': votes.get('negative', 0),
                                'important': votes.get('important', 0),
                                'liked': votes.get('liked', 0),
                                'disliked': votes.get('disliked', 0),
                                'lol': votes.get('lol', 0),
                                'toxic': votes.get('toxic', 0),
                                'comments': votes.get('comments', 0),
                            }
                            
                            # Tạo bản ghi bài viết
                            article = {
                                'article_id': item.get('id', self.generate_article_id(url, title)),
                                'source': f"{source} via CryptoPanic",
                                'aggregator': self.name,
                                'title': title,
                                'url': url,
                                'domain': domain,
                                'published_at': published_at,
                                'collected_at': datetime.datetime.now().isoformat(),
                                'currencies': currencies_mentioned,
                                'sentiment': sentiment,
                                'category': item.get('kind', None),
                                'description': "",  # CryptoPanic API không cung cấp mô tả
                                'content': None,    # Không có nội dung đầy đủ
                            }
                            
                            articles.append(article)
                            
                        except Exception as e:
                            logger.error(f"Lỗi khi xử lý bài viết từ CryptoPanic: {str(e)}")
                    
                    return articles
                else:
                    logger.error(f"Lỗi khi truy cập CryptoPanic API: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Lỗi khi lấy dữ liệu từ CryptoPanic API: {str(e)}")
            return []


class RSSFeedScraper(CryptoNewsSource):
    """Lớp thu thập tin tức từ nguồn RSS Feed."""
    
    def __init__(self, name: str, feed_url: str, base_url: str, 
                 language: str = "en", logo_url: Optional[str] = None):
        """
        Khởi tạo scraper RSS Feed.
        
        Args:
            name: Tên nguồn tin
            feed_url: URL của RSS feed
            base_url: URL cơ sở của trang web
            language: Ngôn ngữ (mặc định: en)
            logo_url: URL logo (tùy chọn)
        """
        super().__init__(
            name=name,
            base_url=base_url,
            language=language,
            logo_url=logo_url
        )
        self.feed_url = feed_url
    
    async def fetch_latest_articles(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Lấy các bài viết mới nhất từ RSS Feed.
        
        Args:
            limit: Số lượng bài viết tối đa cần lấy
            
        Returns:
            Danh sách các bài viết mới nhất
        """
        import feedparser
        
        try:
            # Tải RSS feed
            response = requests.get(self.feed_url, timeout=30)
            if response.status_code != 200:
                logger.error(f"Không thể tải RSS feed từ {self.feed_url}, mã trạng thái: {response.status_code}")
                return []
            
            # Phân tích RSS feed
            feed = feedparser.parse(response.content)
            
            articles = []
            for entry in feed.entries[:limit]:
                try:
                    # Trích xuất thông tin từ entry
                    title = entry.get('title', '')
                    url = entry.get('link', '')
                    
                    # Lấy mô tả
                    description = ""
                    if 'summary' in entry:
                        description = entry.summary
                    elif 'description' in entry:
                        description = entry.description
                    
                    # Xử lý HTML trong mô tả
                    if description:
                        soup = BeautifulSoup(description, 'html.parser')
                        description = soup.get_text()
                    
                    # Lấy thời gian xuất bản
                    published_at = None
                    if 'published' in entry:
                        published_at = self.extract_publish_date(entry.published)
                    elif 'pubDate' in entry:
                        published_at = self.extract_publish_date(entry.pubDate)
                    elif 'updated' in entry:
                        published_at = self.extract_publish_date(entry.updated)
                    
                    # Tìm hình ảnh nếu có
                    image_url = None
                    if 'media_content' in entry and entry.media_content:
                        for media in entry.media_content:
                            if 'url' in media:
                                image_url = media['url']
                                break
                    
                    # Lấy nội dung nếu có
                    content = ""
                    if 'content' in entry and entry.content:
                        for content_item in entry.content:
                            if 'value' in content_item:
                                content_soup = BeautifulSoup(content_item.value, 'html.parser')
                                content = content_soup.get_text()
                                break
                    
                    # Lấy tác giả nếu có
                    author = ""
                    if 'author' in entry:
                        author = entry.author
                    elif 'creator' in entry:
                        author = entry.creator
                    
                    # Lấy tags nếu có
                    tags = []
                    if 'tags' in entry:
                        for tag in entry.tags:
                            if 'term' in tag:
                                tags.append(tag.term)
                    
                    # Tạo ID duy nhất
                    article_id = self.generate_article_id(url, title)
                    
                    # Tạo bản ghi bài viết
                    article = {
                        'article_id': article_id,
                        'source': self.name,
                        'title': title,
                        'url': url,
                        'description': description,
                        'content': content,
                        'published_at': published_at.isoformat() if published_at else None,
                        'author': author,
                        'image_url': image_url,
                        'tags': tags,
                        'collected_at': datetime.datetime.now().isoformat(),
                    }
                    
                    articles.append(article)
                    
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý bài viết từ RSS feed {self.name}: {str(e)}")
            
            return articles
            
        except Exception as e:
            logger.error(f"Lỗi khi lấy bài viết từ RSS feed {self.name}: {str(e)}")
            return []


class CryptoNewsCollector:
    """Lớp chính để thu thập và quản lý dữ liệu tin tức tiền điện tử từ nhiều nguồn."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Khởi tạo collector tin tức.
        
        Args:
            data_dir: Thư mục lưu trữ dữ liệu (tùy chọn)
        """
        self.sources = {}
        self.data_dir = data_dir
        
        if self.data_dir is None:
            # Sử dụng thư mục mặc định nếu không được chỉ định
            from config.system_config import BASE_DIR
            self.data_dir = BASE_DIR / "data" / "news"
        
        # Tạo thư mục nếu chưa tồn tại
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Tạo logger
        self.logger = get_logger("news_collector")
        
        # Khởi tạo các nguồn tin tức
        self._initialize_sources()
    
    def _initialize_sources(self) -> None:
        """Khởi tạo các nguồn tin tức mặc định."""
        # Thêm CoinDesk
        self.add_source(CoinDeskScraper())
        
        # Thêm CryptoPanic nếu có API key
        api_key = get_env("CRYPTOPANIC_API_KEY", "")
        if api_key:
            self.add_source(CryptoPanicScraper(api_key=api_key))
        
        # Thêm một số nguồn RSS
        self.add_source(RSSFeedScraper(
            name="Cointelegraph",
            feed_url="https://cointelegraph.com/rss",
            base_url="https://cointelegraph.com",
            logo_url="https://cointelegraph.com/assets/img/logo.svg"
        ))
        
        self.add_source(RSSFeedScraper(
            name="Bitcoin Magazine",
            feed_url="https://bitcoinmagazine.com/feed",
            base_url="https://bitcoinmagazine.com",
            logo_url="https://bitcoinmagazine.com/static/img/brand/logo.svg"
        ))
    
    def add_source(self, source: CryptoNewsSource) -> None:
        """
        Thêm nguồn tin tức vào collector.
        
        Args:
            source: Đối tượng nguồn tin tức
        """
        self.sources[source.name] = source
        self.logger.info(f"Đã thêm nguồn tin tức: {source.name}")
    
    def remove_source(self, source_name: str) -> bool:
        """
        Xóa nguồn tin tức khỏi collector.
        
        Args:
            source_name: Tên nguồn tin tức
            
        Returns:
            True nếu xóa thành công, False nếu không tìm thấy
        """
        if source_name in self.sources:
            del self.sources[source_name]
            self.logger.info(f"Đã xóa nguồn tin tức: {source_name}")
            return True
        return False
    
    def get_sources(self) -> Dict[str, CryptoNewsSource]:
        """
        Lấy danh sách các nguồn tin tức.
        
        Returns:
            Từ điển các nguồn tin tức
        """
        return self.sources
    
    async def collect_from_source(self, source_name: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Thu thập tin tức từ một nguồn cụ thể.
        
        Args:
            source_name: Tên nguồn tin tức
            limit: Số lượng bài viết tối đa
            
        Returns:
            Danh sách các bài viết thu thập được
        """
        if source_name not in self.sources:
            self.logger.error(f"Không tìm thấy nguồn tin tức: {source_name}")
            return []
        
        source = self.sources[source_name]
        try:
            articles = await source.fetch_latest_articles(limit=limit)
            self.logger.info(f"Đã thu thập {len(articles)} bài viết từ {source_name}")
            return articles
        except Exception as e:
            self.logger.error(f"Lỗi khi thu thập tin tức từ {source_name}: {str(e)}")
            return []
        finally:
            # Đảm bảo đóng session
            await source.close_session()
    
    async def collect_from_all_sources(self, limit_per_source: int = 20) -> Dict[str, List[Dict[str, Any]]]:
        """
        Thu thập tin tức từ tất cả các nguồn.
        
        Args:
            limit_per_source: Số lượng bài viết tối đa từ mỗi nguồn
            
        Returns:
            Từ điển với khóa là tên nguồn và giá trị là danh sách bài viết
        """
        results = {}
        tasks = []
        
        for source_name in self.sources:
            task = self.collect_from_source(source_name, limit=limit_per_source)
            tasks.append((source_name, task))
        
        for source_name, task in tasks:
            try:
                articles = await task
                results[source_name] = articles
            except Exception as e:
                self.logger.error(f"Lỗi khi thu thập tin tức từ {source_name}: {str(e)}")
                results[source_name] = []
        
        return results
    
    def save_articles_to_json(self, articles: List[Dict[str, Any]], filename: Optional[str] = None) -> str:
        """
        Lưu danh sách bài viết vào file JSON.
        
        Args:
            articles: Danh sách bài viết cần lưu
            filename: Tên file (tùy chọn)
            
        Returns:
            Đường dẫn đến file đã lưu
        """
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"crypto_news_{timestamp}.json"
        
        file_path = self.data_dir / filename
        
        try:
            # Chuyển đổi datetime objects thành strings
            serializable_articles = []
            for article in articles:
                article_copy = dict(article)
                
                # Xử lý trường published_at nếu là đối tượng datetime
                if 'published_at' in article_copy and isinstance(article_copy['published_at'], datetime.datetime):
                    article_copy['published_at'] = article_copy['published_at'].isoformat()
                
                serializable_articles.append(article_copy)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_articles, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Đã lưu {len(articles)} bài viết vào {file_path}")
            return str(file_path)
        
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu bài viết vào file JSON: {str(e)}")
            return ""
    
    def load_articles_from_json(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Tải danh sách bài viết từ file JSON.
        
        Args:
            file_path: Đường dẫn đến file JSON
            
        Returns:
            Danh sách bài viết
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            
            self.logger.info(f"Đã tải {len(articles)} bài viết từ {file_path}")
            return articles
        
        except Exception as e:
            self.logger.error(f"Lỗi khi tải bài viết từ file JSON: {str(e)}")
            return []
    
    async def collect_and_save(self, limit_per_source: int = 20, filename: Optional[str] = None) -> str:
        """
        Thu thập tin tức từ tất cả các nguồn và lưu vào file.
        
        Args:
            limit_per_source: Số lượng bài viết tối đa từ mỗi nguồn
            filename: Tên file (tùy chọn)
            
        Returns:
            Đường dẫn đến file đã lưu
        """
        results = await self.collect_from_all_sources(limit_per_source=limit_per_source)
        
        # Ghép tất cả bài viết thành một danh sách
        all_articles = []
        for source_articles in results.values():
            all_articles.extend(source_articles)
        
        # Sắp xếp theo thời gian từ mới đến cũ
        sorted_articles = sorted(
            all_articles,
            key=lambda x: x.get('published_at', ''),
            reverse=True
        )
        
        return self.save_articles_to_json(sorted_articles, filename)
    
    def filter_articles_by_keywords(self, articles: List[Dict[str, Any]], 
                                   keywords: List[str],
                                   fields: List[str] = ['title', 'description', 'content']) -> List[Dict[str, Any]]:
        """
        Lọc bài viết theo từ khóa.
        
        Args:
            articles: Danh sách bài viết cần lọc
            keywords: Danh sách từ khóa tìm kiếm
            fields: Các trường cần tìm kiếm (mặc định: title, description, content)
            
        Returns:
            Danh sách bài viết đã lọc
        """
        filtered_articles = []
        
        # Chuyển từ khóa sang chữ thường
        keywords_lower = [keyword.lower() for keyword in keywords]
        
        for article in articles:
            matched = False
            
            for field in fields:
                if field in article and article[field]:
                    field_value = str(article[field]).lower()
                    
                    for keyword in keywords_lower:
                        if keyword in field_value:
                            matched = True
                            break
                    
                    if matched:
                        break
            
            if matched:
                filtered_articles.append(article)
        
        return filtered_articles


# Phương thức chạy chính
async def main():
    """Hàm test chạy trực tiếp module."""
    collector = CryptoNewsCollector()
    
    # Thu thập tin từ tất cả các nguồn
    print("Đang thu thập tin tức...")
    results = await collector.collect_from_all_sources(limit_per_source=5)
    
    # In số lượng bài viết từ mỗi nguồn
    for source_name, articles in results.items():
        print(f"{source_name}: {len(articles)} bài viết")
    
    # Lưu các bài viết vào file
    all_articles = []
    for articles in results.values():
        all_articles.extend(articles)
    
    if all_articles:
        file_path = collector.save_articles_to_json(all_articles)
        print(f"Đã lưu tất cả bài viết vào {file_path}")

if __name__ == "__main__":
    # Chạy hàm main bất đồng bộ
    asyncio.run(main())
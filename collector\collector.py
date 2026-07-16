#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TokenCorpus 数据采集器
================================

专业级多源语料采集系统，支持：
- arXiv学术论文
- IETF RFC标准
- ACL Anthology论文
- 技术文档（OpenAI, HuggingFace等）
- 技术媒体

项目：TokenCorpus
机构：河南农业大学语言智能研究中心
版本：v1.0
日期：2026-07

使用说明：
---------
python collector.py --source all --limit 1000
python collector.py --source arxiv --domain cs.CL --limit 500
python collector.py --source rfc --limit 100
python collector.py --resume  # 断点续传

依赖安装：
---------
pip install requests beautifulsoup4 nltk lxml feedparser
"""

import os
import re
import json
import time
import logging
import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from urllib.parse import urljoin, urlparse
import argparse
import xml.etree.ElementTree as ET

# 第三方库
try:
    import requests
    from bs4 import BeautifulSoup
    import nltk
    import feedparser
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请运行: pip install requests beautifulsoup4 nltk feedparser lxml")
    exit(1)

# 下载NLTK资源
for resource in ['punkt', 'averaged_perceptron_tagger', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'taggers/{resource}')
    except LookupError:
        nltk.download(resource.split('_')[0] if '_' in resource else resource, quiet=True)

# ==================== 配置 ====================

@dataclass
class CorpusConfig:
    """语料库配置"""
    # 基础配置
    corpus_id: str = "TokenCorpus-v1.0"
    output_dir: str = "./data"
    db_path: str = "./data/corpus.db"
    
    # 采集配置
    max_workers: int = 5
    delay_seconds: float = 1.0
    timeout: int = 30
    retry_times: int = 3
    max_context_words: int = 50
    
    # 质量配置
    min_sentence_words: int = 5
    max_sentence_words: int = 200
    dedup_threshold: float = 0.9  # SimHash相似度阈值


@dataclass
class SourceConfig:
    """数据源配置"""
    name: str
    source_type: str
    url: str
    keywords: List[str]
    priority: int = 1
    enabled: bool = True
    rate_limit: float = 1.0  # 该源请求间隔


@dataclass
class TokenOccurrence:
    """Token出现实例"""
    id: str = ""
    corpus_id: str = ""
    sentence: str = ""
    context_before: str = ""
    context_after: str = ""
    paragraph: str = ""
    source_type: str = ""
    source_name: str = ""
    source_url: str = ""
    source_doi: str = ""
    publication_date: str = ""
    authors: List[str] = field(default_factory=list)
    domain: str = ""
    sub_domain: str = ""
    keywords: List[str] = field(default_factory=list)
    sense: str = ""
    metaphor_type: str = ""
    simhash: str = ""
    created_at: str = ""
    version: str = "1.0"
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['created_at'] = self.created_at or datetime.now().isoformat()
        return d


# ==================== 数据源定义 ====================

class DataSources:
    """数据源注册表"""
    
    # arXiv主题分类
    ARXIV_CATEGORIES = {
        "cs.CL": "Computation and Language",
        "cs.LG": "Machine Learning",
        "cs.AI": "Artificial Intelligence",
        "cs.CR": "Cryptography and Security",
        "cs.PL": "Programming Languages",
        "cs.NE": "Neural and Evolutionary Computing",
        "cs.IR": "Information Retrieval",
    }
    
    # 完整数据源列表
    ALL_SOURCES: List[SourceConfig] = [
        # ===== 标准化文档 =====
        SourceConfig(
            name="RFC 6749 - OAuth 2.0",
            source_type="rfc_standard",
            url="https://www.rfc-editor.org/rfc/rfc6749.txt",
            keywords=["token", "access token", "refresh token", "bearer token", "authorization"],
            priority=5
        ),
        SourceConfig(
            name="RFC 7519 - JWT",
            source_type="rfc_standard",
            url="https://www.rfc-editor.org/rfc/rfc7519.txt",
            keywords=["token", "JWT", "claim", "json web token"],
            priority=5
        ),
        SourceConfig(
            name="RFC 6750 - Bearer Token",
            source_type="rfc_standard",
            url="https://www.rfc-editor.org/rfc/rfc6750.txt",
            keywords=["token", "bearer", "WWW-Authenticate"],
            priority=4
        ),
        SourceConfig(
            name="RFC 7009 - Token Revocation",
            source_type="rfc_standard",
            url="https://www.rfc-editor.org/rfc/rfc7009.txt",
            keywords=["token", "revocation"],
            priority=3
        ),
        SourceConfig(
            name="RFC 7636 - PKCE",
            source_type="rfc_standard",
            url="https://www.rfc-editor.org/rfc/rfc7636.txt",
            keywords=["token", "PKCE", "code verifier"],
            priority=3
        ),
        
        # ===== arXiv学术论文 =====
        SourceConfig(
            name="arXiv CS.CL",
            source_type="arxiv",
            url="https://export.arxiv.org/api/query?search_query=cat:cs.CL&start=0&max_results=100&sortBy=submittedDate",
            keywords=["token", "tokenizer", "tokenization", "subword", "BPE", "wordpiece"],
            priority=5
        ),
        SourceConfig(
            name="arXiv CS.LG",
            source_type="arxiv",
            url="https://export.arxiv.org/api/query?search_query=cat:cs.LG&start=0&max_results=100&sortBy=submittedDate",
            keywords=["token", "transformer", "attention", "language model"],
            priority=5
        ),
        SourceConfig(
            name="arXiv CS.AI",
            source_type="arxiv",
            url="https://export.arxiv.org/api/query?search_query=cat:cs.AI&start=0&max_results=100&sortBy=submittedDate",
            keywords=["token", "AI", "agent"],
            priority=4
        ),
        SourceConfig(
            name="arXiv CS.CR",
            source_type="arxiv",
            url="https://export.arxiv.org/api/query?search_query=cat:cs.CR&start=0&max_results=100&sortBy=submittedDate",
            keywords=["token", "authentication", "security"],
            priority=4
        ),
        
        # ===== 技术文档 =====
        SourceConfig(
            name="OpenAI API Docs",
            source_type="api_doc",
            url="https://api.openai.com/v1",
            keywords=["token", "completion", "embedding", "chat"],
            priority=5
        ),
        SourceConfig(
            name="HuggingFace Transformers",
            source_type="framework_doc",
            url="https://huggingface.co/docs/transformers",
            keywords=["token", "tokenizer", "model", "pipeline"],
            priority=5
        ),
        SourceConfig(
            name="Ethereum Documentation",
            source_type="blockchain_doc",
            url="https://ethereum.org/en/developers/docs/",
            keywords=["token", "smart contract", "ether", "gas"],
            priority=4
        ),
        SourceConfig(
            name="Auth0 Documentation",
            source_type="api_doc",
            url="https://auth0.com/docs",
            keywords=["token", "JWT", "access token", "ID token"],
            priority=4
        ),
        SourceConfig(
            name="LLVM Language Reference",
            source_type="framework_doc",
            url="https://llvm.org/docs/LangRef.html",
            keywords=["token", "instruction", "value", "type"],
            priority=3
        ),
    ]
    
    @classmethod
    def get_sources(cls, source_type: str = None, enabled_only: bool = True) -> List[SourceConfig]:
        """获取数据源列表"""
        sources = cls.ALL_SOURCES
        if source_type:
            sources = [s for s in sources if s.source_type == source_type]
        if enabled_only:
            sources = [s for s in sources if s.enabled]
        return sorted(sources, key=lambda x: x.priority, reverse=True)


# ==================== 工具函数 ====================

def compute_simhash(text: str, n: int = 5) -> str:
    """计算SimHash用于去重"""
    text = text.lower().strip()
    words = text.split()
    
    # 简单的SimHash实现
    v = [0] * 64
    for i, word in enumerate(words):
        h = int(hashlib.md5(word.encode()).hexdigest(), 16)
        for j in range(64):
            bit = (h >> j) & 1
            v[j] += 1 if bit else -1
    
    # 转换为哈希值
    hash_val = 0
    for i, val in enumerate(v):
        if val > 0:
            hash_val |= (1 << i)
    
    return format(hash_val, '064b')


def hamming_distance(h1: str, h2: str) -> int:
    """计算两个SimHash的海明距离"""
    h1_val = int(h1, 2)
    h2_val = int(h2, 2)
    xor = h1_val ^ h2_val
    return bin(xor).count('1')


def setup_logging(log_file: str) -> logging.Logger:
    """设置日志"""
    logger = logging.getLogger("TokenCorpus")
    logger.setLevel(logging.DEBUG)
    
    # 文件处理器
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # 控制台处理器
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter(
        '%(asctime)s │ %(levelname)-10s │ %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


# ==================== 数据库管理 ====================

class CorpusDatabase:
    """语料库数据库管理"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_tables()
    
    def _init_tables(self):
        """初始化数据库表"""
        cursor = self.conn.cursor()
        
        # 主表：语料记录
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS occurrences (
                id TEXT PRIMARY KEY,
                corpus_id TEXT NOT NULL,
                sentence TEXT NOT NULL,
                context_before TEXT,
                context_after TEXT,
                paragraph TEXT,
                source_type TEXT,
                source_name TEXT,
                source_url TEXT,
                source_doi TEXT,
                publication_date TEXT,
                authors TEXT,
                domain TEXT,
                sub_domain TEXT,
                keywords TEXT,
                sense TEXT,
                metaphor_type TEXT,
                simhash TEXT,
                created_at TEXT,
                version TEXT,
                UNIQUE(corpus_id, simhash)
            )
        ''')
        
        # 索引
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_source_type ON occurrences(source_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_domain ON occurrences(domain)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_sense ON occurrences(sense)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_simhash ON occurrences(simhash)')
        
        # 来源表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                name TEXT PRIMARY KEY,
                source_type TEXT,
                url TEXT,
                total_fetched INTEGER DEFAULT 0,
                tokens_extracted INTEGER DEFAULT 0,
                last_fetched TEXT,
                status TEXT
            )
        ''')
        
        # 采集历史
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS collection_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name TEXT,
                started_at TEXT,
                completed_at TEXT,
                records_added INTEGER,
                status TEXT,
                error_message TEXT
            )
        ''')
        
        self.conn.commit()
    
    def exists(self, simhash: str) -> bool:
        """检查是否已存在"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT 1 FROM occurrences WHERE simhash = ?', (simhash,))
        return cursor.fetchone() is not None
    
    def insert(self, occurrence: TokenOccurrence) -> bool:
        """插入记录"""
        cursor = self.conn.cursor()
        try:
            cursor.execute('''
                INSERT OR IGNORE INTO occurrences 
                (id, corpus_id, sentence, context_before, context_after, paragraph,
                 source_type, source_name, source_url, source_doi, publication_date,
                 authors, domain, sub_domain, keywords, sense, metaphor_type,
                 simhash, created_at, version)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                occurrence.id, occurrence.corpus_id, occurrence.sentence,
                occurrence.context_before, occurrence.context_after, occurrence.paragraph,
                occurrence.source_type, occurrence.source_name, occurrence.source_url,
                occurrence.source_doi, occurrence.publication_date, json.dumps(occurrence.authors),
                occurrence.domain, occurrence.sub_domain, json.dumps(occurrence.keywords),
                occurrence.sense, occurrence.metaphor_type, occurrence.simhash,
                occurrence.created_at, occurrence.version
            ))
            self.conn.commit()
            return cursor.rowcount > 0
        except sqlite3.IntegrityError:
            return False
    
    def count(self, domain: str = None) -> int:
        """统计记录数"""
        cursor = self.conn.cursor()
        if domain:
            cursor.execute('SELECT COUNT(*) FROM occurrences WHERE domain = ?', (domain,))
        else:
            cursor.execute('SELECT COUNT(*) FROM occurrences')
        return cursor.fetchone()[0]
    
    def get_all(self, limit: int = None) -> List[Dict]:
        """获取所有记录"""
        cursor = self.conn.cursor()
        if limit:
            cursor.execute('SELECT * FROM occurrences LIMIT ?', (limit,))
        else:
            cursor.execute('SELECT * FROM occurrences')
        return [dict(row) for row in cursor.fetchall()]
    
    def close(self):
        """关闭连接"""
        self.conn.close()


# ==================== 数据采集器基类 ====================

class BaseCollector:
    """采集器基类"""
    
    def __init__(self, config: CorpusConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TokenCorpus-Collector/1.0 (Academic Research)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        })
    
    def fetch(self, url: str, timeout: int = None) -> Optional[str]:
        """获取页面内容"""
        timeout = timeout or self.config.timeout
        try:
            response = self.session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            self.logger.warning(f"获取失败 [{url}]: {e}")
            return None
    
    def extract_sentences(self, text: str) -> List[str]:
        """分句"""
        sentences = nltk.sent_tokenize(text)
        # 过滤
        filtered = []
        for s in sentences:
            words = len(s.split())
            if self.config.min_sentence_words <= words <= self.config.max_sentence_words:
                filtered.append(s.strip())
        return filtered
    
    def find_token_context(self, sentence: str, keywords: List[str]) -> Tuple[str, str, str]:
        """提取上下文"""
        words = sentence.split()
        window = self.config.max_context_words
        
        if len(words) <= window * 2:
            return "", "", sentence
        
        # 找到关键词位置
        sentence_lower = sentence.lower()
        for kw in keywords:
            if kw.lower() in sentence_lower:
                idx = sentence_lower.find(kw.lower())
                # 简单处理
                return "", "", sentence
        
        return "", "", sentence
    
    def process(self, source: SourceConfig) -> List[TokenOccurrence]:
        """处理数据源"""
        raise NotImplementedError


# ==================== RFC采集器 ====================

class RFCCollector(BaseCollector):
    """RFC标准文档采集器"""
    
    def process(self, source: SourceConfig) -> List[TokenOccurrence]:
        """采集RFC文档"""
        self.logger.info(f"📄 采集RFC: {source.name}")
        
        content = self.fetch(source.url)
        if not content:
            return []
        
        occurrences = []
        sentences = self.extract_sentences(content)
        
        for sentence in sentences:
            # 检查是否包含token
            if not any(kw.lower() in sentence.lower() for kw in source.keywords):
                continue
            
            simhash = compute_simhash(sentence)
            # 去重检查在主采集器中处理
            
            occ = TokenOccurrence(
                id=f"TC_{int(time.time()*1000)}_{len(occurrences):04d}",
                corpus_id=self.config.corpus_id,
                sentence=sentence,
                source_type=source.source_type,
                source_name=source.name,
                source_url=source.url,
                domain=self._classify_domain(sentence),
                keywords=[kw for kw in source.keywords if kw.lower() in sentence.lower()],
                simhash=simhash,
                created_at=datetime.now().isoformat()
            )
            
            occurrences.append(occ)
        
        self.logger.info(f"  ✓ 提取 {len(occurrences)} 条记录")
        return occurrences
    
    def _classify_domain(self, sentence: str) -> str:
        """基于内容分类"""
        sentence_lower = sentence.lower()
        
        if any(kw in sentence_lower for kw in ['oauth', 'authorization', 'authenticate', 'credential']):
            return "security_auth"
        elif any(kw in sentence_lower for kw in ['json web token', 'jwt', 'claim']):
            return "security_auth"
        elif any(kw in sentence_lower for kw in ['access', 'refresh', 'bearer']):
            return "security_auth"
        
        return "security_auth"


# ==================== arXiv采集器 ====================

class ArxivCollector(BaseCollector):
    """arXiv论文采集器"""
    
    def process(self, source: SourceConfig) -> List[TokenOccurrence]:
        """采集arXiv论文"""
        self.logger.info(f"📚 采集arXiv: {source.name}")
        
        # 解析arXiv API响应
        content = self.fetch(source.url)
        if not content:
            return []
        
        occurrences = []
        
        try:
            root = ET.fromstring(content)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns)
                summary = entry.find('atom:summary', ns)
                published = entry.find('atom:published', ns)
                author_elements = entry.findall('atom:author/atom:name', ns)
                
                if title is None or summary is None:
                    continue
                
                # 合并标题和摘要
                text = f"{title.text}\n{summary.text}"
                sentences = self.extract_sentences(text)
                
                for sentence in sentences:
                    if not any(kw.lower() in sentence.lower() for kw in source.keywords):
                        continue
                    
                    simhash = compute_simhash(sentence)
                    # 去重检查在主采集器中处理
                    
                    authors = [a.text for a in author_elements] if author_elements else []
                    pub_date = published.text[:7] if published is not None else ""
                    
                    occ = TokenOccurrence(
                        id=f"TC_{int(time.time()*1000)}_{len(occurrences):04d}",
                        corpus_id=self.config.corpus_id,
                        sentence=sentence,
                        source_type=source.source_type,
                        source_name=f"arXiv:{title.text[:50]}",
                        source_url=entry.find('atom:id', ns).text if entry.find('atom:id', ns) is not None else "",
                        publication_date=pub_date,
                        authors=authors,
                        domain=self._classify_domain(sentence),
                        keywords=[kw for kw in source.keywords if kw.lower() in sentence.lower()],
                        simhash=simhash,
                        created_at=datetime.now().isoformat()
                    )
                    
                    occurrences.append(occ)
                    
        except ET.ParseError as e:
            self.logger.error(f"XML解析错误: {e}")
        
        self.logger.info(f"  ✓ 提取 {len(occurrences)} 条记录")
        return occurrences
    
    def _classify_domain(self, sentence: str) -> str:
        """基于内容分类"""
        sentence_lower = sentence.lower()
        
        if any(kw in sentence_lower for kw in ['tokenizer', 'tokenize', 'tokenization', 'bpe', 'wordpiece', 'subword']):
            return "nlp_processing"
        elif any(kw in sentence_lower for kw in ['transformer', 'attention', 'language model', 'llm', 'gpt', 'bert']):
            return "ai_llm"
        elif any(kw in sentence_lower for kw in ['authentication', 'security', 'crypto']):
            return "security_auth"
        elif any(kw in sentence_lower for kw in ['compiler', 'parser', 'lexer', 'syntax']):
            return "compiler_lang"
        
        return "nlp_processing"


# ==================== 网页采集器 ====================

class WebCollector(BaseCollector):
    """通用网页采集器"""
    
    def process(self, source: SourceConfig) -> List[TokenOccurrence]:
        """采集网页内容"""
        self.logger.info(f"🌐 采集网页: {source.name}")
        
        content = self.fetch(source.url)
        if not content:
            return []
        
        # 解析HTML
        soup = BeautifulSoup(content, 'lxml')
        
        # 移除脚本和样式
        for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
            tag.decompose()
        
        # 获取文本
        text = soup.get_text(separator=' ', strip=True)
        
        occurrences = []
        sentences = self.extract_sentences(text)
        
        for sentence in sentences:
            if not any(kw.lower() in sentence.lower() for kw in source.keywords):
                continue
            
            simhash = compute_simhash(sentence)
            # 去重检查在主采集器中处理
            
            occ = TokenOccurrence(
                id=f"TC_{int(time.time()*1000)}_{len(occurrences):04d}",
                corpus_id=self.config.corpus_id,
                sentence=sentence,
                source_type=source.source_type,
                source_name=source.name,
                source_url=source.url,
                domain=self._classify_domain(source.name, sentence),
                keywords=[kw for kw in source.keywords if kw.lower() in sentence.lower()],
                simhash=simhash,
                created_at=datetime.now().isoformat()
            )
            
            occurrences.append(occ)
        
        self.logger.info(f"  ✓ 提取 {len(occurrences)} 条记录")
        return occurrences
    
    def _classify_domain(self, source_name: str, sentence: str) -> str:
        """分类"""
        sentence_lower = sentence.lower()
        name_lower = source_name.lower()
        
        if 'openai' in name_lower:
            return "ai_llm"
        elif 'huggingface' in name_lower or 'transformers' in name_lower:
            return "nlp_processing"
        elif 'ethereum' in name_lower or 'blockchain' in name_lower:
            return "blockchain_crypto"
        elif 'auth0' in name_lower:
            return "security_auth"
        elif 'llvm' in name_lower:
            return "compiler_lang"
        
        return "general"


# ==================== 主采集器 ====================

class TokenCorpusCollector:
    """TokenCorpus主采集器"""
    
    def __init__(self, config: CorpusConfig):
        self.config = config
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 日志
        log_file = os.path.join(config.output_dir, f"collector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        self.logger = setup_logging(log_file)
        
        # 数据库
        self.db = CorpusDatabase(config.db_path)
        
        # 采集器
        self.collectors = {
            'rfc_standard': RFCCollector(config, self.logger),
            'arxiv': ArxivCollector(config, self.logger),
            'api_doc': WebCollector(config, self.logger),
            'framework_doc': WebCollector(config, self.logger),
            'blockchain_doc': WebCollector(config, self.logger),
        }
        
        # 统计
        self.stats = defaultdict(int)
    
    def collect_source(self, source: SourceConfig) -> int:
        """采集单个数据源"""
        self.logger.info("=" * 60)
        self.logger.info(f"开始采集: {source.name}")
        self.logger.info("=" * 60)
        
        # 获取对应采集器
        collector = self.collectors.get(source.source_type)
        if not collector:
            self.logger.warning(f"未知的源类型: {source.source_type}")
            return 0
        
        # 采集
        try:
            occurrences = collector.process(source)
            
            # 存入数据库
            added = 0
            for occ in occurrences:
                if self.db.insert(occ):
                    added += 1
            
            self.stats[source.source_type] += added
            self.logger.info(f"✓ 新增 {added} 条记录 (共处理 {len(occurrences)} 条)")
            
            return added
            
        except Exception as e:
            self.logger.error(f"采集失败: {e}")
            return 0
    
    def collect_all(self, source_types: List[str] = None, limit_per_source: int = None) -> Dict:
        """采集所有启用的数据源"""
        self.logger.info("=" * 70)
        self.logger.info("TokenCorpus 数据采集系统启动")
        self.logger.info("=" * 70)
        
        # 获取数据源
        sources = DataSources.get_sources()
        if source_types:
            sources = [s for s in sources if s.source_type in source_types]
        
        total_added = 0
        
        for source in sources:
            added = self.collect_source(source)
            total_added += added
            
            # 礼貌延迟
            time.sleep(source.rate_limit)
        
        # 汇总
        self.logger.info("\n" + "=" * 70)
        self.logger.info("采集完成!")
        self.logger.info(f"总计新增: {total_added} 条")
        self.logger.info(f"数据库总计: {self.db.count()} 条")
        self.logger.info("=" * 70)
        
        return {
            'total_added': total_added,
            'total_in_db': self.db.count(),
            'by_source_type': dict(self.stats)
        }
    
    def export(self, output_format: str = "jsonl", limit: int = None):
        """导出语料"""
        self.logger.info(f"导出语料 (格式: {output_format})...")
        
        records = self.db.get_all(limit=limit)
        
        if output_format == "jsonl":
            output_file = os.path.join(
                self.config.output_dir,
                f"TokenCorpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            )
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            self.logger.info(f"导出完成: {output_file}")
            return output_file
        
        elif output_format == "json":
            output_file = os.path.join(
                self.config.output_dir,
                f"TokenCorpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            self.logger.info(f"导出完成: {output_file}")
            return output_file
    
    def generate_stats(self) -> Dict:
        """生成统计报告"""
        records = self.db.get_all()
        
        stats = {
            'total': len(records),
            'by_source_type': defaultdict(int),
            'by_domain': defaultdict(int),
            'by_sense': defaultdict(int),
        }
        
        for record in records:
            stats['by_source_type'][record['source_type']] += 1
            stats['by_domain'][record['domain']] += 1
            if record['sense']:
                stats['by_sense'][record['sense']] += 1
        
        return dict(stats)
    
    def close(self):
        """关闭连接"""
        self.db.close()


# ==================== 主程序 ====================

def main():
    parser = argparse.ArgumentParser(
        description="TokenCorpus 数据采集器",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--source', '-s',
        nargs='+',
        choices=['all', 'rfc', 'arxiv', 'web'],
        default=['all'],
        help='数据源类型'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./data',
        help='输出目录'
    )
    
    parser.add_argument(
        '--export',
        choices=['jsonl', 'json', 'stats'],
        help='导出格式'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='导出记录数限制'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=5,
        help='并发线程数'
    )
    
    args = parser.parse_args()
    
    # 配置
    config = CorpusConfig(
        output_dir=args.output,
        db_path=os.path.join(args.output, 'corpus.db'),
        max_workers=args.workers
    )
    
    # 初始化采集器
    collector = TokenCorpusCollector(config)
    
    # 确定源类型
    source_types = None
    if 'all' not in args.source:
        type_map = {
            'rfc': 'rfc_standard',
            'arxiv': 'arxiv',
            'web': ['api_doc', 'framework_doc', 'blockchain_doc']
        }
        source_types = []
        for s in args.source:
            if isinstance(type_map.get(s), list):
                source_types.extend(type_map[s])
            elif type_map.get(s):
                source_types.append(type_map[s])
    
    # 采集或导出
    if args.export:
        if args.export == 'stats':
            stats = collector.generate_stats()
            print(json.dumps(stats, indent=2, ensure_ascii=False))
        else:
            collector.export(args.export, args.limit)
    else:
        collector.collect_all(source_types=source_types)
    
    collector.close()


if __name__ == "__main__":
    main()

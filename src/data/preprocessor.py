"""
Text Preprocessing Module
Cleans and normalizes text content while preserving important metadata.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set
import unicodedata
import html
from datetime import datetime

logger = logging.getLogger(__name__)


class PreprocessingError(Exception):
    """Custom exception for preprocessing operations"""
    pass


class ProcessedText:
    """Container for processed text with metadata"""
    
    def __init__(
        self,
        content: str,
        original_length: int,
        processed_length: int,
        transformations: List[str],
        metadata: Dict[str, Any]
    ):
        self.content = content
        self.original_length = original_length
        self.processed_length = processed_length
        self.transformations = transformations
        self.metadata = metadata
        self.processed_at = datetime.now()
    
    def __len__(self) -> int:
        return len(self.content)
    
    def compression_ratio(self) -> float:
        """Calculate compression ratio from preprocessing"""
        if self.original_length == 0:
            return 0.0
        return self.processed_length / self.original_length


class TextPreprocessor:
    """
    Advanced text preprocessing with configurable cleaning strategies
    """
    
    def __init__(
        self,
        normalize_unicode: bool = True,
        remove_html_entities: bool = True,
        normalize_whitespace: bool = True,
        remove_extra_newlines: bool = True,
        normalize_quotes: bool = True,
        remove_urls: bool = False,
        remove_emails: bool = False,
        min_line_length: int = 3,
        preserve_structure: bool = True
    ):
        """
        Initialize text preprocessor
        
        Args:
            normalize_unicode: Normalize Unicode characters
            remove_html_entities: Convert HTML entities to text
            normalize_whitespace: Normalize whitespace characters
            remove_extra_newlines: Remove excessive newlines
            normalize_quotes: Normalize quotation marks
            remove_urls: Remove URLs from text
            remove_emails: Remove email addresses
            min_line_length: Minimum length for keeping lines
            preserve_structure: Preserve document structure markers
        """
        self.normalize_unicode = normalize_unicode
        self.remove_html_entities = remove_html_entities
        self.normalize_whitespace = normalize_whitespace
        self.remove_extra_newlines = remove_extra_newlines
        self.normalize_quotes = normalize_quotes
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.min_line_length = min_line_length
        self.preserve_structure = preserve_structure
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
        
        # Initialize statistics
        self.stats = {
            "documents_processed": 0,
            "total_original_length": 0,
            "total_processed_length": 0,
            "average_compression_ratio": 0.0
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for text cleaning"""
        # URL pattern
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Email pattern
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Multiple whitespace pattern
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Multiple newlines pattern
        self.newlines_pattern = re.compile(r'\n\s*\n\s*\n+')
        
        # Quote normalization patterns
        self.quote_patterns = [
            (re.compile(r'[""]'), '"'),  # Smart quotes to straight quotes
            (re.compile(r'['']'), "'"),  # Smart apostrophes to straight apostrophes
        ]
        
        # Structure preservation patterns
        self.structure_patterns = {
            'header': re.compile(r'^#{1,6}\s+.*$', re.MULTILINE),
            'list_item': re.compile(r'^\s*[-*+]\s+.*$', re.MULTILINE),
            'numbered_list': re.compile(r'^\s*\d+\.\s+.*$', re.MULTILINE),
            'code_block': re.compile(r'```.*?```', re.DOTALL),
            'inline_code': re.compile(r'`[^`]+`')
        }
    
    def _normalize_unicode_text(self, text: str) -> str:
        """Normalize Unicode characters"""
        # Normalize to NFC form (canonical decomposition, followed by canonical composition)
        return unicodedata.normalize('NFC', text)
    
    def _remove_html_entities(self, text: str) -> str:
        """Convert HTML entities to their text equivalents"""
        return html.unescape(text)
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters"""
        # Replace various whitespace characters with standard space
        text = text.replace('\t', ' ')  # Tabs to spaces
        text = text.replace('\xa0', ' ')  # Non-breaking space to space
        text = text.replace('\u200b', '')  # Zero-width space
        text = text.replace('\ufeff', '')  # Byte order mark
        
        # Normalize multiple spaces to single space (but preserve newlines)
        lines = text.split('\n')
        normalized_lines = []
        
        for line in lines:
            # Replace multiple spaces with single space within each line
            normalized_line = self.whitespace_pattern.sub(' ', line.strip())
            normalized_lines.append(normalized_line)
        
        return '\n'.join(normalized_lines)
    
    def _remove_extra_newlines(self, text: str) -> str:
        """Remove excessive newlines while preserving paragraph structure"""
        # Replace multiple consecutive newlines with double newline (paragraph break)
        return self.newlines_pattern.sub('\n\n', text)
    
    def _normalize_quotes(self, text: str) -> str:
        """Normalize quotation marks and apostrophes"""
        for pattern, replacement in self.quote_patterns:
            text = pattern.sub(replacement, text)
        return text
    
    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        return self.url_pattern.sub('', text)
    
    def _remove_emails(self, text: str) -> str:
        """Remove email addresses from text"""
        return self.email_pattern.sub('', text)
    
    def _filter_short_lines(self, text: str) -> str:
        """Remove lines that are too short to be meaningful"""
        lines = text.split('\n')
        filtered_lines = []
        
        for line in lines:
            line = line.strip()
            if len(line) >= self.min_line_length or not line:
                # Keep lines that are long enough or empty (for structure)
                filtered_lines.append(line)
        
        return '\n'.join(filtered_lines)
    
    def _preserve_structure_markers(self, text: str) -> Dict[str, List[str]]:
        """Extract and preserve document structure markers"""
        structures = {}
        
        if self.preserve_structure:
            for structure_type, pattern in self.structure_patterns.items():
                matches = pattern.findall(text)
                if matches:
                    structures[structure_type] = matches
        
        return structures
    
    def _detect_content_language(self, text: str) -> Optional[str]:
        """Basic language detection based on character patterns"""
        # Simple heuristic based on character ranges
        if re.search(r'[\u4e00-\u9fff]', text):  # Chinese
            return 'zh'
        elif re.search(r'[\u0400-\u04ff]', text):  # Cyrillic
            return 'ru'
        elif re.search(r'[\u0590-\u05ff]', text):  # Hebrew
            return 'he'
        elif re.search(r'[\u0600-\u06ff]', text):  # Arabic
            return 'ar'
        else:
            return 'en'  # Default to English
    
    def _calculate_text_statistics(self, text: str) -> Dict[str, Any]:
        """Calculate basic text statistics"""
        lines = text.split('\n')
        words = text.split()
        
        return {
            'character_count': len(text),
            'word_count': len(words),
            'line_count': len(lines),
            'non_empty_lines': len([line for line in lines if line.strip()]),
            'average_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
            'average_word_length': sum(len(word) for word in words) / len(words) if words else 0
        }
    
    def process_text(self, text: str, preserve_metadata: Optional[Dict[str, Any]] = None) -> ProcessedText:
        """
        Process text with all configured transformations
        
        Args:
            text: Input text to process
            preserve_metadata: Optional metadata to preserve
            
        Returns:
            ProcessedText: Processed text with metadata
            
        Raises:
            PreprocessingError: If preprocessing fails
        """
        if not isinstance(text, str):
            raise PreprocessingError("Input must be a string")
        
        if not text.strip():
            raise PreprocessingError("Input text is empty")
        
        original_length = len(text)
        transformations = []
        
        try:
            # Store original structure if needed
            structure_markers = self._preserve_structure_markers(text)
            
            processed_text = text
            
            # Apply transformations in order
            if self.remove_html_entities:
                processed_text = self._remove_html_entities(processed_text)
                transformations.append("html_entities_removed")
            
            if self.normalize_unicode:
                processed_text = self._normalize_unicode_text(processed_text)
                transformations.append("unicode_normalized")
            
            if self.remove_urls:
                processed_text = self._remove_urls(processed_text)
                transformations.append("urls_removed")
            
            if self.remove_emails:
                processed_text = self._remove_emails(processed_text)
                transformations.append("emails_removed")
            
            if self.normalize_quotes:
                processed_text = self._normalize_quotes(processed_text)
                transformations.append("quotes_normalized")
            
            if self.normalize_whitespace:
                processed_text = self._normalize_whitespace(processed_text)
                transformations.append("whitespace_normalized")
            
            if self.remove_extra_newlines:
                processed_text = self._remove_extra_newlines(processed_text)
                transformations.append("extra_newlines_removed")
            
            # Filter short lines (do this last to preserve structure)
            processed_text = self._filter_short_lines(processed_text)
            transformations.append("short_lines_filtered")
            
            # Final cleanup
            processed_text = processed_text.strip()
            
            if not processed_text:
                raise PreprocessingError("Text became empty after preprocessing")
            
            processed_length = len(processed_text)
            
            # Build metadata
            metadata = {
                'language': self._detect_content_language(processed_text),
                'structure_markers': structure_markers,
                'text_statistics': self._calculate_text_statistics(processed_text),
                'preprocessing_config': {
                    'normalize_unicode': self.normalize_unicode,
                    'remove_html_entities': self.remove_html_entities,
                    'normalize_whitespace': self.normalize_whitespace,
                    'remove_extra_newlines': self.remove_extra_newlines,
                    'normalize_quotes': self.normalize_quotes,
                    'remove_urls': self.remove_urls,
                    'remove_emails': self.remove_emails,
                    'min_line_length': self.min_line_length
                }
            }
            
            # Merge with preserved metadata
            if preserve_metadata:
                metadata.update(preserve_metadata)
            
            # Update statistics
            self.stats["documents_processed"] += 1
            self.stats["total_original_length"] += original_length
            self.stats["total_processed_length"] += processed_length
            
            # Calculate rolling average
            if self.stats["documents_processed"] > 0:
                self.stats["average_compression_ratio"] = (
                    self.stats["total_processed_length"] / self.stats["total_original_length"]
                )
            
            result = ProcessedText(
                content=processed_text,
                original_length=original_length,
                processed_length=processed_length,
                transformations=transformations,
                metadata=metadata
            )
            
            logger.debug(f"Processed text: {original_length} -> {processed_length} chars")
            return result
            
        except Exception as e:
            logger.error(f"Text preprocessing failed: {e}")
            raise PreprocessingError(f"Preprocessing failed: {e}")
    
    def process_batch(self, texts: List[str]) -> List[ProcessedText]:
        """
        Process multiple texts in batch
        
        Args:
            texts: List of input texts
            
        Returns:
            List[ProcessedText]: List of processed texts
        """
        results = []
        
        for i, text in enumerate(texts):
            try:
                processed = self.process_text(text)
                results.append(processed)
            except PreprocessingError as e:
                logger.warning(f"Failed to process text {i}: {e}")
                continue
        
        logger.info(f"Processed {len(results)} out of {len(texts)} texts")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        return self.stats.copy()


def create_text_preprocessor(
    aggressive_cleaning: bool = False,
    preserve_structure: bool = True
) -> TextPreprocessor:
    """
    Factory function to create text preprocessor with preset configurations
    
    Args:
        aggressive_cleaning: Enable aggressive cleaning (removes URLs, emails)
        preserve_structure: Preserve document structure markers
        
    Returns:
        TextPreprocessor: Configured text preprocessor
    """
    return TextPreprocessor(
        normalize_unicode=True,
        remove_html_entities=True,
        normalize_whitespace=True,
        remove_extra_newlines=True,
        normalize_quotes=True,
        remove_urls=aggressive_cleaning,
        remove_emails=aggressive_cleaning,
        min_line_length=3 if not aggressive_cleaning else 5,
        preserve_structure=preserve_structure
    ) 
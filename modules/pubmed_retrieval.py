import pandas as pd
import numpy as np
from Bio import Entrez
import time
import xml.etree.ElementTree as ET
from datetime import datetime
import re
from typing import List, Dict, Optional, Callable
import streamlit as st

class PubMedRetriever:
    """
    Handles retrieval of hand surgery literature from PubMed using Biopython's Entrez API.
    Implements rate limiting and batch processing for large-scale data collection.
    """
    
    def __init__(self, email: str, api_key: Optional[str] = None):
        """
        Initialize PubMed retriever with email and optional API key.
        
        Args:
            email: Email address required by NCBI
            api_key: Optional API key for higher rate limits (10 RPS vs 3 RPS)
        """
        self.email = email
        self.api_key = api_key
        
        # Configure Entrez
        Entrez.email = email
        if api_key:
            Entrez.api_key = api_key
            self.requests_per_second = 10
        else:
            self.requests_per_second = 3
        
        # Rate limiting
        self.last_request_time = 0
        self.min_interval = 1.0 / self.requests_per_second
        
        # Search statistics
        self.total_searches = 0
        self.total_abstracts = 0
        self.failed_searches = 0
    
    def _rate_limit(self):
        """Implement rate limiting to comply with NCBI guidelines."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_interval:
            sleep_time = self.min_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _search_pubmed(self, term: str, start_year: int, end_year: int, 
                      retmax: int = 5000) -> List[str]:
        """
        Search PubMed for a specific term and date range.
        
        Args:
            term: Search term
            start_year: Start year for publication date
            end_year: End year for publication date
            retmax: Maximum records to retrieve (limited to 10,000 by PubMed)
            
        Returns:
            List of PubMed IDs
        """
        try:
            self._rate_limit()
            
            # Construct search query with date limits
            date_filter = f"{start_year}[PDAT]:{end_year}[PDAT]"
            search_query = f"({term}) AND ({date_filter})"
            
            # Search PubMed
            handle = Entrez.esearch(
                db="pubmed",
                term=search_query,
                retmax=min(retmax, 10000),  # PubMed limit
                sort="pub_date",
                retmode="xml"
            )
            
            search_results = Entrez.read(handle)
            handle.close()
            
            pmids = search_results.get("IdList", [])
            self.total_searches += 1
            
            return pmids
            
        except Exception as e:
            self.failed_searches += 1
            st.error(f"Search failed for term '{term}': {str(e)}")
            return []
    
    def _fetch_abstracts_batch(self, pmids: List[str], batch_size: int = 200) -> List[Dict]:
        """
        Fetch abstracts for a batch of PubMed IDs.
        
        Args:
            pmids: List of PubMed IDs
            batch_size: Number of abstracts to fetch in each request
            
        Returns:
            List of abstract data dictionaries
        """
        abstracts = []
        
        # Process in batches to avoid API limits
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            
            try:
                self._rate_limit()
                
                # Fetch abstracts
                handle = Entrez.efetch(
                    db="pubmed",
                    id=",".join(batch_pmids),
                    rettype="abstract",
                    retmode="xml"
                )
                
                # Parse XML response
                xml_data = handle.read()
                handle.close()
                
                # Parse XML to extract abstract information
                batch_abstracts = self._parse_pubmed_xml(xml_data)
                abstracts.extend(batch_abstracts)
                
                self.total_abstracts += len(batch_abstracts)
                
            except Exception as e:
                st.warning(f"Failed to fetch batch {i//batch_size + 1}: {str(e)}")
                continue
        
        return abstracts
    
    def _parse_pubmed_xml(self, xml_data: str) -> List[Dict]:
        """
        Parse PubMed XML response to extract abstract information.
        
        Args:
            xml_data: Raw XML response from PubMed
            
        Returns:
            List of parsed abstract dictionaries
        """
        abstracts = []
        
        try:
            root = ET.fromstring(xml_data)
            
            for article in root.findall(".//PubmedArticle"):
                abstract_data = self._extract_article_data(article)
                if abstract_data:
                    abstracts.append(abstract_data)
                    
        except ET.ParseError as e:
            st.error(f"XML parsing error: {str(e)}")
        
        return abstracts
    
    def _extract_article_data(self, article) -> Optional[Dict]:
        """
        Extract relevant data from a single PubMed article XML element.
        
        Args:
            article: XML element representing a PubMed article
            
        Returns:
            Dictionary with article data or None if extraction fails
        """
        try:
            # Extract PMID
            pmid_elem = article.find(".//PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None
            
            if not pmid:
                return None
            
            # Extract title
            title_elem = article.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            
            # Extract abstract
            abstract_text = ""
            abstract_elem = article.find(".//Abstract")
            if abstract_elem is not None:
                # Handle structured abstracts
                abstract_parts = []
                for text_elem in abstract_elem.findall(".//AbstractText"):
                    label = text_elem.get("Label", "")
                    text = text_elem.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract_text = " ".join(abstract_parts)
            
            # Extract authors
            authors = []
            author_list = article.find(".//AuthorList")
            if author_list is not None:
                for author in author_list.findall(".//Author"):
                    last_name = author.find("LastName")
                    first_name = author.find("ForeName")
                    
                    if last_name is not None:
                        author_name = last_name.text
                        if first_name is not None:
                            author_name += f", {first_name.text}"
                        authors.append(author_name)
            
            # Extract journal
            journal_elem = article.find(".//Journal/Title")
            if journal_elem is None:
                journal_elem = article.find(".//Journal/ISOAbbreviation")
            journal = journal_elem.text if journal_elem is not None else ""
            
            # Extract publication year
            year = None
            year_elem = article.find(".//PubDate/Year")
            if year_elem is not None:
                try:
                    year = int(year_elem.text)
                except ValueError:
                    pass
            
            # If year not found, try MedlineDate
            if year is None:
                medline_date = article.find(".//PubDate/MedlineDate")
                if medline_date is not None:
                    # Extract year from MedlineDate (e.g., "2020 Jan-Feb")
                    year_match = re.search(r'\b(19|20)\d{2}\b', medline_date.text)
                    if year_match:
                        year = int(year_match.group())
            
            # Extract DOI
            doi = None
            for article_id in article.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    doi = article_id.text
                    break
            
            # Extract keywords
            keywords = []
            keyword_list = article.find(".//KeywordList")
            if keyword_list is not None:
                for keyword in keyword_list.findall(".//Keyword"):
                    if keyword.text:
                        keywords.append(keyword.text)
            
            # Extract MeSH terms
            mesh_terms = []
            mesh_heading_list = article.find(".//MeshHeadingList")
            if mesh_heading_list is not None:
                for mesh_heading in mesh_heading_list.findall(".//MeshHeading"):
                    descriptor = mesh_heading.find("DescriptorName")
                    if descriptor is not None:
                        mesh_terms.append(descriptor.text)
            
            return {
                'pmid': pmid,
                'title': title,
                'abstract': abstract_text,
                'authors': "; ".join(authors),
                'journal': journal,
                'year': year,
                'doi': doi,
                'keywords': "; ".join(keywords),
                'mesh_terms': "; ".join(mesh_terms),
                'retrieval_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            st.warning(f"Error extracting article data: {str(e)}")
            return None
    
    def retrieve_abstracts(self, search_terms: List[str], start_year: int, 
                          end_year: int, max_records: int = 5000,
                          progress_callback: Optional[Callable] = None) -> Optional[pd.DataFrame]:
        """
        Retrieve abstracts for multiple search terms and combine results.
        
        Args:
            search_terms: List of search terms
            start_year: Start year for publication date filter
            end_year: End year for publication date filter
            max_records: Maximum records per search term
            progress_callback: Optional callback for progress updates
            
        Returns:
            DataFrame with retrieved abstracts or None if failed
        """
        all_abstracts = []
        all_pmids = set()
        
        total_terms = len(search_terms)
        
        for i, term in enumerate(search_terms):
            if progress_callback:
                progress_callback(f"Searching for: {term} ({i+1}/{total_terms})")
            
            # Search for PMIDs
            pmids = self._search_pubmed(term, start_year, end_year, max_records)
            
            if not pmids:
                continue
            
            # Remove duplicates
            new_pmids = [pmid for pmid in pmids if pmid not in all_pmids]
            all_pmids.update(new_pmids)
            
            if progress_callback:
                progress_callback(f"Found {len(new_pmids)} new abstracts for: {term}")
            
            # Fetch abstracts in batches
            if new_pmids:
                abstracts = self._fetch_abstracts_batch(new_pmids)
                all_abstracts.extend(abstracts)
                
                if progress_callback:
                    progress_callback(f"Retrieved {len(abstracts)} abstracts for: {term}")
        
        if not all_abstracts:
            st.error("No abstracts retrieved. Please check your search terms and date range.")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_abstracts)
        
        # Data cleaning and validation
        df = self._clean_dataframe(df)
        
        # Filter by date range to ensure accuracy
        if 'year' in df.columns:
            df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]
        
        # Remove duplicates by PMID
        df = df.drop_duplicates(subset=['pmid'])
        
        # Sort by publication year (newest first)
        if 'year' in df.columns:
            df = df.sort_values('year', ascending=False)
        
        if progress_callback:
            progress_callback(f"Final dataset: {len(df)} unique abstracts")
        
        return df
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the abstracts DataFrame.
        
        Args:
            df: Raw DataFrame with abstract data
            
        Returns:
            Cleaned DataFrame
        """
        # Remove rows with missing critical data
        df = df.dropna(subset=['pmid', 'title'])
        
        # Remove abstracts that are too short (likely incomplete)
        min_abstract_length = 50
        df = df[df['abstract'].str.len() >= min_abstract_length]
        
        # Clean text fields
        text_columns = ['title', 'abstract', 'authors', 'journal', 'keywords', 'mesh_terms']
        for col in text_columns:
            if col in df.columns:
                # Remove extra whitespace and normalize
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace(['nan', 'None', ''], pd.NA)
        
        # Validate year values
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            # Remove invalid years
            df = df[(df['year'] >= 1990) & (df['year'] <= 2024)]
        
        # Add quality scores
        df['abstract_length'] = df['abstract'].str.len()
        df['has_mesh_terms'] = df['mesh_terms'].notna()
        df['has_keywords'] = df['keywords'].notna()
        
        return df
    
    def get_search_statistics(self) -> Dict:
        """
        Get statistics about the search process.
        
        Returns:
            Dictionary with search statistics
        """
        return {
            'total_searches': self.total_searches,
            'total_abstracts': self.total_abstracts,
            'failed_searches': self.failed_searches,
            'success_rate': (self.total_searches - self.failed_searches) / max(self.total_searches, 1),
            'requests_per_second': self.requests_per_second,
            'api_key_used': self.api_key is not None
        }

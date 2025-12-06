"""
Pagination Handler for AIRR API

Handles pagination for large datasets with streaming support.
"""

import logging
from typing import Dict, Any, List, Callable, Optional, Generator
from pathlib import Path

logger = logging.getLogger(__name__)


class PaginationHandler:
    """Handles pagination for large AIRR datasets"""

    def __init__(self, page_size: int = 1000, max_pages: Optional[int] = None):
        """
        Initialize pagination handler

        Args:
            page_size: Number of records per page
            max_pages: Optional maximum number of pages to fetch
        """
        self.page_size = page_size
        self.max_pages = max_pages

    def paginate_results(
        self,
        fetch_func: Callable[[int, int], Dict[str, Any]],
        max_records: Optional[int] = None
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Paginate through results using a fetch function

        Args:
            fetch_func: Function that takes (from_index, size) and returns results
            max_records: Optional maximum number of records to fetch

        Yields:
            Pages of results
        """
        from_index = 0
        page_count = 0
        total_fetched = 0

        while True:
            # Check if we've reached max pages
            if self.max_pages and page_count >= self.max_pages:
                logger.info(f"Reached maximum page limit: {self.max_pages}")
                break

            # Calculate how many records to fetch this page
            if max_records:
                remaining = max_records - total_fetched
                if remaining <= 0:
                    break
                page_size = min(self.page_size, remaining)
            else:
                page_size = self.page_size

            # Fetch page
            try:
                logger.debug(f"Fetching page {page_count + 1}, from={from_index}, size={page_size}")
                response = fetch_func(from_index, page_size)

                # Handle error responses
                if isinstance(response, dict) and "error" in response:
                    logger.error(f"Error fetching page: {response.get('message')}")
                    break

                # Extract results from response
                results = self._extract_results(response)

                if not results:
                    logger.info("No more results to fetch")
                    break

                yield results

                # Update counters
                total_fetched += len(results)
                from_index += len(results)
                page_count += 1

                # If we got fewer results than requested, we've reached the end
                if len(results) < page_size:
                    logger.info(f"Reached end of results (got {len(results)} < {page_size})")
                    break

            except Exception as e:
                logger.error(f"Exception during pagination: {e}")
                break

        logger.info(f"Pagination complete: fetched {total_fetched} records in {page_count} pages")

    def _extract_results(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract results list from API response

        Args:
            response: API response

        Returns:
            List of results
        """
        # AIRR API responses typically have results in "Rearrangement" or "Repertoire" key
        if "Rearrangement" in response:
            return response["Rearrangement"]
        elif "Repertoire" in response:
            return response["Repertoire"]
        elif isinstance(response, list):
            return response
        else:
            logger.warning(f"Unexpected response format: {list(response.keys())}")
            return []

    def collect_all_results(
        self,
        fetch_func: Callable[[int, int], Dict[str, Any]],
        max_records: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Collect all results into a single list (memory-intensive!)

        Args:
            fetch_func: Function that takes (from_index, size) and returns results
            max_records: Optional maximum number of records

        Returns:
            Complete list of results
        """
        all_results = []

        for page in self.paginate_results(fetch_func, max_records):
            all_results.extend(page)

        return all_results

    def stream_to_file(
        self,
        fetch_func: Callable[[int, int], Dict[str, Any]],
        output_file: Path,
        format_func: Callable[[Dict[str, Any]], str],
        header: Optional[str] = None,
        max_records: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Stream paginated results directly to file

        Args:
            fetch_func: Function that fetches data
            output_file: Output file path
            format_func: Function to format each record as a string
            header: Optional header line to write first
            max_records: Optional maximum records to fetch

        Returns:
            Summary statistics
        """
        total_written = 0
        page_count = 0

        try:
            with open(output_file, 'w') as f:
                # Write header if provided
                if header:
                    f.write(header + '\n')

                # Stream pages to file
                for page in self.paginate_results(fetch_func, max_records):
                    for record in page:
                        formatted = format_func(record)
                        f.write(formatted + '\n')
                        total_written += 1

                    page_count += 1

                    # Log progress every 10 pages
                    if page_count % 10 == 0:
                        logger.info(f"Progress: {total_written} records written ({page_count} pages)")

            return {
                "status": "success",
                "records_written": total_written,
                "pages_processed": page_count,
                "output_file": str(output_file)
            }

        except Exception as e:
            logger.error(f"Error streaming to file: {e}")
            return {
                "status": "error",
                "error": str(e),
                "records_written": total_written,
                "pages_processed": page_count
            }

    def estimate_total_count(
        self,
        fetch_func: Callable[[int, int], Dict[str, Any]]
    ) -> Optional[int]:
        """
        Estimate total count by fetching first page and checking response

        Args:
            fetch_func: Function that fetches data

        Returns:
            Estimated total count or None if unavailable
        """
        try:
            response = fetch_func(0, 1)

            # Some AIRR repositories return total count in response
            if isinstance(response, dict):
                if "Info" in response and "total_count" in response["Info"]:
                    return response["Info"]["total_count"]

            return None

        except Exception as e:
            logger.error(f"Error estimating total count: {e}")
            return None


class ChunkedDownloader:
    """Downloads large datasets in chunks with progress tracking"""

    def __init__(self, chunk_size: int = 10000):
        """
        Initialize chunked downloader

        Args:
            chunk_size: Records per chunk
        """
        self.chunk_size = chunk_size
        self.pagination = PaginationHandler(page_size=chunk_size)

    def download_with_progress(
        self,
        fetch_func: Callable[[int, int], Dict[str, Any]],
        output_file: Path,
        format_func: Callable[[Dict[str, Any]], str],
        header: Optional[str] = None,
        max_records: Optional[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Download with progress callbacks

        Args:
            fetch_func: Function that fetches data
            output_file: Output file path
            format_func: Function to format records
            header: Optional header
            max_records: Optional max records
            progress_callback: Optional callback(records_downloaded, pages_processed)

        Returns:
            Download statistics
        """
        total_written = 0
        page_count = 0

        try:
            # Estimate total if possible
            estimated_total = self.pagination.estimate_total_count(fetch_func)
            if estimated_total:
                logger.info(f"Estimated total records: {estimated_total}")

            with open(output_file, 'w') as f:
                if header:
                    f.write(header + '\n')

                for page in self.pagination.paginate_results(fetch_func, max_records):
                    for record in page:
                        formatted = format_func(record)
                        f.write(formatted + '\n')
                        total_written += 1

                    page_count += 1

                    # Call progress callback
                    if progress_callback:
                        progress_callback(total_written, page_count)

            return {
                "status": "success",
                "records_downloaded": total_written,
                "chunks_processed": page_count,
                "output_file": str(output_file),
                "estimated_total": estimated_total
            }

        except Exception as e:
            logger.error(f"Error in chunked download: {e}")
            return {
                "status": "error",
                "error": str(e),
                "records_downloaded": total_written,
                "chunks_processed": page_count
            }

    def download_parallel_chunks(
        self,
        fetch_func: Callable[[int, int], Dict[str, Any]],
        output_file: Path,
        format_func: Callable[[Dict[str, Any]], str],
        header: Optional[str] = None,
        total_records: Optional[int] = None,
        num_workers: int = 3
    ) -> Dict[str, Any]:
        """
        Download chunks in parallel (advanced feature)

        Args:
            fetch_func: Function that fetches data
            output_file: Output file path
            format_func: Function to format records
            header: Optional header
            total_records: Total records (required for parallel)
            num_workers: Number of parallel workers

        Returns:
            Download statistics
        """
        # This is a more advanced feature for very large datasets
        # For now, delegate to sequential download
        logger.info("Parallel download not yet implemented, using sequential")
        return self.download_with_progress(
            fetch_func,
            output_file,
            format_func,
            header,
            total_records,
            None
        )

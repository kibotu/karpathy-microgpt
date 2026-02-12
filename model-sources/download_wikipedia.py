#!/usr/bin/env -S uv run
"""
Download English Wikipedia database dump for training purposes.

This script downloads the English Wikipedia dump from dumps.wikimedia.org.
The recommended dump is pages-articles-multistream.xml.bz2 which contains:
- Current revisions only (no history)
- No talk or user pages
- Compressed size: ~25 GB
- Uncompressed size: ~105 GB

Reference: https://en.wikipedia.org/wiki/Wikipedia:Database_download
"""

import os
import sys
import hashlib
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime


# Wikipedia dump base URL
DUMP_BASE_URL = "https://dumps.wikimedia.org/enwiki"

# Available dump types
DUMP_TYPES = {
    "articles": {
        "filename": "enwiki-{date}-pages-articles-multistream.xml.bz2",
        "index": "enwiki-{date}-pages-articles-multistream-index.txt.bz2",
        "description": "Current article revisions only (recommended, ~25GB compressed)",
    },
    "abstracts": {
        "filename": "enwiki-{date}-abstract.xml.gz",
        "index": None,
        "description": "Page abstracts only (~2GB compressed)",
    },
    "titles": {
        "filename": "enwiki-{date}-all-titles-in-ns0.gz",
        "index": None,
        "description": "Article titles only (~100MB compressed)",
    },
    "meta-current": {
        "filename": "enwiki-{date}-pages-meta-current.xml.bz2",
        "index": None,
        "description": "Current revisions, all pages including talk (~35GB compressed)",
    },
}


def get_latest_dump_date() -> str:
    """Fetch the latest available dump date from Wikimedia."""
    url = f"{DUMP_BASE_URL}/"
    print(f"Fetching available dumps from {url}...")
    
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            html = response.read().decode('utf-8')
    except urllib.error.URLError as e:
        print(f"Error fetching dump list: {e}")
        sys.exit(1)
    
    # Parse dates from directory listing (format: YYYYMMDD)
    import re
    dates = re.findall(r'href="(\d{8})/"', html)
    
    if not dates:
        print("Error: Could not find any dump dates")
        sys.exit(1)
    
    # Sort and get the latest date
    dates.sort(reverse=True)
    latest = dates[0]
    print(f"Latest dump date: {latest}")
    
    return latest


def get_dump_status(date: str) -> dict:
    """Check the status of a specific dump."""
    status_url = f"{DUMP_BASE_URL}/{date}/dumpstatus.json"
    
    try:
        with urllib.request.urlopen(status_url, timeout=30) as response:
            import json
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.URLError:
        return None


def download_file(url: str, output_path: Path, resume: bool = True) -> bool:
    """
    Download a file with progress reporting and optional resume support.
    
    Args:
        url: URL to download
        output_path: Local path to save the file
        resume: Whether to resume partial downloads
        
    Returns:
        True if download successful, False otherwise
    """
    # Check for existing partial download
    start_byte = 0
    if resume and output_path.exists():
        start_byte = output_path.stat().st_size
        print(f"Resuming download from byte {start_byte:,}")
    
    # Create request with range header for resume
    request = urllib.request.Request(url)
    if start_byte > 0:
        request.add_header('Range', f'bytes={start_byte}-')
    
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            # Get total file size
            content_length = response.headers.get('Content-Length')
            if content_length:
                total_size = int(content_length) + start_byte
            else:
                total_size = None
            
            # Check if server supports range requests
            content_range = response.headers.get('Content-Range')
            if start_byte > 0 and not content_range:
                print("Server doesn't support resume, starting from beginning")
                start_byte = 0
            
            # Open file in appropriate mode
            mode = 'ab' if start_byte > 0 else 'wb'
            
            downloaded = start_byte
            chunk_size = 1024 * 1024  # 1MB chunks
            last_progress_time = datetime.now()
            
            with open(output_path, mode) as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Print progress every second
                    now = datetime.now()
                    if (now - last_progress_time).seconds >= 1:
                        if total_size:
                            progress = downloaded / total_size * 100
                            print(f"\rProgress: {downloaded:,} / {total_size:,} bytes ({progress:.1f}%)", end='', flush=True)
                        else:
                            print(f"\rDownloaded: {downloaded:,} bytes", end='', flush=True)
                        last_progress_time = now
            
            print()  # New line after progress
            return True
            
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code} {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"URL Error: {e}")
        return False
    except KeyboardInterrupt:
        print("\nDownload interrupted. Run again to resume.")
        return False


def verify_md5(filepath: Path, expected_md5: str) -> bool:
    """Verify file MD5 checksum."""
    print(f"Verifying MD5 checksum for {filepath.name}...")
    
    md5_hash = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            md5_hash.update(chunk)
    
    actual_md5 = md5_hash.hexdigest()
    
    if actual_md5 == expected_md5:
        print(f"MD5 verified: {actual_md5}")
        return True
    else:
        print(f"MD5 mismatch! Expected: {expected_md5}, Got: {actual_md5}")
        return False


def download_wikipedia(
    output_dir: str = "data",
    dump_type: str = "articles",
    dump_date: str = None,
    include_index: bool = True,
    verify: bool = True,
) -> None:
    """
    Download English Wikipedia dump.
    
    Args:
        output_dir: Directory to save downloaded files
        dump_type: Type of dump to download (articles, abstracts, titles, meta-current)
        dump_date: Specific dump date (YYYYMMDD) or None for latest
        include_index: Whether to also download the index file (for multistream)
        verify: Whether to verify MD5 checksums after download
    """
    if dump_type not in DUMP_TYPES:
        print(f"Unknown dump type: {dump_type}")
        print(f"Available types: {', '.join(DUMP_TYPES.keys())}")
        sys.exit(1)
    
    dump_info = DUMP_TYPES[dump_type]
    print(f"Dump type: {dump_type}")
    print(f"Description: {dump_info['description']}")
    
    # Get dump date
    if dump_date is None:
        dump_date = get_latest_dump_date()
    
    # Check dump status
    status = get_dump_status(dump_date)
    if status:
        jobs = status.get('jobs', {})
        articles_status = jobs.get('articlesmultistreamdump', {}).get('status', 'unknown')
        print(f"Dump status: {articles_status}")
        if articles_status != 'done':
            print("Warning: Dump may not be complete yet")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build file URLs
    filename = dump_info['filename'].format(date=dump_date)
    file_url = f"{DUMP_BASE_URL}/{dump_date}/{filename}"
    local_path = output_path / filename
    
    print(f"\nDownloading: {filename}")
    print(f"URL: {file_url}")
    print(f"Output: {local_path}")
    
    # Download main file
    success = download_file(file_url, local_path)
    
    if not success:
        print("Download failed!")
        sys.exit(1)
    
    print(f"Downloaded: {local_path}")
    print(f"Size: {local_path.stat().st_size:,} bytes")
    
    # Download index file if requested and available
    if include_index and dump_info.get('index'):
        index_filename = dump_info['index'].format(date=dump_date)
        index_url = f"{DUMP_BASE_URL}/{dump_date}/{index_filename}"
        index_path = output_path / index_filename
        
        print(f"\nDownloading index: {index_filename}")
        download_file(index_url, index_path)
    
    # Verify checksums if requested
    if verify:
        md5_url = f"{DUMP_BASE_URL}/{dump_date}/{filename}-md5"
        try:
            with urllib.request.urlopen(md5_url, timeout=30) as response:
                expected_md5 = response.read().decode('utf-8').split()[0]
                verify_md5(local_path, expected_md5)
        except urllib.error.URLError:
            print("Could not fetch MD5 checksum for verification")
    
    print("\nDownload complete!")
    print(f"\nTo extract the dump, use: bunzip2 -k {local_path}")
    print("Note: Extraction will require ~100GB of disk space for the full articles dump")


def main():
    parser = argparse.ArgumentParser(
        description="Download English Wikipedia database dump",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Download latest articles dump
  %(prog)s --type abstracts          # Download abstracts only
  %(prog)s --type titles             # Download article titles only
  %(prog)s --date 20260101           # Download specific date
  %(prog)s --output ./wikipedia      # Custom output directory

Reference: https://en.wikipedia.org/wiki/Wikipedia:Database_download
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        default='data',
        help='Output directory (default: data)'
    )
    
    parser.add_argument(
        '--type', '-t',
        choices=list(DUMP_TYPES.keys()),
        default='articles',
        help='Type of dump to download (default: articles)'
    )
    
    parser.add_argument(
        '--date', '-d',
        help='Dump date in YYYYMMDD format (default: latest)'
    )
    
    parser.add_argument(
        '--no-index',
        action='store_true',
        help='Skip downloading the index file'
    )
    
    parser.add_argument(
        '--no-verify',
        action='store_true',
        help='Skip MD5 verification'
    )
    
    parser.add_argument(
        '--list-dates',
        action='store_true',
        help='List available dump dates and exit'
    )
    
    args = parser.parse_args()
    
    if args.list_dates:
        url = f"{DUMP_BASE_URL}/"
        print(f"Fetching available dumps from {url}...")
        with urllib.request.urlopen(url, timeout=30) as response:
            html = response.read().decode('utf-8')
        import re
        dates = re.findall(r'href="(\d{8})/"', html)
        dates.sort(reverse=True)
        print("\nAvailable dump dates (most recent first):")
        for date in dates[:10]:
            print(f"  {date}")
        return
    
    download_wikipedia(
        output_dir=args.output,
        dump_type=args.type,
        dump_date=args.date,
        include_index=not args.no_index,
        verify=not args.no_verify,
    )


if __name__ == '__main__':
    main()

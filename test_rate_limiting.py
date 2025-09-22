#!/usr/bin/env python3
"""
Test script to verify rate limiting improvements work correctly.
This script simulates the burst request pattern that was causing the 429 errors.
"""

import os
import sys
import time
import logging
from dotenv import load_dotenv

# Add the project root to path
sys.path.insert(0, '.')

# Load environment
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_github_models_rate_limiting():
    """Test GitHub Models provider with rate limiting."""
    from rag.llm_providers import GitHubModelsProvider
    
    provider = GitHubModelsProvider()
    
    if not provider.is_available():
        logger.error("GitHub Models provider not available - check GITHUB_TOKEN")
        return False
    
    logger.info("Testing GitHub Models rate limiting...")
    
    # Test messages - simple requests that should work
    test_messages = [
        [{"role": "user", "content": "Say 'Test 1' in Spanish"}],
        [{"role": "user", "content": "Say 'Test 2' in Spanish"}],
        [{"role": "user", "content": "Say 'Test 3' in Spanish"}],
        [{"role": "user", "content": "Say 'Test 4' in Spanish"}],
        [{"role": "user", "content": "Say 'Test 5' in Spanish"}],
    ]
    
    success_count = 0
    start_time = time.time()
    
    for i, messages in enumerate(test_messages, 1):
        try:
            logger.info(f"Making request {i}/5...")
            response = provider.generate_response(
                messages=messages,
                temperature=0.1,
                max_tokens=50
            )
            logger.info(f"Request {i} successful: {response[:50]}...")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Request {i} failed: {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"Completed {success_count}/5 requests in {elapsed:.1f}s")
    
    return success_count >= 4  # Allow for 1 failure

def test_llm_manager_fallback():
    """Test LLM Manager fallback behavior."""
    from rag.llm_providers import LLMManager
    
    manager = LLMManager()
    
    logger.info("Testing LLM Manager fallback...")
    
    try:
        # Make a request that should work with fallback if needed
        response = manager.generate_response(
            messages=[{"role": "user", "content": "Say 'Fallback test successful' in Spanish"}],
            temperature=0.1,
            max_tokens=50
        )
        logger.info(f"LLM Manager test successful: {response[:50]}...")
        return True
        
    except Exception as e:
        logger.error(f"LLM Manager test failed: {e}")
        return False

def test_simulated_burst():
    """Simulate the burst pattern that was causing issues."""
    from rag.llm_providers import LLMManager
    
    manager = LLMManager()
    
    logger.info("Testing simulated request burst (similar to adaptive analysis)...")
    
    # Simulate multiple quick requests like in adaptive analysis
    requests = [
        "Analyze friendship patterns",
        "Analyze emotional support",
        "Analyze romantic indicators", 
        "Analyze family dynamics",
        "Generate summary"
    ]
    
    success_count = 0
    start_time = time.time()
    
    for i, request in enumerate(requests, 1):
        try:
            logger.info(f"Burst request {i}/5: {request}")
            response = manager.generate_response(
                messages=[{"role": "user", "content": f"Briefly {request.lower()}: 'Analysis complete'"}],
                temperature=0.1,
                max_tokens=30
            )
            logger.info(f"Burst request {i} successful")
            success_count += 1
            
        except Exception as e:
            logger.error(f"Burst request {i} failed: {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"Burst test: {success_count}/5 requests in {elapsed:.1f}s")
    
    return success_count >= 4

def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("RATE LIMITING FIX VERIFICATION")
    logger.info("=" * 60)
    
    results = {}
    
    # Test 1: Basic rate limiting
    logger.info("\n1. Testing GitHub Models rate limiting...")
    results['rate_limiting'] = test_github_models_rate_limiting()
    
    # Test 2: LLM Manager fallback
    logger.info("\n2. Testing LLM Manager fallback...")
    results['fallback'] = test_llm_manager_fallback()
    
    # Test 3: Simulated burst requests
    logger.info("\n3. Testing simulated request burst...")
    results['burst'] = test_simulated_burst()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS:")
    logger.info("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name.upper()}: {status}")
    
    overall_success = all(results.values())
    
    if overall_success:
        logger.info("\n🎉 ALL TESTS PASSED! Rate limiting fix is working.")
        return 0
    else:
        logger.info("\n⚠️  Some tests failed. Check logs above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
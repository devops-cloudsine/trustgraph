#!/usr/bin/env python3
"""
Test script for vLLM vision endpoint.
Sends an image to the vLLM API and prints the response.

Usage:
    python test_vllm_endpoint.py [image_path] [--url URL] [--model MODEL]

Examples:
    python test_vllm_endpoint.py /root/files_to_parse/doc-1/page_1.png
    python test_vllm_endpoint.py ./test.png --url http://localhost:8000/v1/chat/completions
    python test_vllm_endpoint.py ./test.png --model Qwen/Qwen2-VL-7B-Instruct
"""

import argparse
import base64
import json
import requests
import sys
from pathlib import Path


def describe_image_with_vllm(
    image_path: str,
    vllm_api_url: str = "http://vllm:8000/v1/chat/completions",
    vllm_model: str = "Qwen/Qwen3-VL-4B-Instruct",
    prompt: str = "Describe this image.",
    use_base64: bool = False,
    timeout: int = 120,
) -> dict:
    """
    ---> CLI/test > [describe_image_with_vllm] > POST vLLM endpoint, return raw response
    
    Send an image to vLLM and get a description.
    
    Args:
        image_path: Path to the image file
        vllm_api_url: vLLM API endpoint URL
        vllm_model: Model name to use
        prompt: Text prompt to send with the image
        use_base64: If True, encode image as base64 data URL instead of file:// path
        timeout: Request timeout in seconds
    
    Returns:
        dict with 'success', 'description', 'raw_response', and 'error' keys
    """
    result = {
        "success": False,
        "description": None,
        "raw_response": None,
        "error": None,
    }
    
    # Build image URL (file path or base64)
    if use_base64:
        try:
            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            # Detect image type from extension
            ext = Path(image_path).suffix.lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}
            mime_type = mime_map.get(ext, "image/png")
            image_url = f"data:{mime_type};base64,{img_data}"
        except Exception as e:
            result["error"] = f"Failed to read/encode image: {e}"
            return result
    else:
        # Use file:// path (requires vLLM to have access to this path)
        image_url = f"file://{image_path}"
    
    # NOTE: Do NOT include "response_format" parameter for plain text responses.
    # vLLM v0.11.0+ (V1 engine) has a bug where any response_format (even {"type": "text"})
    # triggers structured output validation that fails with:
    # "ValueError: No valid structured output parameter found"
    # Plain text is the default, so omitting response_format works correctly.
    payload = {
        "model": vllm_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            }
        ],
    }
    
    print("=" * 60)
    print("REQUEST DETAILS")
    print("=" * 60)
    print(f"URL: {vllm_api_url}")
    print(f"Model: {vllm_model}")
    print(f"Image: {image_path}")
    print(f"Image URL type: {'base64 data URL' if use_base64 else 'file:// path'}")
    print(f"Prompt: {prompt}")
    print()
    print("Payload (image content truncated if base64):")
    # Print payload but truncate base64 data for readability
    payload_display = json.loads(json.dumps(payload))
    if use_base64:
        try:
            img_url = payload_display["messages"][0]["content"][1]["image_url"]["url"]
            if img_url.startswith("data:"):
                payload_display["messages"][0]["content"][1]["image_url"]["url"] = img_url[:80] + "...[truncated]"
        except (KeyError, IndexError):
            pass
    print(json.dumps(payload_display, indent=2))
    print("=" * 60)
    
    try:
        print(f"\nSending request to {vllm_api_url}...")
        resp = requests.post(vllm_api_url, json=payload, timeout=timeout)
        
        print(f"Response Status: {resp.status_code}")
        print(f"Response Headers: {dict(resp.headers)}")
        print()
        
        result["raw_response"] = resp.text
        
        if resp.status_code != 200:
            result["error"] = f"HTTP {resp.status_code}: {resp.text}"
            return result
        
        data = resp.json()
        print("=" * 60)
        print("RAW JSON RESPONSE")
        print("=" * 60)
        print(json.dumps(data, indent=2, ensure_ascii=False))
        print("=" * 60)
        
        # Parse vLLM OpenAI-compatible response
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        content = msg.get("content")
        
        if isinstance(content, str):
            result["description"] = content.strip()
        elif isinstance(content, list):
            texts = [c.get("text", "") for c in content if c.get("type") == "text"]
            result["description"] = "\n".join([t for t in texts if t]).strip() or "No description returned."
        else:
            result["description"] = "No description returned."
        
        result["success"] = True
        
    except requests.exceptions.Timeout:
        result["error"] = f"Request timed out after {timeout} seconds"
    except requests.exceptions.ConnectionError as e:
        result["error"] = f"Connection error: {e}"
    except json.JSONDecodeError as e:
        result["error"] = f"Invalid JSON response: {e}"
    except Exception as e:
        result["error"] = f"Unexpected error: {type(e).__name__}: {e}"
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Test vLLM vision endpoint with an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default="/root/files_to_parse/doc-1/page_1.png",
        help="Path to the image file (default: /root/files_to_parse/doc-1/page_1.png)",
    )
    parser.add_argument(
        "--url",
        default="http://vllm:8000/v1/chat/completions",
        help="vLLM API URL (default: http://vllm:8000/v1/chat/completions)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-4B-Instruct",
        help="Model name (default: Qwen/Qwen3-VL-4B-Instruct)",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this image.",
        help="Prompt to send with the image (default: 'Describe this image.')",
    )
    parser.add_argument(
        "--base64",
        action="store_true",
        help="Send image as base64 data URL instead of file:// path",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)",
    )
    
    args = parser.parse_args()
    
    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"ERROR: Image file not found: {args.image_path}")
        print("Please provide a valid image path.")
        sys.exit(1)
    
    print()
    print("*" * 60)
    print("  vLLM Vision Endpoint Test")
    print("*" * 60)
    print()
    
    result = describe_image_with_vllm(
        image_path=args.image_path,
        vllm_api_url=args.url,
        vllm_model=args.model,
        prompt=args.prompt,
        use_base64=args.base64,
        timeout=args.timeout,
    )
    
    print()
    print("=" * 60)
    print("RESULT SUMMARY")
    print("=" * 60)
    
    if result["success"]:
        print("Status: SUCCESS ✓")
        print()
        print("Description:")
        print("-" * 40)
        print(result["description"])
        print("-" * 40)
    else:
        print("Status: FAILED ✗")
        print()
        print(f"Error: {result['error']}")
        if result["raw_response"]:
            print()
            print("Raw response:")
            print(result["raw_response"][:1000])
        sys.exit(1)


if __name__ == "__main__":
    main()

